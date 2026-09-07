"""RNaD actor-critic for the decentralized world model.

The only structural change from `RNaDDreamer` is that the critic is
decentralized too: instead of ONE centralized value head reading the flattened
joint infoset and deriving player 2's value as its negation, each player has its
own value head reading only its own infoset.

That forces a variant of `v_trace`, because the original hardcodes the
zero-sum-symmetric critic at `sim_rnad.py:231` (`jnp.stack([v, -v], axis=-2)`).
Everything else -- the NeuRD policy loss, the counterfactual importance
sampling, the target-critic EMA and the prev/_prev policy shift register -- is
reused unchanged.
"""

import jax
import jax.numpy as jnp
import jax.lax as lax
import optax

import flax.nnx as nnx
import chex

from typing import Any
from functools import partial

from games.jax_game import JaxGame
from ma_rssm import MARSSM
from ma_rssm_decentralized import DecentralizedPredictionStep
from train_utils import (RNaDConfig, TimeStep, ActorCriticTimeStep, symlog, tree_where,
                         get_value_from_bins, get_loss_mean_with_mask, policy_ratio,
                         wm_timestep_to_timestep)
from distributions import get_bin_log_prob
from sim_rnad import neurd_loss, EntropySchedule
from networks import ActorNetwork, CriticNetwork


def decentralized_v_trace(
  v: chex.Array,            # [Trajectory, Batch, Player, 1] -- per player, NOT (v, -v)
  valid: chex.Array,
  sampling_policy: chex.Array,
  network_policy: chex.Array,
  regularization_term: chex.Array,
  action_oh: chex.Array,
  reward: chex.Array, # Still not regularized
  lambda_: float = 1.0, # Lambda parameter for V-trace
  c: float = jnp.inf, # Importance sampling clipping
  rho: float = jnp.inf, # Importance sampling clipping
  eta: float = 0.2, # Regularization factor for reward transformation
  gamma: float = 1.0 # Discount factor
):
  """V-trace for a per-player critic.

  Differs from sim_rnad.v_trace in exactly four places, all of which reuse
  quantities the original already computes:
    1. per_player_v is `v` itself rather than jnp.stack([v, -v]).
    2. the bootstrap reward is `q_reward`, which is already the per-player
       analogue of the scalar `entropy_reward`.
    3. the joint importance-sampling products get an extra axis to broadcast
       against [.., Player, 1].
    4. q_term is formed per player instead of by negate-and-stack.
  """

  importance_sampling = policy_ratio(network_policy, sampling_policy, action_oh, valid)

  # The reason we use this is to ensure this is weighted by the amount of the times we sample it
  inverted_sampling = policy_ratio(jnp.ones_like(sampling_policy), sampling_policy, action_oh, valid)

  #[Trajectory, Batch, Player]
  #This actually computes KL-divergence from the reference policy, despite being called entropy.
  regularization_entropy = eta * jnp.sum(network_policy * regularization_term, axis=-1)
  weighted_regularization_term = -eta * regularization_term

  #[Trajectory, Batch, Player]
  # Adding the opponent's KL divergence and subtracting our own, per player.
  # With a centralized critic this had to be split into a scalar `entropy_reward`
  # for the value recursion and a stacked `q_reward` for the Q term; with a
  # per-player critic both roles are served by the same per-player quantity.
  q_reward = jnp.stack((reward, -reward), axis=-1) + regularization_entropy[..., (1, 0)]
  #[Trajectory, Batch, Player, 1]
  q_reward = jnp.expand_dims(q_reward, -1)

  @chex.dataclass(frozen=True)
  class VTraceCarry:
    next_value: chex.Array # Network value in the next timestep
    delta_v: chex.Array # Propagated delta V in V-trace from the next timestep

  init_carry = VTraceCarry(
    next_value=jnp.zeros_like(v[-1]),
    delta_v=jnp.zeros_like(v[-1])
  )

  def _v_trace(carry: VTraceCarry, x) -> tuple[VTraceCarry, Any]:
    (importance_sampling, v, q_reward, weighted_regularization_term, valid, inverted_sampling, action_oh) = x
    #Use the importance sampling for both players,
    # since it is a simultaneous move game. The transition still depends on
    # BOTH actions even though the critics are decentralized, so these stay
    # joint products; they only need an extra axis to broadcast per player.
    rho_is = jnp.minimum(rho, importance_sampling)
    rho_joint_is = jnp.prod(rho_is, axis=-2, keepdims=True)
    c_joint_is = jnp.prod(jnp.minimum(c, importance_sampling), axis=-2, keepdims=True)
    rho_inv_is = jnp.minimum(rho, inverted_sampling)

    delta_v = rho_joint_is * (q_reward + gamma * carry.next_value - v)
    carry_delta_v = delta_v + lambda_ * c_joint_is * gamma * carry.delta_v

    v_target = v + carry_delta_v

    #Already per player -- no stacking or negation.
    per_player_v = v
    q_term = (carry.next_value + carry.delta_v) - v

    # We use importance sampling of the opponent.
    opponent_sampling = jnp.flip(rho_is, -2)

    q_value = per_player_v + weighted_regularization_term + action_oh * opponent_sampling * rho_inv_is * (q_reward + gamma * q_term)

    next_carry = VTraceCarry(
      next_value=v,
      delta_v=carry_delta_v
    )

    reset_v_target = jnp.zeros_like(v_target)
    reset_q_value = jnp.zeros_like(q_value)

    reset_carry = init_carry
    #The centralized version squeezes the last axis here because its value is
    # [Batch, 1]; ours is [Batch, Player, 1], so the predicate is left at
    # [Batch, 1, 1] to broadcast over the player axis instead.
    next_carry, v_target = tree_where(valid, (next_carry, v_target), (reset_carry, reset_v_target))
    q_value = jnp.where(valid, q_value, reset_q_value)
    return (next_carry, (v_target, q_value))

  _, (v_target, q_value) = lax.scan(
    f=_v_trace,
    init=init_carry,
    xs=(importance_sampling, v, q_reward, weighted_regularization_term, valid, inverted_sampling, action_oh),
    reverse=True
  )
  return v_target, q_value


class DecentralizedRNaDDreamer():
  """RNaD on top of the decentralized world model, with a per-player critic."""

  def __init__(self, game: JaxGame, config: RNaDConfig, full_optimizer: nnx.Optimizer,
               target_optimizer: nnx.Optimizer) -> None:
    self.config = config
    self.optimizer = full_optimizer
    self.target_optimizer = target_optimizer
    self.init(game)

  def init(self, game: JaxGame):

    self.actions = game.num_distinct_actions()
    self.num_players = game.num_players()

    ma_rssm = self.optimizer.model
    self.use_real_infoset = ma_rssm.use_real_infoset
    self.input_size = ma_rssm.infoset_size

    num_starts = self.config.num_starts
    #If negative, take all for unroll, the Dreamer trajectories
    # will have one more timestep, hence + 1
    if num_starts <= 0:
      num_starts = game.max_trajectory_lenght_no_chance()
    self.num_starts = num_starts

    self.learner_steps = 0
    self.policy_switch_steps = 0

    self._entropy_schedule = EntropySchedule(
        sizes=self.config.entropy_schedule_size,
        repeats=self.config.entropy_schedule_repeats)

    rnad_graphdef, rnad_state = nnx.split(ma_rssm.actor)
    self.prev_network = nnx.merge(rnad_graphdef, rnad_state)
    self._prev_network = nnx.merge(rnad_graphdef, rnad_state)

    self.metrics_keys = ['img_val', 'img_policy', 'real_val']
    if self.config.train_real_policy:
      self.metrics_keys.append('real_policy')
    self.metrics = {k: 0 for k in self.metrics_keys}
    self.grad_norms = {'img': {}, 'real': {}}
    self.network_keys = (*ma_rssm.network_names[-2:], )

  @partial(nnx.jit, static_argnums=(0, 9))
  def update_parameters_and_model(
    self,
    optimizer: nnx.Optimizer,
    target_optimizer: nnx.Optimizer,
    prev_network: ActorNetwork,
    _prev_network: ActorNetwork,
    trajectory_key,
    wm_timestep: TimeStep,
    wm_prediction_step: DecentralizedPredictionStep,
    learner_steps: int,
    imagine: bool
  ):
    """Compute RNaD loss and use it to perform
    a gradient step of both RNaD and the decentralized Dreamer."""
    alpha, update_regularization = self._entropy_schedule(learner_steps)

    def rnad_loss(
      timestep: ActorCriticTimeStep,
      rnad_network: ActorNetwork,
      critic_network: CriticNetwork,
      target_network: CriticNetwork,
      prev_network: ActorNetwork,
      _prev_network: ActorNetwork,
      start_reaches_is: chex.Array,
      alpha: float,
      compute_actor_loss=True
    ):
      #If the timestep contains real environment infosets,
      # transform them with symlog first
      obs = symlog(timestep.obs) if self.use_real_infoset else timestep.obs
      bins = jnp.arange((2 * self.config.bin_range) + 1) - self.config.bin_range
      # Per player vmap
      per_player_net_apply = nnx.vmap(MARSSM.call_net, in_axes=(None, 0, 0), out_axes=(0))
      #Per trajectory and batch dimensions
      vectorized_net_apply = nnx.vmap(nnx.vmap(per_player_net_apply, in_axes=(None, 0, 0), out_axes=(0)), in_axes=(None, 0, 0), out_axes=(0))
      #Critic is DECENTRALIZED: one head per player, over its own infoset only.
      per_player_critic_apply = nnx.vmap(MARSSM.call_net, in_axes=(None, 0), out_axes=(0))
      vectorized_critic_apply = nnx.vmap(nnx.vmap(per_player_critic_apply, in_axes=(None, 0), out_axes=(0)), in_axes=(None, 0), out_axes=(0))

      pi, log_pi, logit = vectorized_net_apply(rnad_network, obs, timestep.legal)

      #[Trajectory, Batch, Player, 2 * bin_range + 1]
      v_dist_logits = vectorized_critic_apply(critic_network, obs)
      v_target_dist_logits = vectorized_critic_apply(target_network, obs)
      _, log_pi_prev, _ = vectorized_net_apply(prev_network, obs, timestep.legal)
      _, log_pi_prev_, _ = vectorized_net_apply(_prev_network, obs, timestep.legal)

      #[Trajectory, Batch, Player, 1]
      v_target = get_value_from_bins(v_target_dist_logits, self.config.bin_range, use_symexp=False)
      # This creates the regularization term for rewards
      regularized_term = log_pi - (alpha * log_pi_prev + (1 - alpha) * log_pi_prev_)

      expanded_valid = jnp.expand_dims(timestep.valid, (-1, -2))

      v_train_target, q_value = decentralized_v_trace(
        v_target, expanded_valid, timestep.policy, pi, regularized_term, timestep.action, timestep.reward,
        self.config.lambda_vtrace, self.config.c_vtrace, self.config.rho_vtrace,
        self.config.eta, self.config.gamma_vtrace)

      # We do not take into account the player reaches, since infoset is always reached with the same prob
      sampling_policy = jnp.sum(timestep.policy * timestep.action, axis=-1, keepdims=True) * expanded_valid + (1 - expanded_valid)
      network_policy = jnp.sum(pi * timestep.action, axis=-1, keepdims=True) * expanded_valid + (1 - expanded_valid)
      sampling_policy = jnp.prod(sampling_policy, axis=-2, keepdims=True)

      importance_sampling = network_policy / sampling_policy

      importance_sampling = jnp.concatenate((start_reaches_is, importance_sampling[:-1]), axis=0)
      importance_sampling = jnp.cumprod(importance_sampling, axis=0)

      #Flip to turn into counterfactual importance sampling
      importance_sampling = jnp.flip(importance_sampling, axis=-2)
      #Handle the importance sampling divergence, or if
      # it went to NaN (inf * 0 case)
      safe_cf_is = jnp.nan_to_num(
        importance_sampling,
        nan=0.0,               # If inf multiplied by 0, the reach is functionally dead
        posinf=self.config.cf_is_clip,  # Catch raw infinities and clamp them
        neginf=0.0             # Reaches cannot be negative, but good hygiene
        )

      # 2. Standard clip for the finite numbers that are just too large
      safe_cf_is = jnp.clip(safe_cf_is, 0.0, self.config.cf_is_clip)

      #Both the logits and the target now carry the player axis, and both
      # get_bin_log_prob and get_value_from_bins are shape generic.
      v_loss = -get_bin_log_prob(v_dist_logits, bins, jax.lax.stop_gradient(v_train_target))
      v_loss_value = get_loss_mean_with_mask(v_loss, expanded_valid, normalization_mult=2)

      if compute_actor_loss:
        loss_neurd = neurd_loss(logit, pi, q_value, timestep.legal, safe_cf_is,
                                self.config.neurd_clip, self.config.neurd_threshold)

        # The multiplication by -1 is critical here, otherwise we would
        # be minimizing the neurd term, but we want to maximize it.
        neurd_loss_value = -get_loss_mean_with_mask(loss_neurd, expanded_valid, normalization_mult=2)
      else:
        neurd_loss_value = 0
      return v_loss_value + neurd_loss_value, v_loss_value, neurd_loss_value

    def imagination_loss(model,
      target_network: CriticNetwork,
      prev_network: ActorNetwork,
      _prev_network: ActorNetwork,
      trajectory_key,
      starting_points: DecentralizedPredictionStep,
      start_reaches_is: chex.Array,
      alpha: float,
      beta_imagination: float):
        img_keys = self.metrics_keys[:2]
        if beta_imagination == 0.0:
          return 0.0, {k: 0 for k in img_keys}
        timestep = jax.lax.stop_gradient(model.imagine_trajectories(trajectory_key, starting_points))
        loss_val, v_loss, p_loss = rnad_loss(timestep, model.actor, model.critic, target_network, prev_network, _prev_network, start_reaches_is, alpha)
        losses = (v_loss, p_loss)
        metrics = {k: beta_imagination * v for k, v in zip(img_keys, losses)}
        return beta_imagination * loss_val, metrics

    def real_loss(model,
      target_network: CriticNetwork,
      prev_network: ActorNetwork,
      _prev_network: ActorNetwork,
      timestep: ActorCriticTimeStep,
      alpha: float,
      beta_real: float):
        start_reaches_is = jnp.ones((1, *timestep.policy.shape[1:-1], 1))
        loss_val, v_loss, p_loss = rnad_loss(timestep, model.actor, model.critic, target_network, prev_network, _prev_network,
                                         start_reaches_is, alpha, compute_actor_loss=self.config.train_real_policy)
        real_keys = self.metrics_keys[2:]
        losses = (v_loss, p_loss) if self.config.train_real_policy else (v_loss, )
        metrics = {k: beta_real * v for k, v in zip(real_keys, losses)}
        return beta_real * loss_val, metrics

    rnad_timestep = wm_timestep_to_timestep(wm_timestep, wm_prediction_step, self.use_real_infoset)

    #Start imagination from the root and collapse the first two dimensions into num_starts * batch
    starting_points = jax.tree.map(lambda x: jnp.repeat(x[0][None, ...], self.num_starts, axis=0).reshape((-1, *x.shape[2:])), wm_prediction_step)
    #Since we start at the root, the starting reaches are ones
    start_reaches_is = jnp.ones((1, self.num_starts * rnad_timestep.reward.shape[1], self.num_players, 1))

    if imagine:
      img_return, igrad = nnx.value_and_grad(imagination_loss, argnums=(0), has_aux=True)(
        optimizer.model,
        target_optimizer.model,
        prev_network,
        _prev_network,
        trajectory_key, starting_points, start_reaches_is, alpha, self.config.beta_imagination)

      img_loss, img_metrics = img_return
      optimizer.update(igrad)
    else:
      img_metrics = {'img_val': 0, 'img_policy': 0}
      igrad = {k: 0 for k in self.network_keys}
    r_return, rgrad = nnx.value_and_grad(real_loss, argnums=(0), has_aux=True)(
      optimizer.model,
      target_optimizer.model,
      prev_network,
      _prev_network,
      rnad_timestep,
      alpha,
      self.config.beta_real
    )

    grad_norms = self.grad_norms.copy()
    if self.config.report_gradnorms:
      grad_keys = self.grad_norms.keys()
      grads = (igrad, rgrad)
      for k, g in zip(grad_keys, grads):
        for n in self.network_keys:
          grad_norms[k][n] = optax.tree.norm(g[n], ord=2)
    r_loss, r_metrics = r_return
    optimizer.update(rgrad)

    critic_state = nnx.state(optimizer.model.critic)
    actor_graphdef, actor_state = nnx.split(optimizer.model.actor)
    state_target = nnx.state(target_optimizer.model)
    state_prev = nnx.state(prev_network)
    _state_prev = nnx.state(_prev_network)

    #This grad coupled with vanilla SGD optimizer
    # is equivalent to the EMA formula (1 - alpha) * state_target + alpha * state
    # Which, in turn corresponds to TD learning
    target_grad = jax.tree.map(lambda a, b: a - b, state_target, critic_state)
    target_optimizer.update(target_grad)

    img_metrics.update(r_metrics)
    state_prev, _state_prev = jax.lax.cond(
        update_regularization,
        lambda: (actor_state, state_prev),
        lambda: (state_prev, _state_prev))
    prev_network = nnx.merge(actor_graphdef, state_prev)
    _prev_network = nnx.merge(actor_graphdef, _state_prev)
    return prev_network, _prev_network, img_metrics, grad_norms, update_regularization

  def step(self, wm_timestep: TimeStep, wm_prediction_step: DecentralizedPredictionStep,
           trajectory_key: chex.Array):
    should_imagine = self.learner_steps >= self.config.wm_warm_up_period
    self.prev_network, self._prev_network, self.metrics, self.grad_norms, update_regularization = \
      self.update_parameters_and_model(self.optimizer, self.target_optimizer, self.prev_network,
                                       self._prev_network, trajectory_key, wm_timestep, wm_prediction_step,
                                       self.learner_steps, should_imagine)
    self.learner_steps += 1
    self.policy_switch_steps += int(update_regularization)

  def getstate(self):
      return {'learner_steps': self.learner_steps,
              'policy_steps': self.policy_switch_steps,
              'target_optimizer': nnx.state(self.target_optimizer),
              'prev_network': nnx.state(self.prev_network),
              '_prev_network': nnx.state(self._prev_network)
              }

  def setstate(self, state):
    self.learner_steps = state['learner_steps']
    nnx.update(self.target_optimizer, state['target_optimizer'])
    self.policy_switch_steps = state['policy_steps']
    nnx.update(self.prev_network, state['prev_network'])
    nnx.update(self._prev_network, state['_prev_network'])
