
import jax
import jax.numpy as jnp
import optax

import flax.nnx as nnx
import chex


from functools import partial

from envs.jax_game import JaxGame
from nash_dreamer.ma_rssm import *
from nash_dreamer.train_utils import *
from nash_dreamer.distributions import get_bin_log_prob
#The MMD loss pieces are shared with the standalone learner. Importing them from
# there mirrors how rnad_dreamer imports v_trace/neurd_loss from sim_rnad.
from nash_dreamer.sim_mmd import clipped_surrogate, proximal_kl, magnet_kl, normalized_advantage
#TD(lambda) without importance sampling. td_estimate(...) - v is exactly GAE(lambda).
from nash_dreamer.dreamer_actor_critic import td_estimate


class MMDDreamer():
  """Magnetic Mirror Descent with a magnet fixed at the uniform policy,
  made compatible with the Dreamer framework so that it can be trained on
  imagined trajectories as well as real ones. See sim_mmd.SimMMD for the
  standalone version and the description of the loss.

  Unlike RNaD this learner carries no importance sampling corrections. It is
  on-policy, and it takes the behaviour policy stored in the timestep as pi_old
  for both the PPO ratio and the explicit proximal KL term. For imagined
  trajectories that stored policy is the policy of the actor at imagination
  time, provided the imagination uniform mixture is zero."""
  def __init__(self, game: JaxGame, config: MMDConfig, full_optimizer: nnx.Optimizer, target_optimizer: nnx.Optimizer) -> None:
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
      #Unlike World model, we operate with rewards defined
      # as (state, action, next_state) and only care how to
      # act in non-terminal states, hence we end one turn before terminal
      num_starts = game.max_trajectory_lenght_no_chance()
    self.num_starts = num_starts

    #One learner step collects one batch and takes num_epochs inner gradient
    # steps on it per loss term, which is what makes the ratio drift away from 1
    # and hence makes the clipping and the proximal KL do anything at all.
    self.num_epochs = max(1, self.config.num_epochs)
    self.learner_steps = 0
    self.gradient_steps = 0

    #The imagined behaviour policy is pi_old. If the imagination mixes in uniform
    # exploration then it is not the policy of the actor itself, and the ratio is
    # no longer 1 at the first inner epoch. The buffer side of this check lives in
    # train/joint_train.py, which is where the buffer config is in scope.
    if self.config.sampling_epsilon > 0:
      print(f"Warning! MMD is on-policy, but the imagination sampling_epsilon={self.config.sampling_epsilon} > 0. "
            f"The imagined behaviour policy is an epsilon-uniform mixture, so pi_old is not the policy of the "
            f"actor at imagination time and the policy ratio is not 1 at the first inner epoch. "
            f"Supply --img_sampling_epsilon 0 for strictly on-policy MMD.")

    self.metrics_keys = ['img_val', 'img_policy', 'real_val']
    if self.config.train_real_policy:
      self.metrics_keys.append('real_policy')
    #MMD diagnostics, reported for the imagined batch where the policy is always trained
    self.metrics_keys += ['kl_old', 'magnet', 'clip_frac']
    self.metrics = {k: 0 for k in self.metrics_keys}
    self.grad_norms = {'img': {}, 'real': {}}
    self.network_keys = (*ma_rssm.network_names[-2:], )



  @partial(nnx.jit, static_argnums=(0, 6))
  def update_parameters_and_model(
    self,
    optimizer: nnx.Optimizer,
    target_optimizer: nnx.Optimizer,
    trajectory_key,
    wm_timestep: TimeStep,
    wm_prediction_step: PredictionStepWithLegal,
    imagine: bool
  ):
    """Compute the MMD loss and use it to perform a gradient step of both
    the actor-critic and the Dreamer networks."""

    bins = jnp.arange((2 * self.config.bin_range) + 1) - self.config.bin_range
    # Per player vmap
    per_player_net_apply = nnx.vmap(MARSSM.call_net, in_axes=(None, 0, 0), out_axes=(0))
    #Per trajectory and batch dimensions
    vectorized_net_apply = nnx.vmap(nnx.vmap(per_player_net_apply, in_axes=(None, 0, 0), out_axes=(0)), in_axes=(None, 0, 0), out_axes=(0))
    #Critic is centralized
    vectorized_critic_apply = nnx.vmap(nnx.vmap(MARSSM.call_net, in_axes=(None, 0), out_axes=(0)), in_axes=(None, 0), out_axes=(0))

    def get_obs(timestep: ActorCriticTimeStep):
      #If the timestep contains real environment infosets,
      # transform them with symlog first. Latent infosets are used as they are.
      obs = symlog(timestep.obs) if self.use_real_infoset else timestep.obs
      return obs, jnp.reshape(obs, (*obs.shape[:-2], -1))

    def fixed_targets(target_network: CriticNetwork, timestep: ActorCriticTimeStep):
      """The value target and the advantages for one batch, computed once per
      learner step from the pre-update critic and held fixed across the inner
      epochs. That is what makes the K epochs a well defined mirror descent
      iteration rather than K chases of a moving target."""
      _, joint_obs = get_obs(timestep)
      expanded_valid = jnp.expand_dims(timestep.valid, -1)
      v_target_dist_logits = vectorized_critic_apply(target_network, joint_obs)
      v_old = get_value_from_bins(v_target_dist_logits, self.config.bin_range, use_symexp=False)

      v_train_target = td_estimate(v_old, expanded_valid, timestep.reward,
                                   self.config.td_lambda, self.config.gamma)
      #td_estimate returns v + sum (gamma * lambda)^k delta_{t+k},
      # so this difference is exactly the GAE(lambda) advantage.
      p1_advantage = normalized_advantage(v_train_target - v_old, expanded_valid, self.config.adv_norm_eps)
      #The advantage is from the perspective of player 1, the game is zero-sum
      #[Trajectory, Batch, Player, 1]
      advantage = jnp.stack([p1_advantage, -p1_advantage], axis=-2)
      return jax.lax.stop_gradient(v_train_target), jax.lax.stop_gradient(advantage)

    def mmd_loss(
      timestep: ActorCriticTimeStep,
      actor_network: ActorNetwork,
      critic_network: CriticNetwork,
      v_train_target: chex.Array,
      advantage: chex.Array,
      compute_actor_loss=True
    ):
      obs, joint_obs = get_obs(timestep)
      expanded_valid = jnp.expand_dims(timestep.valid, -1)
      player_valid = jnp.expand_dims(timestep.valid, (-1, -2))

      pi, log_pi, logit = vectorized_net_apply(actor_network, obs, timestep.legal)
      v_dist_logits = vectorized_critic_apply(critic_network, joint_obs)

      #The bin categorical loss
      v_loss = -get_bin_log_prob(v_dist_logits, bins, jax.lax.stop_gradient(v_train_target))
      v_loss_value = get_loss_mean_with_mask(v_loss, expanded_valid)

      if compute_actor_loss:
        #Watch out! Do not call legal_log_policy here, as
        # it assumes a logit and not a softmaxed policy, so we get
        # different results
        old_policy_mask = (timestep.policy <= 1e-8)
        log_pi_old = jnp.log(timestep.policy + old_policy_mask)
        log_pi_old = (1 - old_policy_mask) * log_pi_old

        ratio = policy_ratio(pi, timestep.policy, timestep.action, player_valid)
        surrogate = clipped_surrogate(ratio, advantage, self.config.clip_epsilon)
        kl_old = proximal_kl(pi, log_pi, log_pi_old)
        kl_magnet = magnet_kl(pi, log_pi, timestep.legal)

        policy_objective = surrogate - self.config.kl_coeff * kl_old - self.config.magnet_coeff * kl_magnet
        #Each player acts, so the normalization should be multiplied by 2 to get
        # the true mean. The multiplication by -1 is critical here, otherwise we
        # would be minimizing the policy objective, but we want to maximize it.
        policy_loss_value = -get_loss_mean_with_mask(policy_objective, player_valid, normalization_mult=2)
        diagnostics = (get_loss_mean_with_mask(kl_old, player_valid, normalization_mult=2),
                       get_loss_mean_with_mask(kl_magnet, player_valid, normalization_mult=2),
                       get_loss_mean_with_mask((jnp.abs(ratio - 1.0) > self.config.clip_epsilon).astype(v_loss.dtype),
                                               player_valid, normalization_mult=2))
      else:
        policy_loss_value = 0
        diagnostics = (0, 0, 0)

      return v_loss_value + policy_loss_value, v_loss_value, policy_loss_value, diagnostics

    def imagination_loss(model: MARSSM,
      timestep: ActorCriticTimeStep,
      v_train_target: chex.Array,
      advantage: chex.Array,
      beta_imagination: float):
        loss_val, v_loss, p_loss, diagnostics = mmd_loss(timestep, model.actor, model.critic,
                                                         v_train_target, advantage)
        metrics = {'img_val': beta_imagination * v_loss, 'img_policy': beta_imagination * p_loss}
        kl_old, kl_magnet, clip_frac = diagnostics
        metrics.update({'kl_old': kl_old, 'magnet': kl_magnet, 'clip_frac': clip_frac})
        return beta_imagination * loss_val, metrics

    def real_loss(model: MARSSM,
      timestep: ActorCriticTimeStep,
      v_train_target: chex.Array,
      advantage: chex.Array,
      beta_real: float):
        loss_val, v_loss, p_loss, _ = mmd_loss(timestep, model.actor, model.critic,
                                               v_train_target, advantage,
                                               compute_actor_loss=self.config.train_real_policy)
        metrics = {'real_val': beta_real * v_loss}
        if self.config.train_real_policy:
          metrics['real_policy'] = beta_real * p_loss
        return beta_real * loss_val, metrics

    real_timestep = wm_timestep_to_timestep(wm_timestep, wm_prediction_step, self.use_real_infoset)
    real_target, real_advantage = fixed_targets(target_optimizer.model, real_timestep)

    if imagine:
      #Start imagination from the root and collapse the first two dimensions into num_starts * batch
      starting_points = jax.tree.map(lambda x: jnp.repeat(x[0][None, ...], self.num_starts, axis=0).reshape((-1, *x.shape[2:])), wm_prediction_step)
      #Imagine ONCE, outside of value_and_grad. This is the one deliberate departure
      # from rnad_dreamer, which imagines inside the loss. If we re-imagined inside
      # each of the inner epochs, the stored behaviour policy would always equal the
      # current policy, the ratio would be identically 1, the clipping a no-op and the
      # proximal KL exactly zero with zero gradient, which silently reduces MMD to
      # REINFORCE with an entropy bonus.
      imagined_timestep = jax.lax.stop_gradient(optimizer.model.imagine_trajectories(trajectory_key, starting_points))
      img_target, img_advantage = fixed_targets(target_optimizer.model, imagined_timestep)

    summed_metrics = None
    igrad = {k: 0 for k in self.network_keys}
    for _ in range(self.num_epochs):
      epoch_metrics = {}
      if imagine:
        img_return, igrad = nnx.value_and_grad(imagination_loss, argnums=(0), has_aux=True)(
          optimizer.model,
          imagined_timestep,
          img_target,
          img_advantage,
          self.config.beta_imagination)
        _, img_metrics = img_return
        optimizer.update(igrad)
        epoch_metrics.update(img_metrics)
      else:
        epoch_metrics.update({'img_val': 0, 'img_policy': 0, 'kl_old': 0, 'magnet': 0, 'clip_frac': 0})

      r_return, rgrad = nnx.value_and_grad(real_loss, argnums=(0), has_aux=True)(
        optimizer.model,
        real_timestep,
        real_target,
        real_advantage,
        self.config.beta_real)
      _, r_metrics = r_return
      optimizer.update(rgrad)
      epoch_metrics.update(r_metrics)

      summed_metrics = epoch_metrics if summed_metrics is None else jax.tree.map(jnp.add, summed_metrics, epoch_metrics)

    #Report the metrics averaged over the inner epochs, but the gradient norms
    # of the last one, since those are what the optimizer state ends up reflecting.
    metrics = jax.tree.map(lambda x: x / self.num_epochs, summed_metrics)

    grad_norms = self.grad_norms.copy()
    if self.config.report_gradnorms:
      grad_keys = self.grad_norms.keys()
      grads = (igrad, rgrad)
      for k, g in zip(grad_keys, grads):
        for n in self.network_keys:
          grad_norms[k][n] = optax.tree.norm(g[n], ord=2)

    critic_state = nnx.state(optimizer.model.critic)
    state_target = nnx.state(target_optimizer.model)

    #This grad coupled with vanilla SGD optimizer
    # is equivalent to the EMA formula (1 - alpha) * state_target + alpha * state
    # Which, in turn corresponds to TD learning
    target_grad = jax.tree.map(lambda a, b: a - b, state_target, critic_state)
    target_optimizer.update(target_grad)

    return metrics, grad_norms



  def step(self, wm_timestep: TimeStep, wm_prediction_step:PredictionStepWithLegal, trajectory_key: chex.Array):
    should_imagine = self.learner_steps >= self.config.wm_warm_up_period
    self.metrics, self.grad_norms = self.update_parameters_and_model(self.optimizer, self.target_optimizer,
                                                                    trajectory_key, wm_timestep, wm_prediction_step,
                                                                    should_imagine)
    self.learner_steps += 1
    #Each inner epoch takes one imagined and one real gradient step
    self.gradient_steps += 2 * self.num_epochs

  def getstate(self):
      return {'learner_steps': self.learner_steps,
              'gradient_steps': self.gradient_steps,
              'target_optimizer': nnx.state(self.target_optimizer),
              }

  def setstate(self, state):
    self.learner_steps = state['learner_steps']
    self.gradient_steps = state['gradient_steps']
    nnx.update(self.target_optimizer, state['target_optimizer'])
