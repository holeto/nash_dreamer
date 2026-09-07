
import jax
import jax.numpy as jnp
import jax.lax as lax
import optax

import flax.nnx as nnx
import chex

import numpy as np


from functools import partial
from typing import Any

from envs.jax_game import JaxGame, GameState
from nash_dreamer.ma_rssm import *
from nash_dreamer.train_utils import *
from nash_dreamer.replay_buffer import ActorReplayBuffer
from nash_dreamer.distributions import get_bin_log_prob, get_normal_log_prob

from nash_dreamer.optimizer import make_opt


"""BEGGINING OF CODE FROM OpenSpiel RNaD"""
"""The Entropy schedule class taken from the
(now removed) RNaD implementation in the OpenSpiel library"""
class EntropySchedule:
  """An increasing list of steps where the regularisation network is updated.

  Example
    EntropySchedule([3, 5, 10], [2, 4, 1])
    =>   [0, 3, 6, 11, 16, 21, 26, 36]
          | 3 x2 |      5 x4     | 10 x1
  """

  def __init__(self, *, sizes: Sequence[int], repeats: Sequence[int]):
    """Constructs a schedule of entropy iterations.

    Args:
      sizes: the list of iteration sizes.
      repeats: the list, parallel to sizes, with the number of times for each
        size from `sizes` to repeat.
    """
    try:
      if len(repeats) != len(sizes):
        raise ValueError("`repeats` must be parallel to `sizes`.")
      if not sizes:
        raise ValueError("`sizes` and `repeats` must not be empty.")
      if any([(repeat <= 0) for repeat in repeats]):
        raise ValueError("All repeat values must be strictly positive")
      if repeats[-1] != 1:
        raise ValueError("The last value in `repeats` must be equal to 1, "
                         "ince the last iteration size is repeated forever.")
    except ValueError as e:
      raise ValueError(
          f"Entropy iteration schedule: repeats ({repeats}) and sizes"
          f" ({sizes})."
      ) from e

    schedule = [0]
    for size, repeat in zip(sizes, repeats):
      schedule.extend([schedule[-1] + (i + 1) * size for i in range(repeat)])

    self.schedule = np.array(schedule, dtype=np.int32)

  def __call__(self, learner_step: int) -> Tuple[float, bool]:
    """Entropy scheduling parameters for a given `learner_step`.

    Args:
      learner_step: The current learning step.

    Returns:
      alpha: The mixing weight (from [0, 1]) of the previous policy with
        the one before for computing the intrinsic reward.
      update_target_net: A boolean indicator for updating the target network
        with the current network.
    """

    # The complexity below is because at some point we might go past
    # the explicit schedule, and then we'd need to just use the last step
    # in the schedule and apply the logic of
    # ((learner_step - last_step) % last_iteration) == 0)

    # The schedule might look like this:
    # X----X-------X--X--X--X--------X
    # learner_step | might be here ^    |
    # or there     ^                    |
    # or even past the schedule         ^

    # We need to deal with two cases below.
    # Instead of going for the complicated conditional, let's just
    # compute both and then do the A * s + B * (1 - s) with s being a bool
    # selector between A and B.

    # 1. assume learner_step is past the schedule,
    #    ie schedule[-1] <= learner_step.
    last_size = self.schedule[-1] - self.schedule[-2]
    last_start = self.schedule[-1] + (
        learner_step - self.schedule[-1]) // last_size * last_size
    # 2. assume learner_step is within the schedule.
    start = jnp.amax(self.schedule * (self.schedule <= learner_step))
    finish = jnp.amin(
        self.schedule * (learner_step < self.schedule),
        initial=self.schedule[-1],
        where=(learner_step < self.schedule))
    size = finish - start

    # Now select between the two.
    beyond = (self.schedule[-1] <= learner_step)  # Are we past the schedule?
    iteration_start = (last_start * beyond + start * (1 - beyond))
    iteration_size = (last_size * beyond + size * (1 - beyond))

    update_target_net = jnp.logical_and(
        learner_step > 0,
        jnp.sum(learner_step == iteration_start + iteration_size - 1),
    )
    alpha = jnp.minimum(
        (2.0 * (learner_step - iteration_start)) / iteration_size, 1.0)
    return alpha, update_target_net
  
"""END OF CODE FROM OpenSpiel RNaD"""




def neurd_loss(
  logits: chex.Array,
  policy: chex.Array,
  q_values: chex.Array, 
  legal: chex.Array,
  importance_sampling: chex.Array,
  clip: float=10_000,
  threshold: float=2.0
):
  advantage = q_values - jnp.sum(policy * q_values, axis=-1, keepdims=True)
  advantage = advantage * importance_sampling
  advantage = lax.stop_gradient(jnp.clip(advantage, -clip, clip))
  mean_logit = jnp.sum(logits * legal, axis=-1, keepdims=True) / jnp.sum(legal, axis=-1, keepdims=True)
  
  logits_shifted = logits - mean_logit
  threshold_center = jnp.zeros_like(logits_shifted)
  
  neurd_loss_value = jnp.sum(legal * apply_force_with_threshold(logits_shifted, advantage, threshold, threshold_center), axis=-1, keepdims=True)

  return neurd_loss_value

def v_trace(
  v: chex.Array,
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
  
  importance_sampling = policy_ratio(network_policy, sampling_policy, action_oh, valid)
  #jax.debug.breakpoint()
  
  # The reason we use this is to ensure this is weighted by the amount of the times we sample it
  inverted_sampling = policy_ratio(jnp.ones_like(sampling_policy), sampling_policy, action_oh, valid)
  

  #[Trajectory, Batch, Player]
  #This actually computes KL-divergence from the reference policy, despite being called entropy.
  #The reason for being called "entropy", is because it serves simliar purpose.
  #Intuitive explanation below, but it is necessary to 
  # provide an unbiased estimate of counterfactual value as
  # detailed in https://arxiv.org/pdf/2501.16600
  regularization_entropy = eta * jnp.sum(network_policy * regularization_term, axis=-1)
  weighted_regularization_term = -eta * regularization_term
  
  #[Trajectory, Batch]
  #For value estimates. Adding opponents KL divergence
  # amounts to: the value of this state is higher, because the opponent
  # did something surprising, hence, the state should be explored more, to either
  # find out a possible opponent mistake, or confirm that the state is bad.
  # similarly, subtracting our own KL-divergence penalizes being too "surprising"
  # with regards to the reference policy, to discourage erratic policies
  both_player_entropy = (regularization_entropy[..., 1] - regularization_entropy[..., 0])
  #both_player_entropy = 0

  #[Trajectory, Batch]
  entropy_reward = reward + both_player_entropy
  #[Trajectory, Batch, 1]
  entropy_reward = jnp.expand_dims(entropy_reward, -1)
  
  #[Trajectory, Batch, Player]
  # Once again, similar logic. Adding opponents KL divergence
  # to give a bonus to the rewards if the opponent did something surprising there
  # to either confirm that it is not good for us, or take advantage of it.
  q_reward = jnp.stack((reward, -reward), axis=-1) + regularization_entropy[..., (1, 0)]
  
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
    (importance_sampling, v, q_reward, entropy_reward, weighted_regularization_term, valid, inverted_sampling, action_oh) = x 
    #Use the importance sampling for both players,
    # since it is a simultaneous move game
    rho_is = jnp.minimum(rho, importance_sampling)
    rho_joint_is = jnp.prod(rho_is, axis=-2)
    c_joint_is = jnp.prod(jnp.minimum(c, importance_sampling), axis=-2)
    rho_inv_is = jnp.minimum(rho, inverted_sampling)

    
    delta_v = rho_joint_is * (entropy_reward + gamma * carry.next_value - v)
    carry_delta_v = delta_v + lambda_ * c_joint_is * gamma * carry.delta_v
    
    v_target = v + carry_delta_v


    per_player_v = jnp.stack([v, -v], axis=-2)

    q_term = jnp.stack([(carry.next_value + carry.delta_v) - v, v - (carry.next_value + carry.delta_v)], axis=-2)
    
    
    # We use importance sampling of the opponent.
    opponent_sampling = jnp.flip(rho_is, -2)
    
    q_value = per_player_v + weighted_regularization_term  + action_oh * opponent_sampling * rho_inv_is  * (q_reward + gamma * q_term)
    
    next_carry = VTraceCarry(
      next_value=v,
      delta_v=carry_delta_v
    )
  
    reset_v_target = jnp.zeros_like(v_target)
    reset_q_value = jnp.zeros_like(q_value) 
    
    reset_carry = init_carry
    next_carry, v_target = tree_where(jnp.squeeze(valid, axis=-1), (next_carry, v_target), (reset_carry, reset_v_target))
    q_value = jnp.where(valid, q_value, reset_q_value)
    return (next_carry, (v_target, q_value))

    
    
    
  _, (v_target, q_value) = lax.scan(
    f=_v_trace,
    init=init_carry,
    xs=(importance_sampling, v, q_reward, entropy_reward, weighted_regularization_term, valid, inverted_sampling, action_oh),
    reverse=True
  )
  return v_target, q_value

  


class ACNetwork(nnx.Module):
  """A container module, that wraps the actor
  and critic into a single class, to have one optimizer for both.
  Shared by all the standalone actor-critic learners, it only reads the
  network detail and bin range fields that all their configs carry."""

  def __init__(self, game: JaxGame, config: RNaDConfig | MMDConfig, rngs: nnx.Rngs):

    self.actor = ActorNetwork(game.information_state_tensor_shape(), game.num_distinct_actions(),
                              config.actor_network_details[0], config.actor_network_details[1], rngs=rngs)
    #The critic is centralized
    self.critic = CriticNetwork(game.information_state_tensor_shape() * game.num_players(), config.bin_range,
                              config.critic_network_details[0], config.critic_network_details[1], rngs=rngs)
    
  @partial(nnx.jit, static_argnums=3)
  def get_policy(self, obs, legal, use_symlog=True) ->chex.Array:
    if use_symlog:
      obs = symlog(obs)
    return MARSSM.call_net(self.actor, obs, legal)
  
  
  
  @partial(nnx.jit, static_argnums=3)
  def get_policy_both(self, joint_obs, joint_legal, use_symlog=True) ->chex.Array:
    return self.get_policy_both_no_jit(joint_obs, joint_legal, use_symlog)
  
  def get_policy_both_no_jit(self, joint_obs, joint_legal, use_symlog=True) ->chex.Array:
    if use_symlog:
      joint_legal = symlog(joint_obs)
    vectorized_actor = nnx.vmap(MARSSM.call_net, in_axes=(None, 0, 0), out_axes=0)
    return vectorized_actor(self.actor, joint_obs, joint_legal)[0]
      



class SimRNaD():
  """An off-policy simultaneous move game
  version of Regularized Nash Dynamics,
  as described in https://arxiv.org/pdf/2510.05048."""
  def __init__(self, game: JaxGame, config: RNaDConfig, opt_config: OptimizerConfig, buffer_config: BufferConfig, seed:int, batch_size:int=32) -> None:
    self.config = config
    self.opt_config = opt_config
    self.buffer_config = buffer_config
    self.game = game
    self.init_seed = seed
    self.batch_size = batch_size
    self.init()

  def init(self):

    self.actions = self.game.num_distinct_actions()
    self.num_players = self.game.num_players()
    self.buffer = ActorReplayBuffer(self.game, self.buffer_config, self.init_seed, self.batch_size)

    #Subtract one from the trajectory lenght, we do not train on terminal nodes
    self.trajectory_max = self.game.max_trajectory_length() - 1
    self.non_chance_trajectory_max = self.game.max_trajectory_lenght_no_chance() - 1

    #self.cached_step = nnx.cached_partial(self.update_parameters_and_model, self.optimizer, self.target_optimizer, self.prev_network, self._prev_network)
    self.learner_steps = 0
    self.policy_switch_steps = 0

    self.trajectory_key = jax.random.key(self.init_seed)
    self.network_rngs = nnx.Rngs(self.init_seed)


    self.target_network = CriticNetwork(self.game.information_state_tensor_shape() * self.game.num_players(), self.config.bin_range,
                              self.config.critic_network_details[0], self.config.critic_network_details[1], rngs=self.network_rngs)
    self.ac_model = ACNetwork(self.game, self.config, self.network_rngs)
    self.buffer.cache_sampling(self.ac_model.actor)
    optim_tx = make_opt(self.opt_config)
    self.optimizer = nnx.Optimizer(self.ac_model, tx=optim_tx)

    self.target_optimizer = nnx.Optimizer(self.target_network, tx=optax.sgd(learning_rate=self.config.target_network_update))


    self._entropy_schedule = EntropySchedule(
        sizes=self.config.entropy_schedule_size,
        repeats=self.config.entropy_schedule_repeats)
    
    rnad_graphdef, rnad_state = nnx.split(self.ac_model.actor)
    self.prev_network = nnx.merge(rnad_graphdef, rnad_state)

    self._prev_network = nnx.merge(rnad_graphdef, rnad_state)

    self.metrics_keys = ['val', 'policy']
    self.metrics = {k: 0 for k in self.metrics_keys}
    self.network_keys = ['actor', 'critic']
    self.grad_norms = {}

    
  

  def get_next_key(self):
    self.trajectory_key, key = jax.random.split(self.trajectory_key)
    return key

  @partial(nnx.jit, static_argnums=(0,))
  def update_parameters_and_model(
    self,
    optimizer: nnx.Optimizer,
    target_optimizer: nnx.Optimizer,
    prev_network: ActorNetwork,
    _prev_network: ActorNetwork,
    rnad_timestep: ActorCriticTimeStep,
    learner_steps: int
  ):
    """Compute RNaD loss and use it to perform
    a gradient step of both RNaD and Dreamer."""
    alpha, update_regularization = self._entropy_schedule(learner_steps)

    def rnad_loss(
      rnad_network: ACNetwork,
      target_network: CriticNetwork,
      prev_network: ActorNetwork,
      _prev_network: ActorNetwork,
      timestep: ActorCriticTimeStep,
      alpha: float
    ):
      #If the timestep contains real environment infosets,
      # transform them with symlog first
      obs = symlog(timestep.obs)
      bins = jnp.arange((2 * self.config.bin_range) + 1) - self.config.bin_range
      # Per player vmap
      per_player_net_apply = nnx.vmap(MARSSM.call_net, in_axes=(None, 0, 0), out_axes=(0))
      #Per trajectory and batch dimensions
      vectorized_net_apply = nnx.vmap(nnx.vmap(per_player_net_apply, in_axes=(None, 0, 0), out_axes=(0)), in_axes=(None, 0, 0), out_axes=(0))
      #Critic is centralized
      vectorized_critic_apply = nnx.vmap(nnx.vmap(MARSSM.call_net, in_axes=(None, 0), out_axes=(0)), in_axes=(None, 0), out_axes=(0))
    
      pi, log_pi, logit = vectorized_net_apply(rnad_network.actor, obs, timestep.legal)

      joint_obs = jnp.reshape(obs, (*obs.shape[:-2], -1))

      v_dist_logits = vectorized_critic_apply(rnad_network.critic, joint_obs)
      #v = vectorized_critic_apply(rnad_network.critic, joint_obs)

      v_target_dist_logits = vectorized_critic_apply(target_network, joint_obs)
      #v_target = vectorized_critic_apply(target_network, joint_obs)
      _, log_pi_prev, _ = vectorized_net_apply(prev_network, obs, timestep.legal)
      _, log_pi_prev_, _ = vectorized_net_apply(_prev_network, obs, timestep.legal)
       
      v_target = get_value_from_bins(v_target_dist_logits, self.config.bin_range, use_symexp=False)
      # This creates the regularization term for rewards
      regularized_term = log_pi - (alpha * log_pi_prev + (1 - alpha) * log_pi_prev_) 
      
      expanded_valid = jnp.expand_dims(timestep.valid, (-1, -2))
      
      v_train_target, q_value = v_trace(v_target, expanded_valid, timestep.policy, pi, regularized_term, timestep.action, timestep.reward,
                                        self.config.lambda_vtrace, self.config.c_vtrace, self.config.rho_vtrace,
                                        self.config.eta, self.config.gamma_vtrace)
      
      # We do not take into account the player reaches, since infoset is always reached with the same prob
      sampling_policy = jnp.sum(timestep.policy * timestep.action, axis=-1, keepdims=True) * expanded_valid + (1 - expanded_valid)
      network_policy = jnp.sum(pi * timestep.action, axis=-1, keepdims=True)* expanded_valid + (1 - expanded_valid)
      sampling_policy = jnp.prod(sampling_policy, axis=-2, keepdims=True)
      
      importance_sampling = network_policy / sampling_policy
      
      importance_sampling = jnp.concatenate((jnp.ones_like(importance_sampling[0])[None, ...], importance_sampling[:-1]), axis=0)
      importance_sampling = jnp.cumprod(importance_sampling, axis=0)
      #Flip to turn into counterfactual importance sampling
      importance_sampling = jnp.flip(importance_sampling, axis=-2)
      #importance_sampling = 1.0
      #importance_sampling = 1.0)
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

      #The bin categorical loss
      v_loss = -get_bin_log_prob(v_dist_logits, bins, jax.lax.stop_gradient(v_train_target))
      # L2 loss
      #v_loss = -get_normal_log_prob(v, jax.lax.stop_gradient(v_train_target), use_symlog=False)
      v_loss_value = get_loss_mean_with_mask(v_loss, timestep.valid[..., None])

      loss_neurd = neurd_loss(logit, pi, q_value, timestep.legal, safe_cf_is,
                                self.config.neurd_clip, self.config.neurd_threshold)

      #Each player acts, so the normalization for the NeURD loss 
      # should be multiplied by 2 to get the true mean.
      # The multiplication by -1 is critical here, otherwise we would
      # be minimizing the neurd term, but we want to maximize it.
      neurd_loss_value = -get_loss_mean_with_mask(loss_neurd, expanded_valid, normalization_mult=2)
      return v_loss_value + neurd_loss_value, {'val': v_loss_value, 'policy': neurd_loss_value}
                                  

    (r_loss, r_metrics), rgrad = nnx.value_and_grad(rnad_loss, argnums=(0), has_aux=True)(
      optimizer.model,
      target_optimizer.model,
      prev_network,
      _prev_network,
      rnad_timestep,
      alpha,
    )

    grad_norms = self.grad_norms.copy()
    if self.config.report_gradnorms:
      for n in self.network_keys:
        grad_norms[n] = optax.tree.norm(rgrad[n], ord=2)
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
      
    state_prev, _state_prev = jax.lax.cond(
        update_regularization,
        lambda: (actor_state, state_prev),
        lambda: (state_prev, _state_prev))
    prev_network = nnx.merge(actor_graphdef, state_prev)
    _prev_network = nnx.merge(actor_graphdef, _state_prev)
    return prev_network, _prev_network, r_metrics, grad_norms, update_regularization

  
  def step(self):
    sample_key = self.get_next_key()
    timestep = self.buffer.mixed_sample(sample_key)

    self.prev_network, self._prev_network, self.metrics, self.grad_norms, update_regularization =  self.update_parameters_and_model(self.optimizer, self.target_optimizer, self.prev_network, 
                                                                                                                       self._prev_network, timestep,
                                                                                                                      self.learner_steps)
    self.learner_steps += 1
    
    self.policy_switch_steps += int(update_regularization)

  def train_model(self, model_save_dir:str, num_steps:int, print_each: int = -1, 
                  save_each: int = -1,
                  save_first: bool = False):
    
    
    print(f"Training model that is saved at {model_save_dir}")
    def save_latest():
      #Save which model file is the latest
      if latest_step > 0:
        latest_step_file = model_save_dir + LATEST_STEP_FILENAME
        with open(latest_step_file, 'w') as f:
          f.write(f"step_{latest_step}.pkl")
    latest_step = -1
    if save_first:
      model_file = model_save_dir + f"step_{self.learner_steps}.pkl"
      latest_step = self.learner_steps
      save_model(self, model_file)
      save_latest()
    
    #Start the training by sampling into the buffer,
    # to ensure that there are distinct data for at least one step
    init_batch_key = self.get_next_key()
    self.buffer.add_batch(self.batch_size, init_batch_key)
    for i in range(num_steps):
      self.step()
      if print_each > 0 and self.learner_steps % print_each == 0:
        print(f"Step {self.learner_steps}, Losses: {self.metrics}.")
        if self.config.report_gradnorms:
          print(f"Gradnorms:  {self.grad_norms}")
      if save_each > 0 and self.learner_steps % save_each == 0:
        latest_step = self.learner_steps
        model_file = model_save_dir + f"step_{self.learner_steps}.pkl"
        save_model(self, model_file)
        save_latest()

  
  def __getstate__(self):
      gen_state = {'config': self.config,
              'opt_config': self.opt_config,
              'buffer_config': self.buffer_config,
              'seed': self.init_seed,
              'game': self.game,
              'batch_size': self.batch_size,
              'learner_steps': self.learner_steps,
              'policy_steps': self.policy_switch_steps,
              'trajectory_key': self.trajectory_key,
              'optimizer': nnx.state(self.optimizer),
              'target_optimizer': nnx.state(self.target_optimizer),
              'prev_network': nnx.state(self.prev_network),
              '_prev_network': nnx.state(self._prev_network)
              }
      buffer_state = self.buffer.getstate()
      return {'gen': gen_state, 'buffer': buffer_state}
  
  def __setstate__(self, state):
    gen_state = state['gen']
    self.config = gen_state['config']
    self.opt_config = gen_state['opt_config']
    self.buffer_config = gen_state['buffer_config']
    self.init_seed = gen_state['seed']
    self.game = gen_state['game']
    self.batch_size = gen_state['batch_size']

    self.init()

    self.learner_steps = gen_state['learner_steps']
    self.trajectory_key = gen_state['trajectory_key']

    nnx.update(self.optimizer, gen_state['optimizer'])
    nnx.update(self.target_optimizer, gen_state['target_optimizer'])
    self.policy_switch_steps = gen_state['policy_steps']
    nnx.update(self.prev_network, gen_state['prev_network'])
    nnx.update(self._prev_network, gen_state['_prev_network'])

    self.buffer.setstate(state['buffer'])

