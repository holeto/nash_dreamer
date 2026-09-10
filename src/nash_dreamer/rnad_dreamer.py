
import jax
import jax.numpy as jnp
import jax.lax as lax
import optax

import flax.nnx as nnx
import chex


from functools import partial
from collections import defaultdict

from envs.jax_game import JaxGame
from nash_dreamer.ma_rssm import *
from nash_dreamer.train_utils import *
from nash_dreamer.distributions import get_bin_log_prob
from nash_dreamer.sim_rnad import v_trace, neurd_loss, EntropySchedule


class RNaDDreamer():
  """An off-policy simultaneous move game
  version of Regularized Nash Dynamics,
  as described in https://arxiv.org/pdf/2510.05048. 
  This version is additionally made to be compatible with Dreamer
  framework and training on imagined trajectories also."""
  def __init__(self, game: JaxGame, config: RNaDConfig, full_optimizer: nnx.Optimizer, target_optimizer: nnx.Optimizer) -> None:
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

    #self.cached_step = nnx.cached_partial(self.update_parameters_and_model, self.optimizer, self.target_optimizer, self.prev_network, self._prev_network)
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
    wm_prediction_step: PredictionStepWithLegal,
    learner_steps: int,
    imagine: bool
  ):
    """Compute RNaD loss and use it to perform
    a gradient step of both RNaD and Dreamer."""
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
      #Critic is centralized
      vectorized_critic_apply = nnx.vmap(nnx.vmap(MARSSM.call_net, in_axes=(None, 0), out_axes=(0)), in_axes=(None, 0), out_axes=(0))
    
      pi, log_pi, logit = vectorized_net_apply(rnad_network, obs, timestep.legal)

      joint_obs = jnp.reshape(obs, (*obs.shape[:-2], -1))

      v_dist_logits = vectorized_critic_apply(critic_network, joint_obs)

      v_target_dist_logits = vectorized_critic_apply(target_network, joint_obs)
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
      
      importance_sampling = jnp.concatenate((start_reaches_is, importance_sampling[:-1]), axis=0)
      importance_sampling = jnp.cumprod(importance_sampling, axis=0)
      
      #Flip to turn into counterfactual importance sampling
      importance_sampling = jnp.flip(importance_sampling, axis=-2)
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


      v_loss = -get_bin_log_prob(v_dist_logits, bins, jax.lax.stop_gradient(v_train_target))
      v_loss_value = get_loss_mean_with_mask(v_loss, timestep.valid[..., None])


      if compute_actor_loss:
        
        loss_neurd = neurd_loss(logit, pi, q_value, timestep.legal, safe_cf_is,
                                self.config.neurd_clip, self.config.neurd_threshold)

        # The multiplication by -1 is critical here, otherwise we would
        # be minimizing the neurd term, but we want to maximize it.
        neurd_loss_value = -get_loss_mean_with_mask(loss_neurd, expanded_valid, normalization_mult=2)
      else:
        neurd_loss_value = 0
      return v_loss_value + neurd_loss_value, v_loss_value, neurd_loss_value

    def imagination_loss(model: MARSSM,
      target_network: CriticNetwork,
      prev_network: ActorNetwork,
      _prev_network: ActorNetwork,
      trajectory_key,
      starting_points: PredictionStepWithLegal,
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
    
    def real_loss(model: MARSSM,
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
      
    # def take_starts_for_unroll(network_pi:chex.Array, timestep: TimeStep, prediction_step: PredictionStepWithLegal):
    #   """Choose a starting point that is not invalid or terminal in the timestep
    #   uniformly. Chooses over the trajectory dimension and should be 
    #   vmaped over the batch dimension"""
    #   expanded_valid = timestep.valid[..., None, None]
    #   #We also need to compute the importance sampling
    #   # for the player reaches, since we do not start at
    #   # the beggining of the trajectory.
    #   #Shape [T, Pl, 1]
    #   timestep_pi = jnp.sum(timestep.policy* timestep.action, axis=-1, keepdims=True) * expanded_valid + (1 - expanded_valid)
    #   #Shape [T, 1, 1]
    #   timestep_joint_pi = jnp.prod(timestep_pi, axis=-2, keepdims=True)
    #   #[T, Pl, 1]
    #   network_pi = jnp.sum(network_pi * timestep.action, axis=-1, keepdims=True) * expanded_valid + (1 - expanded_valid)
      
    #   #Make sure to shift it in time, since this gives
    #   # us the reaches for the next state
    #   trajectory_is = jnp.concatenate([jnp.ones((1, *network_pi.shape[1:])), network_pi[:-1] / timestep_joint_pi[:-1]], axis=0)
      
    #   #[T, Pl, 1]
    #   start_reaches_is = jnp.cumprod(trajectory_is, axis=0)
    #   #[N_last, ...]
    #   starts = jax.tree.map(lambda x: x[-self.num_starts:], prediction_step)
    #   #[N_last, Pl, 1]
    #   start_reaches_is = jax.tree.map(lambda x: x[-self.num_starts: ], start_reaches_is)
    #   #sampled_start = jax.tree_util.tree_map(lambda x: x[0], prediction_step)
    #   return starts, start_reaches_is
    # vectorized_starting_point = jax.vmap(take_starts_for_unroll, in_axes=(1, 1,1), out_axes=(0, 0))
    # per_player_net_apply = nnx.vmap(MARSSM.call_net, in_axes=(None, 0, 0), out_axes=(0))
    #   #Per trajectory and batch dimensions
    # vectorized_net_apply = nnx.vmap(nnx.vmap(per_player_net_apply, in_axes=(None, 0, 0), out_axes=(0)), in_axes=(None, 0, 0), out_axes=(0))
    
    
    rnad_timestep = wm_timestep_to_timestep(wm_timestep, wm_prediction_step, self.use_real_infoset)   
    #TODO: This will be called again in the real loss. Cannot get rid of the
    # redundant call somehow?
    #obs = symlog(rnad_timestep.obs) if self.use_real_infoset else rnad_timestep.obs
    #timestep_pi, _, _ = vectorized_net_apply(optimizer.model.actor, obs, rnad_timestep.legal)
    #starting_points, start_reaches_is = vectorized_starting_point(jax.lax.stop_gradient(timestep_pi), 
                                                                  #rnad_timestep, jax.tree.map(lambda x: x[:-1], wm_prediction_step))
    
    #Start imagination from the root and collapse the first two dimensions into num_starts * batch
    starting_points = jax.tree.map(lambda x: jnp.repeat(x[0][None, ...], self.num_starts, axis=0).reshape((-1, *x.shape[2:])), wm_prediction_step)
    #jax.debug.breakpoint()

    #jax.tree.map(lambda x: print(x.shape), starting_points)
    #Since we start at the root, the starting reaches are ones
    start_reaches_is = jnp.ones((1, self.num_starts * rnad_timestep.reward.shape[1], self.num_players, 1))
    #For the reaches just add a leading 1 dimension for shape consistency
    #start_reaches_is = jnp.reshape(start_reaches_is, (-1, *start_reaches_is.shape[2:]))[None, ...]

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
  

  
  def step(self, wm_timestep: TimeStep, wm_prediction_step:PredictionStepWithLegal, trajectory_key: chex.Array,
           should_imagine: bool):
    #should_imagine is decided by DreamerMA, which is the only thing that knows where stage
    # one ended and therefore where the warm-up after it ends. This learner's own
    # learner_steps cannot answer that: a hard stage one leaves the counter behind, and a
    # soft one lets it run ahead through stage one.
    self.prev_network, self._prev_network, self.metrics, self.grad_norms, update_regularization =  self.update_parameters_and_model(self.optimizer, self.target_optimizer, self.prev_network, 
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

