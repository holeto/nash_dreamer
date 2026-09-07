
import jax
import jax.numpy as jnp
import optax

import flax.nnx as nnx
import chex


from functools import partial

from envs.jax_game import JaxGame
from nash_dreamer.ma_rssm import *
from nash_dreamer.train_utils import *
from nash_dreamer.replay_buffer import ActorReplayBuffer
from nash_dreamer.distributions import get_bin_log_prob

from nash_dreamer.optimizer import make_opt

#The container wrapping the actor and the centralized critic is shared with RNaD,
# so that the evaluation utilities that dispatch on ACNetwork work for both learners.
from nash_dreamer.sim_rnad import ACNetwork
#TD(lambda) without importance sampling. Note that it returns
# v + sum (gamma * lambda)^k delta_{t+k}, so subtracting v from it
# gives exactly the GAE(lambda) advantage estimate.
from nash_dreamer.dreamer_actor_critic import td_estimate


def clipped_surrogate(
  ratio: chex.Array,
  advantage: chex.Array,
  clip_epsilon: float = 0.2
):
  """The PPO clipped surrogate objective. Both the ratio
  and the advantage are expected to be per player, of shape
  [..., Player, 1]. Multiply this by -1 to get a loss."""
  clipped_ratio = jnp.clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
  return jnp.minimum(ratio * advantage, clipped_ratio * advantage)


def proximal_kl(
  pi: chex.Array,
  log_pi: chex.Array,
  log_pi_old: chex.Array
):
  """The mirror descent proximal term KL(pi || pi_old), summed over
  the action dimension only. This is the reverse KL, the one that the
  mirror descent update actually solves for.
  Illegal actions contribute nothing, since pi is zero there and both
  log policies are zero there by the convention of legal_log_policy."""
  return jnp.sum(pi * (log_pi - log_pi_old), axis=-1, keepdims=True)


def magnet_kl(
  pi: chex.Array,
  log_pi: chex.Array,
  legal: chex.Array
):
  """The magnet term KL(pi || magnet) for a magnet fixed at the uniform policy.
  Since the uniform magnet has probability 1 / n_legal on every legal action,
  this equals -H(pi) + log(n_legal). It is written in that form rather than as an
  explicit KL to avoid taking a logarithm of the zero magnet probability at
  illegal actions. The log(n_legal) part is constant with respect to pi, so
  the gradient is identical to an entropy exploration bonus, but keeping the
  KL form makes a non uniform magnet a drop in replacement later."""
  #Cast first, the legal mask is stored as uint8 in the buffer
  num_legal = jnp.sum(legal.astype(pi.dtype), axis=-1, keepdims=True)
  negative_entropy = jnp.sum(pi * log_pi, axis=-1, keepdims=True)
  return negative_entropy + jnp.log(num_legal + (num_legal == 0))


def normalized_advantage(
  advantage: chex.Array,
  valid: chex.Array,
  eps: float = 1e-8
):
  """The standard PPO advantage normalization, by the mean and the standard
  deviation over the valid steps of the batch. Expects the scalar advantage from
  the perspective of player 1, of shape [Trajectory, Batch, 1], so that the
  centering happens before the zero-sum per player stacking and hence does
  not break the antisymmetry between the players."""
  mean = get_loss_mean_with_mask(advantage, valid)
  variance = get_loss_mean_with_mask(jnp.square(advantage - mean), valid)
  return (advantage - mean) / (jnp.sqrt(variance) + eps)


class SimMMD():
  """An on-policy simultaneous move game version of
  Magnetic Mirror Descent, as described in https://arxiv.org/pdf/2206.05825.
  The magnet is fixed at the uniform policy, which turns the magnet term into
  an entropy exploration bonus. The resulting update is effectively PPO,
  except that the mirror descent proximal term KL(pi || pi_old) is kept
  explicit alongside the usual ratio clipping.

  Unlike RNaD this learner is strictly on-policy, so it carries no
  importance sampling corrections of any kind."""
  def __init__(self, game: JaxGame, config: MMDConfig, opt_config: OptimizerConfig, buffer_config: BufferConfig, seed:int, batch_size:int=32) -> None:
    self.config = config
    self.opt_config = opt_config
    self.buffer_config = buffer_config
    self.game = game
    self.init_seed = seed
    self.batch_size = batch_size
    #Only warn for freshly constructed models. __setstate__ calls init() directly,
    # and the evaluation loads a lot of checkpoints in a row.
    self.warn_on_policy_violations()
    self.init()

  def warn_on_policy_violations(self):
    """MMD uses the behaviour policy stored in the buffer as pi_old. That is only
    the policy of the actor at collection time if the trajectories are freshly
    collected and not mixed with uniform exploration."""
    if self.buffer_config.sampling_epsilon > 0:
      print(f"Warning! MMD is strictly on-policy, but sampling_epsilon={self.buffer_config.sampling_epsilon} > 0. "
            f"The stored behaviour policy is an epsilon-uniform mixture, so pi_old is not the policy of the actor "
            f"at collection time and the policy ratio is not 1 at the first inner epoch. "
            f"Supply --sampling_epsilon 0 for strictly on-policy MMD.")
    if self.buffer_config.replay_ratio >= 1:
      print(f"Warning! MMD is strictly on-policy, but replay_ratio={self.buffer_config.replay_ratio} >= 1. "
            f"That mixes replayed off-policy trajectories into each minibatch, and MMD carries no importance "
            f"sampling correction for them. Supply --replay_ratio -1 for strictly on-policy MMD.")

  def init(self):

    self.actions = self.game.num_distinct_actions()
    self.num_players = self.game.num_players()
    self.buffer = ActorReplayBuffer(self.game, self.buffer_config, self.init_seed, self.batch_size)

    #Subtract one from the trajectory lenght, we do not train on terminal nodes
    self.trajectory_max = self.game.max_trajectory_length() - 1
    self.non_chance_trajectory_max = self.game.max_trajectory_lenght_no_chance() - 1

    #One learner step is one collected batch, i.e. one mirror descent iteration.
    # It takes num_epochs gradient steps, so that the conversion from learner steps
    # to environment steps stays directly comparable with RNaD.
    self.num_epochs = max(1, self.config.num_epochs)
    self.learner_steps = 0
    self.gradient_steps = 0

    self.trajectory_key = jax.random.key(self.init_seed)
    self.network_rngs = nnx.Rngs(self.init_seed)

    self.target_network = CriticNetwork(self.game.information_state_tensor_shape() * self.game.num_players(), self.config.bin_range,
                              self.config.critic_network_details[0], self.config.critic_network_details[1], rngs=self.network_rngs)
    self.ac_model = ACNetwork(self.game, self.config, self.network_rngs)
    self.buffer.cache_sampling(self.ac_model.actor)
    optim_tx = make_opt(self.opt_config)
    self.optimizer = nnx.Optimizer(self.ac_model, tx=optim_tx)

    self.target_optimizer = nnx.Optimizer(self.target_network, tx=optax.sgd(learning_rate=self.config.target_network_update))

    self.metrics_keys = ['val', 'policy', 'kl_old', 'magnet', 'clip_frac']
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
    mmd_timestep: ActorCriticTimeStep
  ):
    """Take one mirror descent iteration on a freshly collected on-policy batch.
    The value target and the advantages are computed once from the pre-update
    critic and then held fixed, while the actor and the critic take
    num_epochs gradient steps on them."""

    #If the timestep contains real environment infosets,
    # transform them with symlog first
    obs = symlog(mmd_timestep.obs)
    joint_obs = jnp.reshape(obs, (*obs.shape[:-2], -1))
    bins = jnp.arange((2 * self.config.bin_range) + 1) - self.config.bin_range

    # Per player vmap
    per_player_net_apply = nnx.vmap(MARSSM.call_net, in_axes=(None, 0, 0), out_axes=(0))
    #Per trajectory and batch dimensions
    vectorized_net_apply = nnx.vmap(nnx.vmap(per_player_net_apply, in_axes=(None, 0, 0), out_axes=(0)), in_axes=(None, 0, 0), out_axes=(0))
    #Critic is centralized
    vectorized_critic_apply = nnx.vmap(nnx.vmap(MARSSM.call_net, in_axes=(None, 0), out_axes=(0)), in_axes=(None, 0), out_axes=(0))

    #[Trajectory, Batch, 1] for the centralized critic and the TD estimate
    expanded_valid = jnp.expand_dims(mmd_timestep.valid, -1)
    #[Trajectory, Batch, 1, 1] for the per player policy terms
    player_valid = jnp.expand_dims(mmd_timestep.valid, (-1, -2))

    def fixed_targets(target_network: CriticNetwork):
      """The value target and the advantages, computed once per mirror descent
      iteration from the pre-update critic and kept fixed across the inner epochs."""
      v_target_dist_logits = vectorized_critic_apply(target_network, joint_obs)
      v_old = get_value_from_bins(v_target_dist_logits, self.config.bin_range, use_symexp=False)

      v_train_target = td_estimate(v_old, expanded_valid, mmd_timestep.reward,
                                   self.config.td_lambda, self.config.gamma)
      #td_estimate returns v + sum (gamma * lambda)^k delta_{t+k},
      # so this difference is exactly the GAE(lambda) advantage.
      p1_advantage = normalized_advantage(v_train_target - v_old, expanded_valid, self.config.adv_norm_eps)
      #The advantage is from the perspective of player 1, the game is zero-sum
      #[Trajectory, Batch, Player, 1]
      advantage = jnp.stack([p1_advantage, -p1_advantage], axis=-2)
      return jax.lax.stop_gradient(v_train_target), jax.lax.stop_gradient(advantage)

    def mmd_loss(
      mmd_network: ACNetwork,
      timestep: ActorCriticTimeStep,
      v_train_target: chex.Array,
      advantage: chex.Array
    ):
      pi, log_pi, logit = vectorized_net_apply(mmd_network.actor, obs, timestep.legal)
      v_dist_logits = vectorized_critic_apply(mmd_network.critic, joint_obs)

      #Watch out! Do not call legal_log_policy here, as
      # it assumes a logit and not a softmaxed policy, so we get
      # different results
      old_policy_mask = (timestep.policy <= 1e-8)
      log_pi_old = jnp.log(timestep.policy + old_policy_mask)
      log_pi_old = (1 - old_policy_mask) * log_pi_old

      #The stored behaviour policy is pi_old. With no uniform exploration mixture
      # and freshly collected trajectories it is exactly the policy of the actor at
      # collection time, so this ratio is 1 at the first inner epoch.
      ratio = policy_ratio(pi, timestep.policy, timestep.action, player_valid)

      surrogate = clipped_surrogate(ratio, advantage, self.config.clip_epsilon)
      kl_old = proximal_kl(pi, log_pi, log_pi_old)
      kl_magnet = magnet_kl(pi, log_pi, timestep.legal)

      policy_objective = surrogate - self.config.kl_coeff * kl_old - self.config.magnet_coeff * kl_magnet

      #The bin categorical loss
      v_loss = -get_bin_log_prob(v_dist_logits, bins, jax.lax.stop_gradient(v_train_target))
      v_loss_value = get_loss_mean_with_mask(v_loss, expanded_valid)

      #Each player acts, so the normalization for the policy loss
      # should be multiplied by 2 to get the true mean.
      # The multiplication by -1 is critical here, otherwise we would
      # be minimizing the policy objective, but we want to maximize it.
      policy_loss_value = -get_loss_mean_with_mask(policy_objective, player_valid, normalization_mult=2)

      #Diagnostics. The clipped fraction is the standard indicator of how far
      # the inner epochs have moved the policy away from the collected batch.
      clip_frac = get_loss_mean_with_mask((jnp.abs(ratio - 1.0) > self.config.clip_epsilon).astype(v_loss.dtype),
                                          player_valid, normalization_mult=2)
      metrics = {'val': v_loss_value,
                 'policy': policy_loss_value,
                 'kl_old': get_loss_mean_with_mask(kl_old, player_valid, normalization_mult=2),
                 'magnet': get_loss_mean_with_mask(kl_magnet, player_valid, normalization_mult=2),
                 'clip_frac': clip_frac}
      return v_loss_value + policy_loss_value, metrics

    v_train_target, advantage = fixed_targets(target_optimizer.model)

    grad_norms = self.grad_norms.copy()
    summed_metrics = None
    for _ in range(self.num_epochs):
      (m_loss, m_metrics), mgrad = nnx.value_and_grad(mmd_loss, argnums=(0), has_aux=True)(
        optimizer.model,
        mmd_timestep,
        v_train_target,
        advantage,
      )
      optimizer.update(mgrad)
      summed_metrics = m_metrics if summed_metrics is None else jax.tree.map(jnp.add, summed_metrics, m_metrics)
    #Report the metrics averaged over the inner epochs, but the gradient norms
    # of the last one, since those are what the optimizer state ends up reflecting.
    m_metrics = jax.tree.map(lambda x: x / self.num_epochs, summed_metrics)
    if self.config.report_gradnorms:
      for n in self.network_keys:
        grad_norms[n] = optax.tree.norm(mgrad[n], ord=2)

    critic_state = nnx.state(optimizer.model.critic)
    state_target = nnx.state(target_optimizer.model)

    #This grad coupled with vanilla SGD optimizer
    # is equivalent to the EMA formula (1 - alpha) * state_target + alpha * state
    # Which, in turn corresponds to TD learning
    target_grad = jax.tree.map(lambda a, b: a - b, state_target, critic_state)
    target_optimizer.update(target_grad)

    return m_metrics, grad_norms


  def step(self):
    sample_key = self.get_next_key()
    #With replay_ratio < 1 this collects a fresh fully online batch, while
    # still going through store_batch so that the return logging keeps working.
    timestep = self.buffer.mixed_sample(sample_key)

    self.metrics, self.grad_norms = self.update_parameters_and_model(self.optimizer, self.target_optimizer, timestep)
    self.learner_steps += 1
    self.gradient_steps += self.num_epochs

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

    #Unlike RNaD there is no initial buffer seeding here. MMD never replays,
    # so each step collects the fresh batch it trains on.
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
    self.buffer.store_returns(model_save_dir)


  def __getstate__(self):
      gen_state = {'config': self.config,
              'opt_config': self.opt_config,
              'buffer_config': self.buffer_config,
              'seed': self.init_seed,
              'game': self.game,
              'batch_size': self.batch_size,
              'learner_steps': self.learner_steps,
              'gradient_steps': self.gradient_steps,
              'trajectory_key': self.trajectory_key,
              'optimizer': nnx.state(self.optimizer),
              'target_optimizer': nnx.state(self.target_optimizer),
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
    self.gradient_steps = gen_state['gradient_steps']
    self.trajectory_key = gen_state['trajectory_key']

    nnx.update(self.optimizer, gen_state['optimizer'])
    nnx.update(self.target_optimizer, gen_state['target_optimizer'])

    self.buffer.setstate(state['buffer'])
