
import jax
import jax.numpy as jnp
import optax

import flax.nnx as nnx
import chex

import numpy as np

from functools import partial

from envs.jax_game import JaxGame, GameState
from nash_dreamer.ma_rssm import *
from nash_dreamer.train_utils import *
from nash_dreamer.distributions import get_bin_log_prob

from nash_dreamer.optimizer import make_opt
#The chance-node reward attribution, shared with ActorReplayBuffer so the two rollouts
# cannot drift apart, and the startup invariant check over it.
from nash_dreamer.replay_buffer import filter_chance_rewards, check_reward_alignment_once

#The PPO pieces shared with MMD. Deliberately NOT proximal_kl or magnet_kl, this is
# plain PPO with no mirror descent regularizers.
from nash_dreamer.sim_mmd import clipped_surrogate, normalized_advantage
#TD(lambda) without importance sampling. td_estimate(...) - v is exactly GAE(lambda).
from nash_dreamer.dreamer_actor_critic import td_estimate

u8 = jnp.uint8

#The non-finite guard in SimPPO.step. Divergence shows up in the first handful of
# updates, so those are all checked; after that it only costs a few scalar syncs
# occasionally. See SimPPO._check_finite for why this is worth paying for.
FINITE_CHECK_STEPS = 5
FINITE_CHECK_EVERY = 500


def policy_entropy(pi: chex.Array, log_pi: chex.Array):
  """Shannon entropy of the policy, summed over the action dimension.
  Illegal actions contribute nothing, since pi is zero there and log_pi is zero
  there by the convention of legal_log_policy. Multiply by the entropy coefficient
  and ADD to the objective (it is maximized)."""
  return -jnp.sum(pi * log_pi, axis=-1, keepdims=True)


class PPOActorCritic(nnx.Module):
  """Actor and critic for the single agent best-response learner.

  Deliberately not sim_rnad.ACNetwork: that one has a centralized critic over the
  joint infoset of both players, whereas this is a true single agent method whose
  critic sees only the learning player's own infoset.

  get_policy mirrors ACNetwork.get_policy exactly so that
  policy_eval_utils.head_to_head_play accepts this container with no changes."""

  def __init__(self, game: JaxGame, config: PPOConfig, rngs: nnx.Rngs):
    self.actor = ActorNetwork(game.information_state_tensor_shape(), game.num_distinct_actions(),
                              config.actor_network_details[0], config.actor_network_details[1], rngs=rngs)
    #Single agent: the critic sees one player's infoset, NOT the joint one
    self.critic = CriticNetwork(game.information_state_tensor_shape(), config.bin_range,
                                config.critic_network_details[0], config.critic_network_details[1], rngs=rngs)

  @partial(nnx.jit, static_argnums=3)
  def get_policy(self, obs, legal, use_symlog=True) ->chex.Array:
    if use_symlog:
      obs = symlog(obs)
    return MARSSM.call_net(self.actor, obs, legal)


class SimPPO():
  """Single agent PPO that learns an approximate best response to a FROZEN opponent.

  The opponent may be a DreamerMA, a SimRNaD or a SimMMD. Its actor is extracted at
  construction and never trained, so the environment is a stationary POMDP from the
  learner's point of view and plain PPO applies.

  The intended use is approximate exploitability: train this against a checkpoint,
  then play the two head to head. The mean return is a lower bound on how exploitable
  that checkpoint is. This is the scalable stand-in for
  policy_eval_utils.model_best_response, which is exact but only viable on tiny games."""

  def __init__(self, game: JaxGame, config: PPOConfig, opt_config: OptimizerConfig,
               opponent, seed: int, batch_size: int = 32) -> None:
    self.config = config
    self.opt_config = opt_config
    self.game = game
    self.init_seed = seed
    self.batch_size = batch_size
    self.opponent_actor = self._extract_opponent_actor(opponent, game)
    self.init()

  @staticmethod
  def _extract_opponent_actor(opponent, game: JaxGame) -> ActorNetwork:
    """Pull the actor out of a trained model. DreamerMA, SimRNaD and SimMMD all keep
    it at optimizer.model.actor, and in every supported case it is an ActorNetwork
    over the real game infoset."""
    if isinstance(opponent, ActorNetwork):
      #Already an extracted actor, used when restoring from a checkpoint
      return opponent
    net = getattr(getattr(opponent, "optimizer", None), "model", None)
    if net is None:
      raise ValueError(f"Cannot extract an actor from an opponent of type {type(opponent).__name__}. "
                       f"Expected a DreamerMA, SimRNaD or SimMMD.")
    if isinstance(net, MARSSM) and not net.use_real_infoset:
      raise ValueError(
          "SimPPO currently requires an opponent trained on the original game infosets. "
          "This DreamerMA was trained with latent infosets (use_real_infoset=False), whose "
          "actor consumes a learned latent infoset produced by a recurrence over the "
          "action-observation history. Supporting it would mean carrying that recurrence "
          "through the rollout, which is not implemented yet.")
    actor = getattr(net, "actor", None)
    if not isinstance(actor, ActorNetwork):
      raise ValueError(f"Opponent {type(opponent).__name__} has no ActorNetwork at optimizer.model.actor.")
    if hasattr(opponent, "game") and opponent.game.to_compact_str() != game.to_compact_str():
      raise ValueError(f"Opponent was trained on {opponent.game.to_compact_str()!r} but this "
                       f"best response is being trained on {game.to_compact_str()!r}. The infoset "
                       f"shapes would silently mismatch.")
    return actor

  def init(self):

    self.actions = self.game.num_distinct_actions()
    self.num_players = self.game.num_players()
    self.player_id = int(self.config.player_id)
    assert 0 <= self.player_id < self.num_players, \
        f"player_id {self.player_id} is out of range for a {self.num_players} player game."

    #These drive the rollout scan and the chance filter, so they must match
    # ActorReplayBuffer's values exactly (replay_buffer.py:97,101). Note SimRNaD and
    # SimMMD set decremented copies of these, but those are never read: their rollout
    # is done by the buffer, which keeps its own un-decremented lengths. Subtracting
    # one here truncates the trajectory and drops the terminal reward entirely.
    self.trajectory_max = self.game.max_trajectory_length()
    self.non_chance_trajectory_max = self.game.max_trajectory_lenght_no_chance()
    #There is no ReplayBuffer here to do it, so run the same startup invariant check
    # directly. A game that loses reward under the chance filter gives a zero
    # advantage, and the best response then silently never leaves its initial policy.
    check_reward_alignment_once(self.game)

    self.num_epochs = max(1, self.config.num_epochs)
    self.learner_steps = 0
    self.gradient_steps = 0

    self.trajectory_key = jax.random.key(self.init_seed)
    self.network_rngs = nnx.Rngs(self.init_seed)

    if self.config.sampling_epsilon > 0:
      print(f"Warning! PPO is on-policy, but sampling_epsilon={self.config.sampling_epsilon} > 0. "
            f"The stored behaviour policy is an epsilon-uniform mixture, so pi_old is not the policy "
            f"of the actor at collection time and the policy ratio is not 1 at the first inner epoch. "
            f"It also softens the best response, which under-estimates exploitability.")

    self.target_network = CriticNetwork(self.game.information_state_tensor_shape(), self.config.bin_range,
                                        self.config.critic_network_details[0],
                                        self.config.critic_network_details[1], rngs=self.network_rngs)
    self.ac_model = PPOActorCritic(self.game, self.config, self.network_rngs)
    optim_tx = make_opt(self.opt_config)
    self.optimizer = nnx.Optimizer(self.ac_model, tx=optim_tx)
    self.target_optimizer = nnx.Optimizer(self.target_network, tx=optax.sgd(learning_rate=self.config.target_network_update))

    self.example_timestep = self._make_example_timestep()

    self.metrics_keys = ['val', 'policy', 'entropy', 'clip_frac']
    self.metrics = {k: 0 for k in self.metrics_keys}
    self.network_keys = ['actor', 'critic']
    self.grad_norms = {}

    #The learner's mean episode return IS the best-response value estimate, so it is
    # tracked here even though there is no replay buffer to do it for us.
    self.smoothing_window = max(1, self.config.smoothing_window)
    self.return_log_frequency = self.config.return_log_frequency
    self.smoothing_returns = np.zeros(self.smoothing_window)
    self.smoothing_idx = 0
    self.smoothing_full = False
    self.minibatches = 0
    self.smoothed_returns = [0]

  def get_next_key(self):
    self.trajectory_key, key = jax.random.split(self.trajectory_key)
    return key

  # ----------------------------------------------------------------- collection

  @partial(nnx.jit, static_argnums=(0, 4))
  def sample_batch_trajectories(self, actor_network: ActorNetwork, opponent_actor: ActorNetwork,
                                key, batch_size: int):
    batch_keys = jax.random.split(key, batch_size)
    batch_sample = nnx.vmap(self.sample_trajectory, in_axes=(None, None, 0), out_axes=1)
    return batch_sample(actor_network, opponent_actor, batch_keys)

  @partial(nnx.jit, static_argnums=0)
  def sample_trajectory(self, actor_network: ActorNetwork, opponent_actor: ActorNetwork,
                        key) -> ActorCriticTimeStep:
    """Structured exactly like ActorReplayBuffer.sample_trajectory, except that each
    player is driven by its OWN actor network instead of a single vmapped one: the
    learner's actor at player_id and the frozen opponent's actor everywhere else.
    PPO is on-policy, so the batch is returned directly with no buffer behind it."""
    trajectory_key = jax.random.split(key, self.trajectory_max)
    actions = self.actions

    game_state, legal_actions = self.game.initialize_structures()

    @chex.dataclass(frozen=True)
    class SampleTrajectoryCarry:
      game_state: GameState
      legal_actions: chex.Array
      valid: bool

    init_carry = SampleTrajectoryCarry(
      game_state = game_state,
      legal_actions = legal_actions,
      valid = jnp.array(True),
    )

    @nnx.jit
    def choice_wrapper(key, p):
      action = jax.random.choice(key, actions, p=p)
      action_oh = jax.nn.one_hot(action, actions)
      return action, action_oh

    vectorized_sample_action = nnx.vmap(choice_wrapper, in_axes=(0, 0), out_axes=0)

    @nnx.scan(in_axes=(nnx.Carry, None, None, 0), out_axes=(nnx.Carry, 0))
    def _sample_trajectory(carry: SampleTrajectoryCarry, actor_network: ActorNetwork,
                           opponent_actor: ActorNetwork, key) -> tuple[SampleTrajectoryCarry, chex.Array]:

      state, p1_infoset, p2_infoset, public_state = self.game.get_info(carry.game_state)
      action_key, chance_key = jax.random.split(key, 2)

      obs = jnp.stack((p1_infoset, p2_infoset), axis=0)
      #Every training and collection path in this codebase feeds the actor symlog of
      # the real infoset, so that is what both actors get here.
      obs_for_actor = symlog(obs)

      #One network per player. player_id is a static Python int, so this unrolls at
      # trace time, mirroring how head_to_head_play stacks two separate actors.
      normalization = jnp.sum(carry.legal_actions, axis=-1, keepdims=True)
      uniform_pi = carry.legal_actions / (normalization + (normalization == 0))
      per_player_pi = []
      for p in range(self.num_players):
        net = actor_network if p == self.player_id else opponent_actor
        pi_p = jax.lax.stop_gradient(MARSSM.call_net(net, obs_for_actor[p], carry.legal_actions[p])[0])
        if p == self.player_id:
          #Only the learner explores. The opponent is queried at its true policy,
          # since the point is to best respond to THAT policy.
          pi_p = self.config.sampling_epsilon * uniform_pi[p] + (1 - self.config.sampling_epsilon) * pi_p
        per_player_pi.append(pi_p)
      pi = jnp.stack(per_player_pi, axis=0)

      is_chance = self.game.is_chance(carry.game_state)
      action_key = jax.random.split(action_key, self.num_players)
      action, action_oh = vectorized_sample_action(action_key, pi)

      def apply_action():
        return self.game.apply_action(carry.game_state, action)
      def sample_chance():
        outcomes, probs = self.game.get_outcomes_and_probs(carry.game_state)
        # Do not forget for deterministic games to put nonzero probs
        # to sample something for shape consistency
        probs = jnp.where(is_chance, probs, jnp.ones_like(probs) / probs.shape[0])
        chosen_outcome = jax.random.choice(chance_key, outcomes, p=probs)
        outcome, terminal, reward, chosen_legals = self.game.apply_action(carry.game_state, chosen_outcome)
        return outcome, terminal, reward, chosen_legals

      next_game_state, next_terminal, next_rewards, next_legal = jax.lax.cond(is_chance, sample_chance, apply_action)
      timestep = ActorCriticTimeStep(
        obs = obs,
        legal = carry.legal_actions.astype(u8),
        action = action_oh.astype(u8),
        reward = next_rewards,
        policy = pi,
        valid = carry.valid,
      )
      #We do not train actor/critic on terminal steps
      next_valid = jnp.logical_and(carry.valid, jnp.logical_not(next_terminal))

      new_carry = SampleTrajectoryCarry(
        game_state = next_game_state,
        legal_actions = next_legal,
        valid = next_valid,
      )

      timestep = tree_where(carry.valid, timestep, self.example_timestep)
      return new_carry, (timestep, is_chance)

    _, ys = _sample_trajectory(init_carry, actor_network, opponent_actor, trajectory_key)
    timestep, is_chance = ys
    #This is used to remove the chance nodes from the trajectory
    non_chance = jnp.nonzero(~is_chance, size=self.non_chance_trajectory_max)[0]
    filtered_timestep = jax.tree.map(
        lambda x: jnp.take_along_axis(x, jnp.expand_dims(non_chance, axis=range(1, x.ndim)), axis=0).astype(x.dtype),
        timestep)

    non_chance_timestep = ActorCriticTimeStep(
        obs = filtered_timestep.obs,
        legal = filtered_timestep.legal,
        action = filtered_timestep.action,
        policy = filtered_timestep.policy,
        reward = filter_chance_rewards(timestep.reward, is_chance, self.non_chance_trajectory_max),
        valid = filtered_timestep.valid)
    return non_chance_timestep

  def _make_example_timestep(self) -> ActorCriticTimeStep:
    """The all-invalid timestep that post-terminal steps are overwritten with,
    built exactly as ReplayBuffer._get_example_timestep does."""
    example_state, example_legals = self.game.initialize_structures()
    _, ex_p1, ex_p2, _ = self.game.get_info(example_state)
    ex_obs = jnp.stack([ex_p1, ex_p2], axis=0)
    legal = jnp.ones(example_legals.shape, dtype=u8)
    action = jax.nn.one_hot(jnp.argmax(legal, -1), legal.shape[-1]).astype(u8)
    policy = legal.astype(float) / jnp.sum(legal, axis=-1, keepdims=True)
    return ActorCriticTimeStep(obs=ex_obs, action=action, legal=legal,
                               policy=policy, reward=0.0, valid=False)

  # ----------------------------------------------------------------- learning

  @partial(nnx.jit, static_argnums=(0,))
  def update_parameters_and_model(
    self,
    optimizer: nnx.Optimizer,
    target_optimizer: nnx.Optimizer,
    ppo_timestep: ActorCriticTimeStep
  ):
    """One PPO iteration on a freshly collected on-policy batch. The value target and
    the advantages are computed once from the pre-update critic and held fixed while
    the actor and critic take num_epochs gradient steps on them."""

    pid = self.player_id
    bins = jnp.arange((2 * self.config.bin_range) + 1) - self.config.bin_range

    obs = symlog(ppo_timestep.obs)
    #Single agent: everything below is the learner's own slice
    own_obs = obs[..., pid, :]
    own_legal = ppo_timestep.legal[..., pid, :]
    own_action = ppo_timestep.action[..., pid, :]
    own_old_policy = ppo_timestep.policy[..., pid, :]
    #The stored reward is from player 0's perspective
    learner_reward = ppo_timestep.reward if pid == 0 else -ppo_timestep.reward
    expanded_valid = jnp.expand_dims(ppo_timestep.valid, -1)

    vectorized_apply = nnx.vmap(nnx.vmap(MARSSM.call_net, in_axes=(None, 0), out_axes=(0)), in_axes=(None, 0), out_axes=(0))
    vectorized_actor_apply = nnx.vmap(nnx.vmap(MARSSM.call_net, in_axes=(None, 0, 0), out_axes=(0)), in_axes=(None, 0, 0), out_axes=(0))

    def fixed_targets(target_network: CriticNetwork):
      v_old = get_value_from_bins(vectorized_apply(target_network, own_obs),
                                  self.config.bin_range, use_symexp=False)
      v_train_target = td_estimate(v_old, expanded_valid, learner_reward,
                                   self.config.td_lambda, self.config.gamma)
      #td_estimate returns v + sum (gamma * lambda)^k delta_{t+k}, so the difference
      # is exactly GAE(lambda). Single agent, so this stays a scalar advantage, with
      # none of the zero-sum per player stacking that MMD needs.
      advantage = normalized_advantage(v_train_target - v_old, expanded_valid, self.config.adv_norm_eps)
      return jax.lax.stop_gradient(v_train_target), jax.lax.stop_gradient(advantage)

    def ppo_loss(network: PPOActorCritic, v_train_target: chex.Array, advantage: chex.Array):
      pi, log_pi, logit = vectorized_actor_apply(network.actor, own_obs, own_legal)
      v_dist_logits = vectorized_apply(network.critic, own_obs)

      #Watch out! Do not call legal_log_policy here, as it assumes a logit and not a
      # softmaxed policy, so we get different results
      old_mask = (own_old_policy <= 1e-8)
      log_pi_old = jnp.log(own_old_policy + old_mask)
      log_pi_old = (1 - old_mask) * log_pi_old

      ratio = policy_ratio(pi, own_old_policy, own_action, expanded_valid)
      surrogate = clipped_surrogate(ratio, advantage, self.config.clip_epsilon)
      entropy = policy_entropy(pi, log_pi)

      #Plain PPO: clipped surrogate plus an entropy bonus. No magnet, no proximal KL.
      policy_objective = surrogate + self.config.entropy_coeff * entropy

      v_loss = -get_bin_log_prob(v_dist_logits, bins, jax.lax.stop_gradient(v_train_target))
      v_loss_value = get_loss_mean_with_mask(v_loss, expanded_valid)
      #Only one player is trained, so no doubling of the normalization here
      policy_loss_value = -get_loss_mean_with_mask(policy_objective, expanded_valid)

      metrics = {
        'val': v_loss_value,
        'policy': policy_loss_value,
        'entropy': get_loss_mean_with_mask(entropy, expanded_valid),
        'clip_frac': get_loss_mean_with_mask(
            (jnp.abs(ratio - 1.0) > self.config.clip_epsilon).astype(v_loss.dtype), expanded_valid),
      }
      return v_loss_value + policy_loss_value, metrics

    v_train_target, advantage = fixed_targets(target_optimizer.model)

    grad_norms = self.grad_norms.copy()
    summed_metrics = None
    for _ in range(self.num_epochs):
      (p_loss, p_metrics), pgrad = nnx.value_and_grad(ppo_loss, argnums=(0), has_aux=True)(
        optimizer.model, v_train_target, advantage)
      optimizer.update(pgrad)
      summed_metrics = p_metrics if summed_metrics is None else jax.tree.map(jnp.add, summed_metrics, p_metrics)
    metrics = jax.tree.map(lambda x: x / self.num_epochs, summed_metrics)

    if self.config.report_gradnorms:
      for n in self.network_keys:
        grad_norms[n] = optax.tree.norm(pgrad[n], ord=2)

    critic_state = nnx.state(optimizer.model.critic)
    state_target = nnx.state(target_optimizer.model)
    #This grad coupled with vanilla SGD optimizer is equivalent to the EMA formula
    # (1 - alpha) * state_target + alpha * state, which in turn corresponds to TD learning
    target_grad = jax.tree.map(lambda a, b: a - b, state_target, critic_state)
    target_optimizer.update(target_grad)

    return metrics, grad_norms

  def _track_returns(self, timestep: ActorCriticTimeStep):
    """The learner's mean episode return, which is the best-response value estimate."""
    reward = np.asarray(timestep.reward)
    if self.player_id != 0:
      reward = -reward
    #[Trajectory, Batch] -> one total return per trajectory
    returns = reward.sum(axis=0)
    for r in returns:
      self.smoothing_returns[self.smoothing_idx] = r
      self.smoothing_idx = (self.smoothing_idx + 1) % self.smoothing_window
      if self.smoothing_idx == 0:
        self.smoothing_full = True
    self.minibatches += 1
    if self.return_log_frequency > 0 and self.minibatches % self.return_log_frequency == 0:
      window = self.smoothing_returns if self.smoothing_full else self.smoothing_returns[:max(1, self.smoothing_idx)]
      self.smoothed_returns.append(float(window.mean()))

  @property
  def best_response_value(self) -> float:
    """Current smoothed estimate of the best-response value against the frozen opponent."""
    window = self.smoothing_returns if self.smoothing_full else self.smoothing_returns[:max(1, self.smoothing_idx)]
    return float(window.mean())

  def _check_finite(self):
    """Raise if the losses or the networks have gone non-finite.

    A NaN actor does not crash: legal_policy propagates the NaN and
    jax.random.choice(key, n, p=NaN) collapses to action 0 on every step, so training
    continues on a single degenerate line. The visible symptom is a plausible looking
    number -- a best-response value of exactly 0 with a standard error of exactly 0
    across a hundred thousand games, identical for every seed. That is a whole run
    wasted, so it is worth a handful of scalar syncs to turn it into an exception."""
    for name, value in self.metrics.items():
      if not bool(jnp.isfinite(jnp.asarray(value)).all()):
        raise FloatingPointError(
            f"SimPPO loss '{name}' is not finite at learner step {self.learner_steps}. "
            f"Training cannot recover from this: the policy collapses to a single action "
            f"and every trajectory becomes identical. Metrics: {self.metrics}")
    for name in self.network_keys:
      state = nnx.state(getattr(self.ac_model, name))
      if not all(bool(jnp.isfinite(leaf).all()) for leaf in jax.tree.leaves(state)):
        raise FloatingPointError(
            f"SimPPO {name} network has non-finite parameters at learner step "
            f"{self.learner_steps}. Metrics at failure: {self.metrics}")

  def step(self):
    sample_key = self.get_next_key()
    #Strictly on-policy: a fresh batch every step, no replay
    timestep = self.sample_batch_trajectories(self.ac_model.actor, self.opponent_actor,
                                             sample_key, self.batch_size)
    self._track_returns(timestep)
    self.metrics, self.grad_norms = self.update_parameters_and_model(
        self.optimizer, self.target_optimizer, timestep)
    self.learner_steps += 1
    self.gradient_steps += self.num_epochs
    #Divergence shows up in the first few updates, so check those and then only rarely.
    if self.learner_steps <= FINITE_CHECK_STEPS or self.learner_steps % FINITE_CHECK_EVERY == 0:
      self._check_finite()

  def store_returns(self, store_dir: str):
    """Same format as ReplayBuffer.store_returns, written inline since this learner
    has no replay buffer."""
    if not self.smoothed_returns:
      return
    os.makedirs(store_dir, exist_ok=True)
    total_minibatch_size = self.batch_size * self.game.max_trajectory_lenght_no_chance()
    env_steps = np.arange(len(self.smoothed_returns)) * total_minibatch_size * self.return_log_frequency
    with open(store_dir + "env_returns.txt", 'w') as f:
      f.write(f"{self.game.to_compact_str()}\n")
      f.write(f'Smoothing window: {self.smoothing_window}\n')
      for step, ret in zip(env_steps, self.smoothed_returns):
        f.write(f"Step: {step}, Return: {ret}\n")

  def train_model(self, model_save_dir: str, num_steps: int, print_each: int = -1,
                  save_each: int = -1, save_first: bool = False):

    print(f"Training model that is saved at {model_save_dir}")
    def save_latest():
      if latest_step > 0:
        with open(model_save_dir + LATEST_STEP_FILENAME, 'w') as f:
          f.write(f"step_{latest_step}.pkl")
    latest_step = -1
    if save_first:
      latest_step = self.learner_steps
      save_model(self, model_save_dir + f"step_{self.learner_steps}.pkl")
      save_latest()

    for i in range(num_steps):
      self.step()
      if print_each > 0 and self.learner_steps % print_each == 0:
        print(f"Step {self.learner_steps}, Losses: {self.metrics}. "
              f"BR value (player {self.player_id}): {self.best_response_value:.4f}")
        if self.config.report_gradnorms:
          print(f"Gradnorms:  {self.grad_norms}")
      if save_each > 0 and self.learner_steps % save_each == 0:
        latest_step = self.learner_steps
        save_model(self, model_save_dir + f"step_{self.learner_steps}.pkl")
        save_latest()
    self.store_returns(model_save_dir)

  def __getstate__(self):
    return {'gen': {
              'config': self.config,
              'opt_config': self.opt_config,
              'seed': self.init_seed,
              'game': self.game,
              'batch_size': self.batch_size,
              'learner_steps': self.learner_steps,
              'gradient_steps': self.gradient_steps,
              'trajectory_key': self.trajectory_key,
              'optimizer': nnx.state(self.optimizer),
              'target_optimizer': nnx.state(self.target_optimizer),
              #Only the frozen opponent's actor, not the whole opponent model. Keeps
              # the checkpoint self-contained and small.
              'opponent_actor': nnx.state(self.opponent_actor),
              'smoothed_returns': self.smoothed_returns,
            }}

  def __setstate__(self, state):
    gen = state['gen']
    self.config = gen['config']
    self.opt_config = gen['opt_config']
    self.init_seed = gen['seed']
    self.game = gen['game']
    self.batch_size = gen['batch_size']

    #Rebuild the opponent actor shell from the game and config, then restore its weights
    rngs = nnx.Rngs(self.init_seed)
    self.opponent_actor = ActorNetwork(self.game.information_state_tensor_shape(),
                                       self.game.num_distinct_actions(),
                                       self.config.actor_network_details[0],
                                       self.config.actor_network_details[1], rngs=rngs)
    self.init()

    self.learner_steps = gen['learner_steps']
    self.gradient_steps = gen['gradient_steps']
    self.trajectory_key = gen['trajectory_key']
    self.smoothed_returns = gen.get('smoothed_returns', [0])

    nnx.update(self.optimizer, gen['optimizer'])
    nnx.update(self.target_optimizer, gen['target_optimizer'])
    nnx.update(self.opponent_actor, gen['opponent_actor'])
