import jax
import os
import jax.numpy as jnp
import chex
import numpy as np

import flax.nnx as nnx
from dataclasses import dataclass
from functools import partial

from nash_dreamer.distributions import sample_categorical
from nash_dreamer.ma_rssm import MARSSM
from nash_dreamer.networks import *
from envs.jax_game import JaxGame, GameState
from nash_dreamer.train_utils import TimeStep, ActorCriticTimeStep,BufferConfig, DreamerMAConfig, tree_where, symlog

u8 = jnp.uint8
nu8 = np.uint8


def filter_chance_rewards(reward: chex.Array, is_chance: chex.Array, non_chance_max: int) -> chex.Array:
  """Attribute the per-step rewards of a raw trajectory to its non-chance steps.

  The actor-critic timestep stores the reward for *acting* at a step, so each
  non-chance step must carry its own reward plus the rewards of every chance step
  that follows it, up to the next non-chance step. Expressed as a segment sum this
  is independent of how chance and play nodes interleave.

  It replaces a `jnp.nonzero(~jnp.roll(is_chance, 1), ...)` mask. `jnp.roll(x, 1)`
  shifts right, so that mask actually tested whether the *previous* step was a
  chance node, which credited every step with the reward of the step after it. In
  games where chance and play strictly alternate (`JaxRandomGoofspiel`) every play
  was therefore credited with a chance node's reward -- always zero -- and the whole
  trajectory return was silently destroyed.

  Note WMReplayBuffer does NOT use this: its TimeStep stores the reward for
  *arriving* at a step, which stays aligned with obs under the plain non-chance
  mask, and train_utils.wm_timestep_to_timestep does the conversion shift.
  """
  #Index of the most recent non-chance step. Leading chance steps get -1 and are
  # dropped, they precede every decision in the filtered trajectory.
  segment = jnp.cumsum((~is_chance).astype(jnp.int32), axis=0) - 1
  contribution = jnp.where(segment >= 0, reward, jnp.zeros_like(reward))
  return jax.ops.segment_sum(contribution, jnp.maximum(segment, 0),
                             num_segments=non_chance_max).astype(reward.dtype)


#Games already checked in this process, keyed by to_compact_str(). The check is a
# property of the game, not of the learner, so once per game per process is enough.
_CHECKED_GAMES = set()


def check_reward_alignment(game: JaxGame, num_trajectories: int = 4, seed: int = 0):
  """Verify that filtering the chance nodes out of a trajectory preserves its return.

  Runs a handful of uniform-policy rollouts and compares the sum of the raw per-step
  rewards against the sum after `filter_chance_rewards`. A mismatch means reward is
  being silently discarded, which shows up as a zero advantage and a learner that
  never moves off its initial policy -- not as an error. It has happened: every
  `JaxRandomGoofspiel` run lost 100% of its reward this way.

  The two ways a game can still fail this after the filter itself is correct are the
  two length declarations, both of which this catches:
    * `max_trajectory_length()` too small -- the game never reaches terminal inside
      the scan, so the terminal reward is never collected.
    * `max_trajectory_lenght_no_chance()` too small -- trailing segments fall off the
      end of the segment sum.
  """
  key = jax.random.key(seed)
  actions = game.num_distinct_actions()
  trajectory_max = game.max_trajectory_length()
  non_chance_max = game.max_trajectory_lenght_no_chance()
  for _ in range(num_trajectories):
    key, traj_key = jax.random.split(key)
    game_state, legal_actions = game.initialize_structures()
    rewards, chance_flags = [], []
    valid, reached_terminal = True, False
    for _ in range(trajectory_max):
      traj_key, action_key, chance_key = jax.random.split(traj_key, 3)
      is_chance = bool(game.is_chance(game_state))
      if is_chance:
        outcomes, probs = game.get_outcomes_and_probs(game_state)
        chosen = jax.random.choice(chance_key, outcomes, p=probs)
        game_state, terminal, reward, legal_actions = game.apply_action(game_state, chosen)
      else:
        normalization = jnp.sum(legal_actions, axis=-1, keepdims=True)
        pi = legal_actions / (normalization + (normalization == 0))
        action_keys = jax.random.split(action_key, game.num_players())
        action = jnp.stack([jax.random.choice(action_keys[p], actions, p=pi[p])
                            for p in range(game.num_players())])
        game_state, terminal, reward, legal_actions = game.apply_action(game_state, action)
      #Post-terminal steps are absorbing and must not contribute, exactly as the
      # rollout masks them out with the example timestep.
      rewards.append(float(reward) if valid else 0.0)
      chance_flags.append(is_chance)
      if valid and bool(terminal):
        reached_terminal = True
      valid = valid and not bool(terminal)

    raw = jnp.asarray(rewards)
    filtered = filter_chance_rewards(raw, jnp.asarray(chance_flags), non_chance_max)
    if not reached_terminal:
      raise AssertionError(
          f"Game {game.to_compact_str()} did not reach a terminal state within "
          f"max_trajectory_length()={trajectory_max}. The terminal reward is never "
          f"collected, so every trajectory return is wrong.")
    if not bool(jnp.allclose(jnp.sum(filtered), jnp.sum(raw))):
      raise AssertionError(
          f"Game {game.to_compact_str()} loses reward when its chance nodes are filtered "
          f"out: raw return {float(jnp.sum(raw)):+.6f} but {float(jnp.sum(filtered)):+.6f} "
          f"after filtering. Check max_trajectory_length()={trajectory_max} and "
          f"max_trajectory_lenght_no_chance()={non_chance_max} against the real game.")


def check_reward_alignment_once(game: JaxGame):
  """check_reward_alignment, skipped if this game was already checked in this process."""
  name = game.to_compact_str()
  if name in _CHECKED_GAMES:
    return
  _CHECKED_GAMES.add(name)
  check_reward_alignment(game)


@dataclass
class BufferTimeStep():
  """Same structure as TimeStep, only
  is intended to be used on CPU, hence the arrays are numpy"""
  obs: np.ndarray
  action: np.ndarray
  legal: np.ndarray
  policy: np.ndarray
  reward: np.ndarray
  terminal: np.ndarray
  valid: np.ndarray

  def __getitem__(self, indices):
    return BufferTimeStep(
      obs = self.obs[indices],
      action = self.action[indices],
      legal = self.legal[indices],
      policy = self.policy[indices],
      reward = self.reward[indices],
      terminal = self.terminal[indices],
      valid = self.valid[indices]
    )

  def __setitem__(self, indices, value):
    self.obs[indices] = value.obs
    self.action[indices] = value.action
    self.legal[indices] = value.legal
    self.policy[indices] = value.policy
    self.reward[indices] = value.reward
    self.terminal[indices] = value.terminal
    self.valid[indices] = value.valid

  def __len__(self):
    return self.obs.shape[0]

@dataclass
class BufferActorStep():
  """Same structure as ActorCriticTimeStep, only
  is intended to be used on CPU, hence the arrays are numpy"""
  obs: np.ndarray
  action: np.ndarray
  policy: np.ndarray
  legal: np.ndarray
  reward: np.ndarray
  valid: np.ndarray
  def __getitem__(self, indices):
    return BufferActorStep(
      obs = self.obs[indices],
      action = self.action[indices],
      legal = self.legal[indices],
      policy = self.policy[indices],
      reward = self.reward[indices],
      valid = self.valid[indices]
    )

  def __setitem__(self, indices, value):
    self.obs[indices] = value.obs
    self.action[indices] = value.action
    self.legal[indices] = value.legal
    self.policy[indices] = value.policy
    self.reward[indices] = value.reward
    self.valid[indices] = value.valid

  def __len__(self):
    return self.obs.shape[0]


class ReplayBuffer():
  def __init__(self, game: JaxGame, config: BufferConfig, seed: int, batch_size: int, actor_critic: bool = False):
    self.game = game
    self.np_rng = np.random.default_rng(seed=0)
    self.config = config
    self.action_dimension = self.game.num_distinct_actions()
    self.num_players = self.game.num_players()
    self.trajectory_max = self.game.max_trajectory_length()
    #Chance nodes are not explicitly stored in the buffer, instead
    # they are skipped and only the next outcome sampled from it 
    # is stored.
    self.non_chance_trajectory_max = self.game.max_trajectory_lenght_no_chance()
    #A game whose declared lengths do not match its real trajectories silently loses
    # reward, which looks like a learner that simply refuses to improve rather than
    # like an error. Cheap enough to always pay, once per game per process.
    check_reward_alignment_once(self.game)
    self.batch_size = batch_size
    self.total_minibatch_size = batch_size * self.non_chance_trajectory_max
    #If we supply replay ratio < 1, it is assumed
    # that we want all steps online
    if self.config.replay_ratio < 1:
      self.online_trajectories = batch_size
      self.online_trajectories_remainder = 0.0
      self.replayed_batches = 0
    else:
      online_steps_per_batch = self.total_minibatch_size / self.config.replay_ratio
      online_trajectories_float = online_steps_per_batch / self.non_chance_trajectory_max
      self.online_trajectories = int(online_trajectories_float)
      self.online_trajectories_remainder = online_trajectories_float - self.online_trajectories
      self.replayed_batches = batch_size - self.online_trajectories
      # By design, one online trajectory should carry non_chance_trajectory_max replayed
      # trajectories alongside it (batch_size and replay_ratio are meant to be chosen so
      # this divides evenly) -- so this branch runs only for replay_ratio >= 1, i.e.
      # already non-negative. A mismatch here previously produced a real, silent
      # difference in how many real environment steps a run consumed per gradient step
      # relative to a non-replayed run (e.g. NashDreamer) at the same batch_size, without
      # any error or NaN to surface it.
      if self.online_trajectories > 0:
        actual_ratio = self.replayed_batches / self.online_trajectories
        if actual_ratio != self.non_chance_trajectory_max:
          print(f"Warning! The actual ratio of replayed trajectories per real trajectory is {actual_ratio} "
                f"with batch_size {batch_size} and replay ratio {self.config.replay_ratio}. If you are "
                f"comparing with NashDreamer, the comparison is not 1:1")
    self.online_trajectories_accumulator = 0.0
    
    self.smoothed_returns = [0]
    self.smoothing_rewards = np.zeros(self.config.smoothing_window)
    self.smoothing_idx = 0
    self.smoothing_full = False
    self.minibatches = 0

    self.actor_critic = actor_critic

    self.buffer_index = 0
    self.full = False

    self._get_example_timestep()
    self.init_buffer()

  def _get_example_timestep(self):
    #This can produce a chance node, but that 
    # one by default produces invalid infosets
    # and legals so it is not a problem 
    example_state, example_legals = self.game.initialize_structures()
    _, ex_p1_infoset, ex_p2_infoset, _ = self.game.get_info(example_state)
    ex_obs = jnp.stack([ex_p1_infoset, ex_p2_infoset], axis=0)
    legal = jnp.ones(example_legals.shape, dtype=u8)
    action = jax.nn.one_hot(jnp.argmax(legal, -1), legal.shape[-1]).astype(u8)
    policy = legal.astype(float) / jnp.sum(legal, axis=-1, keepdims=True)
    if self.actor_critic:
      self.example_timestep = ActorCriticTimeStep(obs= ex_obs,
                                      action=action,
                                      legal=legal,
                                      policy = policy,
                                      reward = 0.0,
                                      valid = False)
    else:  
      self.example_timestep = TimeStep(
                                      obs= ex_obs,
                                      action=action,
                                      legal=legal,
                                      policy = policy,
                                      reward = 0.0,
                                      terminal = False,
                                      valid = False)
    
  def init_buffer(self):
    #Use np for the buffer itself, since it is supposed to run
    # on cpu. The actual sampling can be done on gpu
    obs = np.zeros((self.config.buffer_size, self.non_chance_trajectory_max, *self.example_timestep.obs.shape))
    action = np.zeros((self.config.buffer_size, self.non_chance_trajectory_max, *self.example_timestep.action.shape), dtype=nu8)
    legal = np.zeros((self.config.buffer_size, self.non_chance_trajectory_max, *self.example_timestep.legal.shape), dtype=nu8)
    policy = np.zeros((self.config.buffer_size, self.non_chance_trajectory_max,  *self.example_timestep.policy.shape))
    reward = np.zeros((self.config.buffer_size, self.non_chance_trajectory_max))
    terminal = np.zeros((self.config.buffer_size, self.non_chance_trajectory_max), dtype=bool)
    valid = np.zeros((self.config.buffer_size, self.non_chance_trajectory_max), dtype=bool)
    if self.actor_critic:
      self.buffer = BufferActorStep(obs=obs,
                                    action=action,
                                    legal=legal,
                                    policy=policy,
                                    reward=reward,
                                    valid=valid)
    else:
      self.buffer = BufferTimeStep(
        obs = obs,
        action = action,
        legal = legal,
        policy = policy,
        reward = reward,
        terminal = terminal,
        valid = valid,
      )
  
  def env_to_buffer_timestep(self, env_timestep: TimeStep| ActorCriticTimeStep) ->BufferTimeStep|BufferActorStep:
    """Convert a timestep sampled from the environment 
    to the buffer timestep. Also, transposes from [Time, Batch, ...]
    format in which the environment steps are collected
    into [Batch, Time, ...] format in which they are stored in the buffer"""
    env_timestep = jax.tree.map(lambda x: x.transpose(1, 0, *(range(2, x.ndim))), env_timestep)
    if self.actor_critic:
      buffer_timestep = BufferActorStep(
                                    obs= np.asarray(env_timestep.obs),
                                    action= np.asarray(env_timestep.action),
                                    legal= np.asarray(env_timestep.legal),
                                    policy= np.asarray(env_timestep.policy),
                                    reward= np.asarray(env_timestep.reward),
                                    valid= np.asarray(env_timestep.valid))
    else:
      buffer_timestep = BufferTimeStep(
                                      obs= np.asarray(env_timestep.obs),
                                      action= np.asarray(env_timestep.action),
                                      legal= np.asarray(env_timestep.legal),
                                      policy= np.asarray(env_timestep.policy),
                                      reward= np.asarray(env_timestep.reward),
                                      terminal= np.asarray(env_timestep.terminal),
                                      valid= np.asarray(env_timestep.valid))
    return buffer_timestep

  def buffer_to_env_timestep(self, buffer_timestep: BufferTimeStep|BufferActorStep) ->TimeStep|ActorCriticTimeStep:
    """Convert a timestep sampled from buffer into environment timestep.
    Also, transposes from [Batch, Time, ...] to [Time, Batch, ...], since 
    the training expects time major format."""
    if self.actor_critic:
      env_timestep = ActorCriticTimeStep(
                                    obs= jnp.asarray(buffer_timestep.obs),
                                    action= jnp.asarray(buffer_timestep.action),
                                    legal= jnp.asarray(buffer_timestep.legal),
                                    policy= jnp.asarray(buffer_timestep.policy),
                                    reward= jnp.asarray(buffer_timestep.reward),
                                    valid= jnp.asarray(buffer_timestep.valid))
    else:
      env_timestep = TimeStep(
                                      obs= jnp.asarray(buffer_timestep.obs),
                                      action= jnp.asarray(buffer_timestep.action),
                                      legal= jnp.asarray(buffer_timestep.legal),
                                      policy= jnp.asarray(buffer_timestep.policy),
                                      reward= jnp.asarray(buffer_timestep.reward),
                                      terminal= jnp.asarray(buffer_timestep.terminal),
                                      valid= jnp.asarray(buffer_timestep.valid))
    env_timestep = jax.tree.map(lambda x: x.transpose((1, 0, *range(2, x.ndim))).astype(x.dtype), env_timestep)
    return env_timestep

  def add_single(self, buffer_timestep: BufferTimeStep| BufferActorStep):
    """Add a single trajectory into the buffer"""
    self.buffer[self.buffer_index] = buffer_timestep

    self.buffer_index = self.buffer_index + 1
    if self.buffer_index == self.config.buffer_size:
      self.buffer_index = 0
      self.full = True

  def store_batch(self, batch_trajectories: TimeStep| ActorCriticTimeStep):
    buffer_timestep = self.env_to_buffer_timestep(batch_trajectories)
    batch_size = len(buffer_timestep)
    #Assuming one reward per trajectory
    rewards = np.sum(buffer_timestep.reward, axis=1)
    if batch_size + self.smoothing_idx < self.config.smoothing_window:
      self.smoothing_rewards[self.smoothing_idx: self.smoothing_idx + batch_size] = rewards
      self.smoothing_idx += batch_size
    elif self.config.smoothing_window <= batch_size:
      self.smoothing_rewards = rewards[-self.config.smoothing_window:]
      self.smoothing_full = True
    else:
      space_left = self.config.smoothing_window - self.smoothing_idx
      self.smoothing_rewards[self.smoothing_idx:] = rewards[:space_left]
      remaining_items = batch_size - space_left
      self.smoothing_rewards[:remaining_items] = rewards[space_left:]
      self.smoothing_idx = remaining_items
      self.smoothing_full = True
    self.minibatches += 1
    if self.config.return_log_frequency > 0 and self.minibatches % self.config.return_log_frequency == 0:
        if self.smoothing_full:
          self.smoothed_returns.append(self.smoothing_rewards.mean())
        else:
          self.smoothed_returns.append(self.smoothing_rewards[:self.smoothing_idx].mean())
    for i in range(batch_size):
      self.add_single(buffer_timestep[i])
  
  def mixed_sample(self, env_sample_key: chex.Array):
    """Handles the case where we both want to sample trajectories online
    and from the buffer. Samples online_batch_size online batches,
    buffer_batch_size batches from replay buffer and concatenates them
    together. The size of these batches is computed based on replay
    ratio, such that there are (batch size * trajectory len) / replay ratio
    online steps per minibatch"""
    self.online_trajectories_accumulator += self.online_trajectories_remainder
    extra = 0
    if self.online_trajectories_accumulator >= 1.0:
      extra = 1
      self.online_trajectories_accumulator -= 1.0
    online_traj = self.online_trajectories + extra
    replayed_traj = self.batch_size - online_traj

    if online_traj == 0:
      return self.sample_batch(replayed_traj)
    elif replayed_traj == 0:
      return self.add_batch(online_traj, env_sample_key)
    #Sample first, before putting the new online trajectories there
    buffer_batch = self.sample_batch(replayed_traj)
    online_batch = self.add_batch(online_traj, env_sample_key)
    #The second axis is batch size
    compound_batch = jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=1), online_batch, buffer_batch)
    return compound_batch
  
  def add_batch(self, batch_size: int, env_sample_key):
    raise NotImplementedError("Calling the abstract add_batch method of the ReplayBuffer parent class.")
  
  
  def store_returns(self, store_dir: str):
    if not self.config.log_returns:
      return
    if not self.smoothed_returns:
        print("No returns to log.")
        return
    os.makedirs(store_dir, exist_ok=True)

    
    # Generate X-axis (Total Trajectories)
    # We know we log every 'return_log_frequency' trajectories
    env_steps = np.arange(len(self.smoothed_returns)) * self.total_minibatch_size * self.config.return_log_frequency
    
    return_file = store_dir + "env_returns.txt"
    
    with open(return_file, 'w') as f:
      #The first line contains the game name string
      f.write(f"{self.game.to_compact_str()}\n")
      #The second line defines the smoothing
      # window size (so that it can be written for plotting)
      f.write(f'Smoothing window: {self.config.smoothing_window}\n')
      for step, ret in zip(env_steps, self.smoothed_returns):
        f.write(f"Step: {step}, Return: {ret}\n")
    
    

  def sample_batch(self, batch_size: int) ->TimeStep| ActorCriticTimeStep:
    #empty buffer
    high = self.config.buffer_size if self.full else self.buffer_index
    sampled_indices = self.np_rng.integers(0, high, size=batch_size)
    sampled_timesteps = self.buffer[sampled_indices]
    timesteps_for_train = self.buffer_to_env_timestep(sampled_timesteps)
    return timesteps_for_train


  def getstate(self):
    return {
      "numpy_rng_state": self.np_rng.bit_generator.state,
      #"full" : self.full,
      #"smoothed_returns" : self.smoothed_returns,
      #"smoothing_idx": self.smoothing_idx,
      #"smoothing_full": self.smoothing_full,
      #"buffer_index": self.buffer_index,
      #"buffer": self.buffer,
    }
    
  
  def setstate(self, state):
    self.np_rng.bit_generator.state = state["numpy_rng_state"]
    #self.full = state["full"]
    #self.buffer_index = state["buffer_index"]
    #self.buffer = state["buffer"]
    #self.smoothed_returns = state["smoothed_returns"]
    #self.smoothing_full = state["smoothing_full"]
    #self.smoothing_idx = state["smoothing_idx"]





class ActorReplayBuffer(ReplayBuffer):
  def __init__(self, game: JaxGame, config: BufferConfig, seed:int, batch_size: int):
    """Circular replay buffer for collecting trajectories from the game for standalone actor-critic algorithm.
    Individual elements are trajectories."""
    super().__init__(game, config, seed, batch_size, actor_critic=True)
    self.cached_sample = None

  
  def cache_sampling(self, actor_network: ActorNetwork):
    self.cached_sample = nnx.cached_partial(self.sample_batch_trajectories, actor_network)

  
  
  def add_batch(self, batch_size: int, env_sample_key, actor_network: ActorNetwork| None = None):
    if self.cached_sample is None:
      assert actor_network is not None, f"Called add_batch without supplying Actor network and without caching sampling!"
      batch_trajectories = self.sample_batch_trajectories(actor_network, env_sample_key, batch_size)
    else:
      batch_trajectories = self.cached_sample(env_sample_key, batch_size)
    self.store_batch(batch_trajectories)
    return batch_trajectories

  
  @partial(nnx.jit, static_argnums=(0, 3))
  def sample_batch_trajectories(self, actor_network: ActorNetwork, key, batch_size: int):
    batch_keys = jax.random.split(key, batch_size)
    batch_sample_trajectories = nnx.vmap(self.sample_trajectory, in_axes=(None, 0), out_axes=(1))
    batch_trajectories = batch_sample_trajectories(actor_network, batch_keys)
    return batch_trajectories

  @partial(nnx.jit, static_argnums=0)
  def sample_trajectory(self, actor_network:ActorNetwork, key) ->TimeStep:
    trajectory_key = jax.random.split(key, self.trajectory_max)
    actions = self.action_dimension

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

    def get_actor_policy(actor_network: ActorNetwork, obs, legal_actions):
      return actor_network(obs, legal_actions)[0]
    #per player vmap
    vectorized_get_actor = nnx.vmap(get_actor_policy, in_axes=(None, 0, 0), out_axes=0)

    @nnx.scan(in_axes=(nnx.Carry, None, 0), out_axes=(nnx.Carry, 0))
    def _sample_trajectory(carry: SampleTrajectoryCarry, actor_network:ActorNetwork, key) -> tuple[SampleTrajectoryCarry, chex.Array]:
      
      
      state, p1_infoset, p2_infoset, public_state = self.game.get_info(carry.game_state)
      action_key, chance_key = jax.random.split(key, 2)
      
      obs = jnp.stack((p1_infoset, p2_infoset), axis=0)
      obs_for_actor = symlog(obs)
        
      pi = jax.lax.stop_gradient(vectorized_get_actor(actor_network, obs_for_actor, carry.legal_actions))
      #uniform mix to the policy
      normalization = jnp.sum(carry.legal_actions, axis=-1, keepdims=True)
      uniform_pi = carry.legal_actions / (normalization + (normalization == 0))
      pi = self.config.sampling_epsilon * uniform_pi + (1 - self.config.sampling_epsilon) * pi
      is_chance = self.game.is_chance(carry.game_state)
      action_key = jax.random.split(action_key, self.game.num_players())
      action, action_oh = vectorized_sample_action(action_key, pi)
      def apply_action():
        return self.game.apply_action(carry.game_state, action)
      def sample_chance():
        outcomes, probs = self.game.get_outcomes_and_probs(carry.game_state)
        # Do not forget for deterministic games to put nonzero probs
        # to sample something for shape consistency
        probs = jnp.where(is_chance, probs, jnp.ones_like(probs)/ probs.shape[0])
        chosen_outcome = jax.random.choice(chance_key, outcomes, p=probs)
        outcome, terminal, reward, chosen_legals = self.game.apply_action(carry.game_state, chosen_outcome)
        return outcome, terminal, reward, chosen_legals
      next_game_state, next_terminal, next_rewards, next_legal = jax.lax.cond(is_chance, sample_chance, apply_action)
      timestep = ActorCriticTimeStep(
        obs = obs,
        legal = carry.legal_actions.astype(u8),
        action = action_oh.astype(u8),
        reward=next_rewards,
        policy = pi,
        valid = carry.valid,
      )
      #We do not train actor/critic on terminal steps
      next_valid = jnp.logical_and(carry.valid, jnp.logical_not(next_terminal))

      new_carry = SampleTrajectoryCarry(
        game_state = next_game_state,
        legal_actions=next_legal,
        valid = next_valid

      )
        
      
      timestep = tree_where(carry.valid, timestep, self.example_timestep)
      
      return new_carry, (timestep, is_chance)
    _, ys = _sample_trajectory(init_carry, actor_network, trajectory_key)
    timestep, is_chance = ys
    #This is used to remove the chance nodes from the trajectory
    non_chance = jnp.nonzero(~is_chance, size=self.non_chance_trajectory_max)[0]
    filtered_timestep = jax.tree.map(lambda x: jnp.take_along_axis(x, jnp.expand_dims(non_chance, axis=range(1, x.ndim)), axis=0).astype(x.dtype), timestep)

    non_chance_timestep = ActorCriticTimeStep(obs= filtered_timestep.obs,
                                    legal=filtered_timestep.legal,
                                    action=filtered_timestep.action,
                                    policy = filtered_timestep.policy,
                                    reward = filter_chance_rewards(timestep.reward, is_chance, self.non_chance_trajectory_max),
                                    valid = filtered_timestep.valid)

    #[Trajectory, ...]
    return non_chance_timestep


class WMReplayBuffer(ReplayBuffer):

  def __init__(self, game: JaxGame, config: BufferConfig, world_model_config: DreamerMAConfig, seed:int,
              stoch_state_sample_threshold: float = 0):
    """Circular replay buffer for collecting trajectories from the game for NashDreamer.
    Individual elements are trajectories."""
    self.wm_config = world_model_config
    self.stoch_state_sample_threshold = stoch_state_sample_threshold
    super().__init__(game, config, seed, world_model_config.batch_size, actor_critic=False)
    self.init_constants()

  def init_constants(self):
    
    recurrent_state_size = self.wm_config.sequential_network_details[0]
    latent_infoset_dim = self.wm_config.infoset_network_details[0]

    if recurrent_state_size < 1:
      recurrent_state_size = self.game.information_state_tensor_shape() * self.num_players

    if latent_infoset_dim < 1:
      latent_infoset_dim = self.game.information_state_tensor_shape()
    self.recurrent_state_size = recurrent_state_size
    self.latent_infoset_dim = latent_infoset_dim

    self.use_real_infoset = self.wm_config.use_original_infoset


    self._get_example_timestep()


    self.cached_sample = None
    
  
  
    

  def cache_sampling(self, recurrent_network: SequenceModel, encoder_network: Encoder,
                     observer_network: ObservedPredictor, infoset_network: InfosetModel, actor_network: ActorNetwork):
    self.cached_sample = nnx.cached_partial(self.sample_batch_trajectories, recurrent_network,
                                            encoder_network, observer_network, infoset_network, actor_network)
    
  
  
  def add_batch(self, batch_size: int, sample_key: chex.Array, recurrent_network: SequenceModel| None = None,
                observer_network: ObservedPredictor | None = None,
                encoder_network: Encoder | None = None, infoset_network: InfosetModel | None = None,
                actor_network: ActorNetwork| None =None):
    """Sample a batch of trajectories from the environment and add them to the buffer.
    Also returns the trajectories if you want to perform online training on them."""
    if all([net is not None for net in (recurrent_network, encoder_network, observer_network,infoset_network, actor_network)]):
      batch_trajectories = self.sample_batch_trajectories(recurrent_network, encoder_network, observer_network, infoset_network, actor_network, batch_size, sample_key)
    else:
      assert self.cached_sample is not None, "The variant of add_batch where one or more of the networks are unset was called, but cached_sample is not set. Please call cache_sampling first."
      batch_trajectories = self.cached_sample(batch_size, sample_key)
    self.store_batch(batch_trajectories)
    return batch_trajectories
  


  @partial(nnx.jit, static_argnums=(0, 6))
  def sample_batch_trajectories(self, recurrent_network: SequenceModel, encoder_network: Encoder, observer_network: ObservedPredictor, infoset_network: InfosetModel, actor_network: ActorNetwork, batch_size:int, key):
    batch_keys = jax.random.split(key, batch_size)
    batch_sample_trajectories = nnx.vmap(self.sample_trajectory, in_axes=(None, None, None, None, None, 0), out_axes=(1))
    batch_trajectories = batch_sample_trajectories(recurrent_network, encoder_network, observer_network,infoset_network, actor_network, batch_keys)
    return batch_trajectories

  @partial(nnx.jit, static_argnums=0)
  def sample_trajectory(self, recurrent_network: SequenceModel, encoder_network:Encoder, observer_network: ObservedPredictor, infoset_network: InfosetModel, actor_network:ActorNetwork, key) ->TimeStep:
    trajectory_key = jax.random.split(key, self.trajectory_max)
    actions = self.action_dimension

    vectorized_next_infoset = nnx.vmap(MARSSM.call_net, in_axes=(None, 0, 0, 0), out_axes=(0))

    
    #TODO: Will later have to properly split the segments 
    # and also store the starting model state
    game_state, legal_actions = self.game.initialize_structures()
    dummy_deter = jnp.zeros((self.wm_config.encoded_classes, self.wm_config.encoded_categories))
    dummy_recur = jnp.zeros((self.recurrent_state_size))
    dummy_action = jnp.zeros((self.game.num_players(), self.game.num_distinct_actions()))

    init_recur = recurrent_network(dummy_recur, dummy_deter, dummy_action)
    init_infoset = jnp.zeros((self.num_players, self.latent_infoset_dim))
    
    @chex.dataclass(frozen=True)
    class SampleTrajectoryCarry:
      game_state: GameState
      legal_actions: chex.Array
      reward: chex.Array
      terminal: bool
      valid: bool
      recurrent_state: chex.Array
      joint_latent_infoset: chex.Array
      prev_action: chex.Array

    init_carry = SampleTrajectoryCarry(
      game_state = game_state,
      legal_actions = legal_actions,
      reward = jnp.array(0),
      terminal = jnp.array(False),
      valid = jnp.array(True),
      recurrent_state = init_recur,
      joint_latent_infoset= init_infoset,
      prev_action = dummy_action
    )
    
    
    @nnx.jit
    def choice_wrapper(key, p):
      action = jax.random.choice(key, actions, p=p)
      action_oh = jax.nn.one_hot(action, actions)
      return action, action_oh

    
    vectorized_sample_action = nnx.vmap(choice_wrapper, in_axes=(0, 0), out_axes=0)

    def get_actor_policy(actor_network: ActorNetwork, obs, legal_actions):
      return actor_network(obs, legal_actions)[0]
    #per player vmap
    vectorized_get_actor = nnx.vmap(get_actor_policy, in_axes=(None, 0, 0), out_axes=0)

    @nnx.scan(in_axes=(nnx.Carry, None, None, None, None,None, 0), out_axes=(nnx.Carry, 0))
    def _sample_trajectory(carry: SampleTrajectoryCarry, recurrent_network: SequenceModel, encoder_network:Encoder, observer_network: ObservedPredictor,
                            infoset_network: InfosetModel, actor_network:ActorNetwork, key) -> tuple[SampleTrajectoryCarry, chex.Array]:
      
      
      state, p1_infoset, p2_infoset, public_state = self.game.get_info(carry.game_state)
      action_key, chance_key, deter_sample_key = jax.random.split(key, 3)
      
      obs = jnp.stack((p1_infoset, p2_infoset), axis=0)
      enc_obs = symlog(obs) if not self.wm_config.obs_loss_bce else obs
      tokens = encoder_network(enc_obs)
      encoded_stoch = observer_network(carry.recurrent_state, tokens)
      encoded_deter = sample_categorical(encoded_stoch, deter_sample_key, self.stoch_state_sample_threshold)
      joint_latent_infoset = carry.joint_latent_infoset
      joint_latent_infoset = vectorized_next_infoset(infoset_network, carry.joint_latent_infoset, enc_obs, carry.prev_action)
      obs_for_actor = symlog(obs) if self.use_real_infoset else joint_latent_infoset
        
      pi = jax.lax.stop_gradient(vectorized_get_actor(actor_network, obs_for_actor, carry.legal_actions))
      #uniform mix to the policy
      normalization = jnp.sum(carry.legal_actions, axis=-1, keepdims=True)
      uniform_pi = carry.legal_actions / (normalization + (normalization == 0))
      pi = self.config.sampling_epsilon * uniform_pi + (1 - self.config.sampling_epsilon) * pi
      is_chance = self.game.is_chance(carry.game_state)
      action_key = jax.random.split(action_key, self.game.num_players())
      action, action_oh = vectorized_sample_action(action_key, pi)
      next_recur = recurrent_network(carry.recurrent_state, encoded_deter, action_oh)
      def apply_action():
        return self.game.apply_action(carry.game_state, action)
      def sample_chance():
        outcomes, probs = self.game.get_outcomes_and_probs(carry.game_state)
        # Do not forget for deterministic games to put nonzero probs
        # to sample something for shape consistency
        probs = jnp.where(is_chance, probs, jnp.ones_like(probs)/ probs.shape[0])
        chosen_outcome = jax.random.choice(chance_key, outcomes, p=probs)
        outcome, terminal, reward, chosen_legals = self.game.apply_action(carry.game_state, chosen_outcome)
        return outcome, terminal, reward, chosen_legals

      next_game_state, next_terminal, next_rewards, next_legal = jax.lax.cond(is_chance,
                                    sample_chance, apply_action)

      timestep = TimeStep(
        obs = obs,
        legal = carry.legal_actions.astype(u8),
        action = action_oh.astype(u8),
        policy = pi,
        reward = carry.reward,
        valid = carry.valid,
        terminal = carry.terminal
      )
      #Action in terminal state is not valid
      next_terminal = jnp.logical_or(carry.terminal, next_terminal)
      next_valid = jnp.logical_not(carry.terminal)

      new_carry = SampleTrajectoryCarry(
        game_state = next_game_state,
        legal_actions=jnp.where(next_terminal, self.example_timestep.legal, next_legal),
        reward = next_rewards,
        terminal = next_terminal,
        valid = next_valid,
        recurrent_state = next_recur,
        joint_latent_infoset = joint_latent_infoset,
        prev_action = action_oh

      )
        
      
      timestep = tree_where(carry.valid, timestep, self.example_timestep)
      
      return new_carry, (timestep, is_chance)
    _, ys = _sample_trajectory(init_carry, recurrent_network, encoder_network, observer_network, infoset_network,  actor_network, trajectory_key)
    timestep, is_chance = ys
    #This is used to remove the chance nodes from the trajectory
    non_chance = jnp.nonzero(~is_chance, size=self.non_chance_trajectory_max)[0]
    filtered_timestep = jax.tree.map(lambda x: jnp.take_along_axis(x, jnp.expand_dims(non_chance, axis=range(1, x.ndim)), axis=0).astype(x.dtype), timestep)
    #[Trajectory, ...]
    return filtered_timestep
