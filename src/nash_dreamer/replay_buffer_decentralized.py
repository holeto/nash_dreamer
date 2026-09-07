"""Replay buffer for the decentralized world model.

Only the trajectory-sampling side differs from `WMReplayBuffer`: each player
carries its OWN recurrent state and advances it on its OWN action, and there is
no latent-infoset network to run.  Storage, `add_batch`, `mixed_sample`,
`store_batch` and the checkpoint hooks are all inherited unchanged -- the stored
`TimeStep` is already per-player (`obs [T, B, Player, obs_dim]`).

Because the decentralized variant always trains on real infosets, the actor here
reads `symlog(obs)` directly and no latent state is needed during sampling at
all.
"""

import jax
import jax.numpy as jnp
import chex
import numpy as np

import flax.nnx as nnx
from functools import partial

from nash_dreamer.distributions import sample_categorical
from nash_dreamer.networks import ObservedPredictor, ActorNetwork
from nash_dreamer.networks_decentralized import DecSequenceModel, DecEncoder
from envs.jax_game import GameState
from nash_dreamer.train_utils import TimeStep, tree_where, symlog
from nash_dreamer.replay_buffer import WMReplayBuffer

u8 = jnp.uint8


class DecentralizedWMReplayBuffer(WMReplayBuffer):

  def init_constants(self):
    """Same as the centralized version, except the default recurrent state size
    is a SINGLE player's infoset and there is no latent infoset dimension."""
    recurrent_state_size = self.wm_config.sequential_network_details[0]
    if recurrent_state_size < 1:
      recurrent_state_size = self.game.information_state_tensor_shape()
    self.recurrent_state_size = recurrent_state_size
    self.latent_infoset_dim = recurrent_state_size + (self.wm_config.encoded_classes
                                                      * self.wm_config.encoded_categories)

    self.use_real_infoset = self.wm_config.use_original_infoset
    assert self.use_real_infoset, (
      "The decentralized replay buffer is only defined for real infosets. "
      "Pass --use_original_infoset.")

    self._get_example_timestep()

    self.cached_sample = None

  def cache_sampling(self, recurrent_network: DecSequenceModel, encoder_network: DecEncoder,
                     observer_network: ObservedPredictor, actor_network: ActorNetwork):
    """Four networks instead of five -- there is no infoset network to cache."""
    self.cached_sample = nnx.cached_partial(self.sample_batch_trajectories, recurrent_network,
                                            encoder_network, observer_network, actor_network)

  def add_batch(self, batch_size: int, sample_key: chex.Array,
                recurrent_network: DecSequenceModel | None = None,
                observer_network: ObservedPredictor | None = None,
                encoder_network: DecEncoder | None = None,
                actor_network: ActorNetwork | None = None):
    """Sample a batch of trajectories from the environment and add them to the buffer.
    Also returns the trajectories if you want to perform online training on them."""
    if all([net is not None for net in (recurrent_network, encoder_network, observer_network, actor_network)]):
      batch_trajectories = self.sample_batch_trajectories(recurrent_network, encoder_network,
                                                          observer_network, actor_network,
                                                          batch_size, sample_key)
    else:
      assert self.cached_sample is not None, "The variant of add_batch where one or more of the networks are unset was called, but cached_sample is not set. Please call cache_sampling first."
      batch_trajectories = self.cached_sample(batch_size, sample_key)
    self.store_batch(batch_trajectories)
    return batch_trajectories

  @partial(nnx.jit, static_argnums=(0, 5))
  def sample_batch_trajectories(self, recurrent_network: DecSequenceModel, encoder_network: DecEncoder,
                                observer_network: ObservedPredictor, actor_network: ActorNetwork,
                                batch_size: int, key):
    batch_keys = jax.random.split(key, batch_size)
    batch_sample_trajectories = nnx.vmap(self.sample_trajectory, in_axes=(None, None, None, None, 0), out_axes=(1))
    return batch_sample_trajectories(recurrent_network, encoder_network, observer_network,
                                     actor_network, batch_keys)

  @partial(nnx.jit, static_argnums=0)
  def sample_trajectory(self, recurrent_network: DecSequenceModel, encoder_network: DecEncoder,
                        observer_network: ObservedPredictor, actor_network: ActorNetwork, key) -> TimeStep:
    trajectory_key = jax.random.split(key, self.trajectory_max)
    actions = self.action_dimension
    num_players = self.game.num_players()

    #Every world-model network is applied per player, over a leading player axis.
    def call_net(net, *args):
      return net(*args)
    vectorized_seq = nnx.vmap(call_net, in_axes=(None, 0, 0, 0), out_axes=0)
    vectorized_enc = nnx.vmap(call_net, in_axes=(None, 0, 0), out_axes=0)
    vectorized_observer = nnx.vmap(call_net, in_axes=(None, 0), out_axes=0)

    def sample_deter_per_player(stoch_logits, deter_key):
      """Each player samples independently: sample_categorical derives its
      threshold with a min() over the whole array, so a stacked call would
      couple the players' thresholds."""
      keys = jax.random.split(deter_key, num_players)
      return jax.vmap(sample_categorical, in_axes=(0, 0, None))(
        stoch_logits, keys, self.stoch_state_sample_threshold)

    game_state, legal_actions = self.game.initialize_structures()
    dummy_deter = jnp.zeros((num_players, self.wm_config.encoded_classes, self.wm_config.encoded_categories))
    dummy_recur = jnp.zeros((num_players, self.recurrent_state_size))
    dummy_action = jnp.zeros((num_players, self.game.num_distinct_actions()))

    init_recur = vectorized_seq(recurrent_network, dummy_recur, dummy_deter, dummy_action)

    @chex.dataclass(frozen=True)
    class SampleTrajectoryCarry:
      game_state: GameState
      legal_actions: chex.Array
      reward: chex.Array
      terminal: bool
      valid: bool
      recurrent_state: chex.Array  # [Player, recurrent_state_size]
      prev_action: chex.Array
      prev_is_chance: chex.Array #Was the previous row a chance node?
      pending_negative_obs: chex.Array #Flattened negative observations from the
                                            # previous row's chance resolution

    init_carry = SampleTrajectoryCarry(
      game_state = game_state,
      legal_actions = legal_actions,
      reward = jnp.array(0),
      terminal = jnp.array(False),
      valid = jnp.array(True),
      recurrent_state = init_recur,
      prev_action = dummy_action,
      prev_is_chance = jnp.array(False),
      pending_negative_obs = jnp.zeros((self.num_negatives, int(np.prod(self.example_timestep.obs.shape))))
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

    def encode_negative_outcome(game_state, outcome):
      neg_state, _, _, _ = self.game.apply_action(game_state, outcome)
      _, neg_p1_infoset, neg_p2_infoset, _ = self.game.get_info(neg_state)
      neg_obs = jnp.stack((neg_p1_infoset, neg_p2_infoset), axis=0)
      return neg_obs.reshape(-1)
    #vmap over the negative outcomes; game_state is broadcast as a closure
    vectorized_encode_negative = jax.vmap(encode_negative_outcome, in_axes=(None, 0), out_axes=0)

    @nnx.scan(in_axes=(nnx.Carry, None, None, None, None, 0), out_axes=(nnx.Carry, 0))
    def _sample_trajectory(carry: SampleTrajectoryCarry, recurrent_network: DecSequenceModel,
                           encoder_network: DecEncoder, observer_network: ObservedPredictor,
                           actor_network: ActorNetwork, key) -> tuple[SampleTrajectoryCarry, chex.Array]:

      state, p1_infoset, p2_infoset, public_state = self.game.get_info(carry.game_state)
      action_key, chance_key, deter_sample_key = jax.random.split(key, 3)

      obs = jnp.stack((p1_infoset, p2_infoset), axis=0)
      enc_obs = symlog(obs) if not self.wm_config.obs_loss_bce else obs
      #Each player encodes only its own observation into its own posterior.
      tokens = vectorized_enc(encoder_network, carry.recurrent_state, enc_obs)
      encoded_stoch = vectorized_observer(observer_network, tokens)
      encoded_deter = sample_deter_per_player(encoded_stoch, deter_sample_key)
      #No latent infoset is computed: with real infosets the actor reads the
      # game's own tensors, so nothing downstream would consume one.
      obs_for_actor = symlog(obs)

      pi = jax.lax.stop_gradient(vectorized_get_actor(actor_network, obs_for_actor, carry.legal_actions))
      #uniform mix to the policy
      normalization = jnp.sum(carry.legal_actions, axis=-1, keepdims=True)
      uniform_pi = carry.legal_actions / (normalization + (normalization == 0))
      pi = self.config.sampling_epsilon * uniform_pi + (1 - self.config.sampling_epsilon) * pi
      is_chance = self.game.is_chance(carry.game_state)
      action_key = jax.random.split(action_key, num_players)
      action, action_oh = vectorized_sample_action(action_key, pi)
      #Does not depend on the cond's outcome, so it can be computed here already
      # and used to encode negatives meant for the *next* row.
      #Player i advances on action_oh[i] only.
      next_recur = vectorized_seq(recurrent_network, carry.recurrent_state, encoded_deter, action_oh)
      def apply_action():
        next_state, terminal, reward, legal = self.game.apply_action(carry.game_state, action)
        next_pending_negative_obs = jnp.zeros((self.num_negatives, obs.size), dtype=obs.dtype)
        return next_state, terminal, reward, legal, next_pending_negative_obs
      def sample_chance():
        outcomes, probs = self.game.get_outcomes_and_probs(carry.game_state)
        # Do not forget for deterministic games to put nonzero probs
        # to sample something for shape consistency
        probs = jnp.where(is_chance, probs, jnp.ones_like(probs)/ probs.shape[0])
        num_outcomes = outcomes.shape[0]

        choice_key, negative_key = jax.random.split(chance_key, 2)
        chosen_idx = jax.random.choice(choice_key, num_outcomes, p=probs)
        chosen_outcome = outcomes[chosen_idx]

        #Sample self.num_negatives outcomes with replacement, excluding the chosen
        # outcome. Fall back to the unmasked distribution if it was the only
        # outcome with nonzero probability.
        neg_probs = probs * (1 - jax.nn.one_hot(chosen_idx, num_outcomes))
        neg_probs_sum = jnp.sum(neg_probs)
        neg_probs = jnp.where(neg_probs_sum > 0, neg_probs / (neg_probs_sum + (neg_probs_sum == 0)), probs)
        neg_indices = jax.random.choice(negative_key, num_outcomes, shape=(self.num_negatives,), p=neg_probs, replace=True)
        neg_outcomes = outcomes[neg_indices]

        outcome, terminal, reward, chosen_legals = self.game.apply_action(carry.game_state, chosen_outcome)

        next_pending_negative_obs = vectorized_encode_negative(carry.game_state, neg_outcomes)

        return outcome, terminal, reward, chosen_legals, next_pending_negative_obs

      next_game_state, next_terminal, next_rewards, next_legal, next_pending_negative_obs = jax.lax.cond(is_chance,
                                    sample_chance, apply_action)

      #This row's own negatives: real ones handed down from a chance resolution
      # at the previous row (now correctly conditioned on this row's
      # or trivial repeats of this row's own obs if nothing preceded it.
      own_negative_obs = jnp.where(
          carry.prev_is_chance,
          carry.pending_negative_obs,
          jnp.zeros((self.num_negatives, obs.size), dtype=obs.dtype))

      timestep = TimeStep(
        obs = obs,
        negative_obs = own_negative_obs,
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
        prev_action = action_oh,
        prev_is_chance = is_chance,
        pending_negative_obs = next_pending_negative_obs
      )

      timestep = tree_where(carry.valid, timestep, self.example_timestep)

      return new_carry, (timestep, is_chance)

    _, ys = _sample_trajectory(init_carry, recurrent_network, encoder_network, observer_network,
                               actor_network, trajectory_key)
    timestep, is_chance = ys
    #This is used to remove the chance nodes from the trajectory
    non_chance = jnp.nonzero(~is_chance, size=self.non_chance_trajectory_max)[0]
    filtered_timestep = jax.tree.map(lambda x: jnp.take_along_axis(x, jnp.expand_dims(non_chance, axis=range(1, x.ndim)), axis=0).astype(x.dtype), timestep)
    #[Trajectory, ...]
    return filtered_timestep
