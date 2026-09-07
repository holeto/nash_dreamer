"""Decentralized variant of the MA-RSSM world model.

Where `MARSSM` keeps ONE recurrent state and ONE stochastic latent shared by both
players -- fed the joint observation and the joint action -- this variant gives
each player its own RSSM that sees only its own observation and its own action.
The three latent-infoset networks (`InfosetModel`, `InfosetDecoder`,
`InfosetPredictor`) are removed entirely: with `use_original_infoset` the actor
consumes the real infoset on real trajectories and the per-player decoder's
reconstruction during imagination, so nothing needs them.

Weights are shared across players and vmapped over a leading player axis, the
same idiom `ActorNetwork` already uses; the inputs are what is decentralized.

Because a player never observes the opponent's action, its transition model is
not identifiable, and during imagination the two latent chains sample
independently and desynchronize.  That instability is the point of the variant,
not a defect to be fixed.
"""

import jax
import jax.numpy as jnp
import chex
from flax import nnx

from functools import partial

from networks import (ActorNetwork, CriticNetwork, DynamicsPredictor, ObservedPredictor,
                      RewardPredictor, DonePredictor)
from networks_decentralized import (DecSequenceModel, DecEncoder, DecDecoder,
                                    DecLegalActionsNetwork, DecEmbeddingCritic)
from ma_rssm import MARSSM
from optimizer import make_opt
from distributions import sample_categorical
from train_utils import *
from games.jax_game import JaxGame

f32 = jnp.float32
u8 = jnp.uint8


@chex.dataclass(frozen=True)
class DecentralizedPredictionStep():
  """Per-player counterpart of PredictionStepWithLegal.

  Every field carries a leading player axis, and the four latent-infoset fields
  are gone.  `joint_latent_infoset` is kept under its original name so that
  `train_utils.wm_timestep_to_timestep` works on it unchanged; here it holds
  concat(recurrent_state, flat(deter_state)) per player."""
  recurrent_state: chex.Array       # [..., Player, rec_state_size]
  repr_state: chex.Array            # [..., Player, K, C] posterior logits
  tokens: chex.Array                # [..., Player, encoder_tokens]
  deter_state: chex.Array           # [..., Player, K, C] sampled one-hot
  decoded_obs: chex.Array           # [..., Player, observation_size]
  reward_dist_logit: chex.Array     # [..., Player, 2 * bin_range + 1]
  done_logit: chex.Array            # [..., Player, 1]
  legal_logit: chex.Array           # [..., Player, num_actions]
  dynamics_state: chex.Array        # [..., Player, K, C] prior logits
  joint_latent_infoset: chex.Array  # [..., Player, rec_state_size + K * C]


class DecentralizedMARSSM(nnx.Module):

  def __init__(self, game: JaxGame, wm_config: DreamerMAConfig,
               ac_config: RNaDConfig | ActorCriticConfig, rngs: nnx.Rngs, init_seed: int = 0):
    """A decentralized world model for 2p0s games: one RSSM per player, no
    latent-infoset networks.

    Args:
        game (JaxGame): The environment to train on
        wm_config (DreamerMAConfig): A full configuration of the Dreamer world model.
          Its infoset_* fields and beta_infoset are ignored, exactly as MMDConfig
          ignores the fields that only the Dreamer integration reads.
        ac_config (RNaDConfig | ActorCriticConfig): Actor-critic configuration
        init_seed (int): Seeds the constant fallback key of get_next_infoset_all_no_jit
    """
    assert wm_config.use_original_infoset, (
      "The decentralized variant is only defined for real infosets. "
      "Pass --use_original_infoset.")
    self.use_real_infoset = True
    self.obs_loss_bce = wm_config.obs_loss_bce
    self.num_actions = game.num_distinct_actions()
    self.num_players = game.num_players()
    rec_state_size = wm_config.sequential_network_details[0]
    #Unset recurrent state size defaults to a SINGLE player's infoset, not the
    # joint one -- each player only ever carries its own context here.
    if rec_state_size < 1:
      rec_state_size = game.information_state_tensor_shape()
    self.rec_state_size = rec_state_size
    enc_tokens = wm_config.encoder_network_details[0]
    self.encoded_classes = wm_config.encoded_classes
    self.encoded_categories = wm_config.encoded_categories
    #The per-player latent state, used only as imagination plumbing and by the
    # world-model rollout evaluation. Never an actor input in this variant.
    self.latent_infoset_size = rec_state_size + (self.encoded_classes * self.encoded_categories)
    #The actor/critic input is the real infoset.
    self.infoset_size = game.information_state_tensor_shape()
    self.observation_size = game.observation_tensor_shape()
    self.sampling_epsilon = ac_config.sampling_epsilon
    self.legal_threshold = ac_config.legal_threshold
    self.terminal_threshold = ac_config.terminal_threshold
    self.state_sample_threshold = ac_config.state_sample_threshold
    self.wm_bin_range = wm_config.bin_range
    self.use_rnad = isinstance(ac_config, RNaDConfig)
    #Constant fallback key for the key-less latent update; see
    # get_next_infoset_all_no_jit.
    self.default_infoset_seed = init_seed
    #Always imagine at most 1 less than max trajectory
    # length, since the last step is terminal and we do not train on those
    self.ac_trajectory_len = game.max_trajectory_lenght_no_chance() - 1

    self.seq = DecSequenceModel(wm_config.encoded_classes,
                                wm_config.encoded_categories,
                                self.num_actions,
                                wm_config.sequential_network_details[1],
                                wm_config.sequential_network_details[2],
                                rec_state_size,
                                rngs)
    self.rew = RewardPredictor(wm_config.bin_range,
                               wm_config.encoded_classes,
                               wm_config.encoded_categories,
                               rec_state_size,
                               wm_config.reward_predictor_network_details[0],
                               wm_config.reward_predictor_network_details[1],
                               rngs=rngs)
    self.term = DonePredictor(wm_config.encoded_classes,
                              wm_config.encoded_categories,
                              rec_state_size,
                              wm_config.done_predictor_network_details[0],
                              wm_config.done_predictor_network_details[1],
                              rngs)
    self.leg = DecLegalActionsNetwork(self.num_actions,
                                      wm_config.encoded_classes,
                                      wm_config.encoded_categories,
                                      rec_state_size,
                                      wm_config.legal_actions_network_details[0],
                                      wm_config.legal_actions_network_details[1],
                                      rngs)
    self.dyn = DynamicsPredictor(rec_state_size,
                                 wm_config.encoded_classes,
                                 wm_config.encoded_categories,
                                 wm_config.dynamics_network_details[0],
                                 wm_config.dynamics_network_details[1],
                                 rngs=rngs)
    self.enc = DecEncoder(self.observation_size,
                          rec_state_size,
                          enc_tokens,
                          wm_config.encoder_network_details[1],
                          wm_config.encoder_network_details[2],
                          rngs)
    self.dec = DecDecoder(rec_state_size,
                          self.observation_size,
                          wm_config.encoded_classes,
                          wm_config.encoded_categories,
                          wm_config.decoder_network_details[0],
                          wm_config.decoder_network_details[1],
                          rngs)
    self.observer = ObservedPredictor(enc_tokens,
                                      wm_config.encoded_classes,
                                      wm_config.encoded_categories,
                                      wm_config.observer_network_details[0],
                                      wm_config.observer_network_details[1],
                                      rngs)
    self.embed_critic = DecEmbeddingCritic(enc_tokens, self.observation_size, rngs)

    #The last two entries must stay ('actor', 'critic'): DreamerMA slices [:-2]
    # for world-model gradient norms and the actor-critic learners slice [-2:].
    self.network_names = ['dyn', 'seq', 'enc', 'leg', 'observer', 'dec', 'rew', 'term',
                          'embed_critic', 'actor', 'critic']

    self.actor = ActorNetwork(self.infoset_size,
                              self.num_actions,
                              ac_config.actor_network_details[0],
                              ac_config.actor_network_details[1],
                              rngs)
    #Decentralized: the critic sees ONE player's infoset, not the joint one.
    self.critic = CriticNetwork(self.infoset_size,
                                ac_config.bin_range,
                                ac_config.critic_network_details[0],
                                ac_config.critic_network_details[1],
                                rngs)

  def default_ac_timestep(self):
    obs = jnp.zeros((1, self.infoset_size), dtype=f32)

    legal = jnp.ones((1, self.num_actions), dtype=u8)
    action = jnp.ones((1, self.num_actions), dtype=f32)
    policy = jnp.ones((1, self.num_actions), dtype=u8)
    valid = jnp.array(0, dtype=f32)
    reward = jnp.array(0, dtype=f32)

    return ActorCriticTimeStep(
      valid = valid,
      obs = obs,
      legal = legal,
      action = action,
      policy = policy,
      reward = reward
    )

  # --- per-player vmap helpers -------------------------------------------------

  @staticmethod
  def _per_player(net, *args):
    """Apply a shared module to a stack of per-player inputs."""
    return nnx.vmap(MARSSM.call_net, in_axes=(None, *(0 for _ in args)), out_axes=0)(net, *args)

  def _sample_deter(self, stoch_logits: chex.Array, key):
    """Sample each player's stochastic state INDEPENDENTLY.

    Not a stylistic choice: sample_categorical derives its threshold as
    min(sample_threshold, min(max_probs)) over the whole array, so calling it on
    a stacked [Player, K, C] tensor would couple the players' thresholds."""
    keys = jax.random.split(key, self.num_players)
    return jax.vmap(sample_categorical, in_axes=(0, 0, None))(
      stoch_logits, keys, self.state_sample_threshold)

  def _unpack_latent(self, latent: chex.Array):
    """Split concat(recurrent_state, flat(deter_state)) back into its parts."""
    rec = latent[..., :self.rec_state_size]
    deter = latent[..., self.rec_state_size:].reshape(
      *latent.shape[:-1], self.encoded_classes, self.encoded_categories)
    return rec, deter

  def _pack_latent(self, rec: chex.Array, deter: chex.Array):
    return jnp.concatenate([rec, deter.reshape(*deter.shape[:-2], -1)], axis=-1)

  # --- world-model heads -------------------------------------------------------

  @nnx.jit
  def get_predictor(self, recurrent_state: chex.Array, deterministic_state: chex.Array):
    return self.get_predictor_no_jit(recurrent_state, deterministic_state)

  def get_predictor_no_jit(self, recurrent_state: chex.Array, deterministic_state: chex.Array):
    """Per-player reward/done/legal heads, collapsed back to the CENTRALIZED
    return signature (scalar reward, scalar terminal, [Player, A] legals) that
    every caller downstream expects.

    Each player's reward head is trained on its OWN perspective (r for p1, -r for
    p2), so the p1-perspective scalar is recovered as (r_0 - r_1) / 2.  When the
    two heads disagree -- which is exactly what decentralization causes -- this
    averages the disagreement rather than silently trusting one player."""
    reward_bin_logits = self._per_player(self.rew, recurrent_state, deterministic_state)
    done_logit = self._per_player(self.term, recurrent_state, deterministic_state)
    legal_logit = self._per_player(self.leg, recurrent_state, deterministic_state)
    #[Player, 1]
    per_player_reward = get_value_from_bins(reward_bin_logits, self.wm_bin_range)
    reward = (per_player_reward[0, 0] - per_player_reward[1, 0]) / 2
    done_prob = nnx.sigmoid(done_logit)
    #If either player believes the game ended, stop the rollout.
    terminal = jnp.any(done_prob >= self.terminal_threshold)
    legal_prob = nnx.sigmoid(legal_logit)
    legal_actions = (legal_prob >= self.legal_threshold).astype(u8)
    return reward, terminal, legal_actions

  @partial(nnx.jit, static_argnums=(3))
  def get_decoder(self, recurrent_state: chex.Array, deter_state: chex.Array, return_logits=False):
    return self.get_decoder_no_jit(recurrent_state, deter_state, return_logits)

  def get_decoder_no_jit(self, recurrent_state: chex.Array, deter_state: chex.Array, return_logits=False):
    """Each player reconstructs only its OWN observation from its own latent.
    Returns [Player, observation_size] -- the same shape the centralized joint
    decoder produces, so callers do not have to branch."""
    decoder_output = self._per_player(self.dec, recurrent_state, deter_state)
    if return_logits:
      return decoder_output
    if self.obs_loss_bce:
      return (nnx.sigmoid(decoder_output) >= 0.5).astype(f32)
    else:
      return symexp(decoder_output)  # L2: output is in symlog space

  @nnx.jit
  def get_dynamics(self, recurrent_state: chex.Array):
    return self.get_dynamics_no_jit(recurrent_state)

  def get_dynamics_no_jit(self, recurrent_state: chex.Array):
    return self._per_player(self.dyn, recurrent_state)

  @partial(nnx.jit, static_argnums=3)
  def get_encoder(self, recurrent_state: chex.Array, obs: chex.Array, use_symlog=True):
    return self.get_encoder_no_jit(recurrent_state, obs, use_symlog)

  def get_encoder_no_jit(self, recurrent_state: chex.Array, obs: chex.Array, use_symlog=True):
    if not self.obs_loss_bce:
      obs = symlog(obs)
    tokens = self._per_player(self.enc, recurrent_state, obs)
    return self._per_player(self.observer, tokens)

  @nnx.jit
  def get_next_recurrent(self, recurrent_state: chex.Array, deterministic_state: chex.Array, action: chex.Array):
    return self.get_next_recurrent_no_jit(recurrent_state, deterministic_state, action)

  def get_next_recurrent_no_jit(self, recurrent_state: chex.Array, deterministic_state: chex.Array, action: chex.Array):
    """Player i advances on action[i] ONLY. It never sees the opponent's action,
    which is what makes its transition model unidentifiable."""
    return self._per_player(self.seq, recurrent_state, deterministic_state, action)

  @partial(nnx.jit, static_argnums=1)
  def get_init_recurrent(self, n_starts: int = 0):
    dummy_rec = jnp.zeros((self.num_players, self.rec_state_size))
    dummy_deter = jnp.zeros((self.num_players, self.encoded_classes, self.encoded_categories))
    dummy_action = jnp.zeros((self.num_players, self.num_actions))
    init_rec = self.get_next_recurrent_no_jit(dummy_rec, dummy_deter, dummy_action)
    #Just handle 0, or negative value as a special case for only one
    # start, without the leading batch dimension
    if n_starts > 0:
      init_rec = jnp.tile(init_rec[None, ...], (n_starts, 1, 1))
    return init_rec

  # --- latent-state compatibility shims ----------------------------------------
  # Only world_model_experiments/world_model_sampling_eval.py reaches these: it
  # carries a joint_latent_infoset of its own and asks the model to advance and
  # decode it. With real infosets the exploitability path never builds an
  # InformedRealGame, so nothing else depends on them.

  def get_infoset_decoder_all_no_jit(self, joint_latent_infoset: chex.Array,
                                     use_symexp=False, return_logits=False):
    """Reconstruct each player's own observation from its packed latent state.
    Pure unpack-and-decode, so no PRNG key is involved.  The second return value
    stands in for the centralized InfosetDecoder's previous-action head, which
    has no decentralized counterpart and no caller."""
    rec, deter = self._unpack_latent(joint_latent_infoset)
    obs = self.get_decoder_no_jit(rec, deter, return_logits)
    dummy_action = jnp.zeros((*obs.shape[:-1], self.num_actions))
    return obs, dummy_action

  @partial(nnx.jit, static_argnums=(2, 3))
  def get_infoset_decoder_all(self, joint_latent_infoset: chex.Array,
                              use_symexp=False, return_logits=False):
    return self.get_infoset_decoder_all_no_jit(joint_latent_infoset, use_symexp, return_logits)

  def get_next_infoset_all_no_jit(self, joint_latent_infoset: chex.Array, joint_cur_obs: chex.Array,
                                  joint_action: chex.Array, use_symlog=True, key=None):
    """Advance each player's packed latent state after it observes joint_cur_obs
    having played joint_action.

    Re-deriving the latent requires sampling a fresh stochastic state, but the
    call sites in world_model_sampling_eval.py supply no key, so key=None falls
    back to a constant seeded from the model's init_seed.  A constant is
    deliberate: a stateful nnx.Rngs would be frozen anyway wherever a caller
    splits the module once and closes over the captured state, so this is the
    explicit version of what would otherwise happen silently.  Training always
    passes a real split key."""
    rec, deter = self._unpack_latent(joint_latent_infoset)
    new_rec = self.get_next_recurrent_no_jit(rec, deter, joint_action)
    new_stoch = self.get_encoder_no_jit(new_rec, joint_cur_obs)
    if key is None:
      key = jax.random.key(self.default_infoset_seed)
    new_deter = self._sample_deter(new_stoch, key)
    return self._pack_latent(new_rec, new_deter)

  @partial(nnx.jit, static_argnums=4)
  def get_next_infoset_all(self, joint_latent_infoset: chex.Array, joint_cur_obs: chex.Array,
                           joint_action: chex.Array, use_symlog=True, key=None):
    return self.get_next_infoset_all_no_jit(joint_latent_infoset, joint_cur_obs,
                                            joint_action, use_symlog, key)

  # --- policy ------------------------------------------------------------------

  @nnx.jit
  def get_policy(self, obs, legal) -> chex.Array:
    if not self.obs_loss_bce:
      obs = symlog(obs)
    return MARSSM.call_net(self.actor, obs, legal)

  @nnx.jit
  def get_policy_both(self, joint_obs, joint_legal) -> chex.Array:
    return self.get_policy_both_no_jit(joint_obs, joint_legal)

  def get_policy_both_no_jit(self, joint_obs, joint_legal) -> chex.Array:
    if not self.obs_loss_bce and self.use_real_infoset:
      joint_obs = symlog(joint_obs)
    return self._per_player(self.actor, joint_obs, joint_legal)[0]

  # --- imagination -------------------------------------------------------------

  @nnx.jit
  def imagine_trajectories(self, key, starting_points: DecentralizedPredictionStep) -> ActorCriticTimeStep:
    batch_size = starting_points.done_logit.shape[0]
    keys = jax.random.split(key, batch_size)
    batch_sample_trajectory = nnx.vmap(self.imagine_trajectory, in_axes=(0, 0, None), out_axes=1)
    return batch_sample_trajectory(keys, starting_points, self)

  def imagine_trajectory(self, key, starting_point: DecentralizedPredictionStep,
                         ma_rssm) -> ActorCriticTimeStep:
    trajectory_key = jax.random.split(key, self.ac_trajectory_len)
    ac_default = self.default_ac_timestep()

    @chex.dataclass(frozen=True)
    class SampleTrajectoryCarry:
      recurrent_state: chex.Array   # [Player, rec_state_size]
      deter_state: chex.Array       # [Player, K, C]
      legal_actions: chex.Array     # [Player, A]
      terminal: bool

    init_carry = SampleTrajectoryCarry(
      recurrent_state = starting_point.recurrent_state,
      deter_state = starting_point.deter_state,
      legal_actions = (nnx.sigmoid(starting_point.legal_logit) >= self.legal_threshold).astype(u8),
      terminal = jnp.any(nnx.sigmoid(starting_point.done_logit) >= self.terminal_threshold)
    )

    def choice_wrapper(key, p):
      action = jax.random.choice(key, self.num_actions, p=p)
      action_oh = jax.nn.one_hot(action, self.num_actions)
      return action, action_oh

    vectorized_sample_action = nnx.vmap(choice_wrapper, in_axes=(0, 0), out_axes=0)

    @nnx.scan(in_axes = (nnx.Carry, 0, None), out_axes=(nnx.Carry, 0))
    def _imagine_trajectory(carry: SampleTrajectoryCarry, key,
                            ma_rssm) -> tuple[SampleTrajectoryCarry, chex.Array]:

      #There is no real infoset inside imagination, so each player acts on the
      # infoset its OWN decoder reconstructs from its OWN latent. This is where
      # the desynchronization between the two chains becomes visible to the policy.
      obs = ma_rssm.get_decoder_no_jit(carry.recurrent_state, carry.deter_state,
                                       return_logits=False)

      #get policy
      pi = ma_rssm.get_policy_both_no_jit(obs, carry.legal_actions)
      #uniform mix to the policy
      normalization = jnp.sum(carry.legal_actions, axis=-1, keepdims=True)
      uniform_pi = carry.legal_actions / (normalization + (normalization == 0))
      pi = self.sampling_epsilon * uniform_pi + (1 - self.sampling_epsilon) * pi
      # For each player samples a single action

      action_sample_key, state_sample_key = jax.random.split(key)
      action_sample_keys = jax.random.split(action_sample_key, self.num_players)
      action, action_oh = vectorized_sample_action(action_sample_keys, pi)

      #Each player advances its own chain on its own action and samples its own
      # stochastic state independently. Nothing couples the two players here.
      next_recurrent = ma_rssm.get_next_recurrent_no_jit(carry.recurrent_state,
                                                         carry.deter_state, action_oh)
      next_stoch = ma_rssm.get_dynamics_no_jit(next_recurrent)
      next_deter = ma_rssm._sample_deter(next_stoch, state_sample_key)

      next_reward, next_terminal, next_legal = ma_rssm.get_predictor_no_jit(next_recurrent, next_deter)
      next_terminal = jnp.logical_or(carry.terminal, next_terminal)
      # The world model can produce all actions to be invalid
      # even when one of the players does not act, he always has one legal
      # NOOP action. So, if one of the players has all actions invalid, then
      # the state is not valid
      valid = jnp.logical_and(jnp.logical_not(carry.terminal), jnp.all(normalization > 0))
      timestep = ActorCriticTimeStep(
        obs = obs,
        legal = carry.legal_actions.astype(u8),
        action = action_oh.astype(u8),
        policy = pi,
        reward = next_reward,
        valid = valid
      )
      new_carry = SampleTrajectoryCarry(
        recurrent_state = next_recurrent,
        deter_state = next_deter,
        legal_actions = jnp.where(next_terminal, ac_default.legal, next_legal),
        terminal = jnp.logical_or(next_terminal, jnp.logical_not(valid)),
      )

      timestep = tree_where(timestep.valid, timestep, ac_default)
      return new_carry, timestep

    _, timestep = _imagine_trajectory(init_carry, trajectory_key, ma_rssm)
    #[Trajectory, ...]
    return timestep


def create_decentralized_dreamer_optimizer(game: JaxGame, wm_config: DreamerMAConfig,
                                           ac_config: RNaDConfig | ActorCriticConfig,
                                           opt_config: OptimizerConfig, rngs: nnx.Rngs,
                                           init_seed: int = 0, return_tx = False):
  """Create a single optimizer for the entire DecentralizedMARSSM."""
  ma_rssm = DecentralizedMARSSM(game, wm_config, ac_config, rngs, init_seed)
  opt_tx = make_opt(opt_config)
  optimizer = nnx.Optimizer(model=ma_rssm, tx=opt_tx)
  if not return_tx:
    return optimizer
  return optimizer, opt_tx
