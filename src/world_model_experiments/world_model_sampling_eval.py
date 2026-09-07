"""World-model imagined-rollout validity experiment.

Samples batched imagined rollouts from the world model (mirroring
`WMReplayBuffer.sample_trajectory`): actions are sampled from the model's own joint policy
and the latent from the FILTERED PRIOR (dynamics), while the real game is stepped in
parallel with the same actions. At each step the observation, reward, terminal flag and
legal mask are reconstructed from the model state and compared to the real game. A step is
"erroneous" if the reconstructed observation is L-inf >= 1-upper_bound from the real one,
the reconstructed terminal/legal (already thresholded by ac_config) mismatch the real ones,
or the reward is off by more than reward_threshold. At chance nodes the model's sampled
stochastic state is snapped onto the closest real chance outcome by decoded-observation
distance.

Outputs, per (seed, step):
  1. Fraction of erroneous trajectories (>=1 erroneous valid step) over all sampled.
  2. Per-category (obs, reward, terminal, legal) mean error among erroneous trajectories,
     joint-pooled over all valid steps of those trajectories.
"""

from argparse import ArgumentParser
from functools import partial
import json
import os
import time

import chex
import numpy as np
import jax
import jax.numpy as jnp
import flax.nnx as nnx

from nash_dreamer.distributions import sample_categorical
from nash_dreamer.dreamer_ma import DreamerMA, MARSSM
from envs.jax_game import JaxGame, GameState
from nash_dreamer.train_utils import load_model, parse_sequence


@chex.dataclass(frozen=True)
class SampleData:
    valid: chex.Array
    # Whether the state reached by this step's transition is terminal. Legal-action masks
    # are not meaningfully defined for a terminal state (nobody acts there), so legal_errors
    # is masked out one step earlier than obs/reward/terminal_errors -- see aggregate().
    next_terminal: chex.Array
    obs_errors: chex.Array
    reward_errors: chex.Array
    terminal_errors: chex.Array
    legal_errors: chex.Array
    pi: chex.Array
    action: chex.Array
    pred_reward: chex.Array
    real_reward: chex.Array
    pred_obs: chex.Array
    real_obs: chex.Array


class Sampler:
    def __init__(self, game: JaxGame, batch_size: int = 2058, seed: int = 0,
                 sample_threshold: float = 0.05, sampling_epsilon: float = 0.0,
                 ac_trajectory_len: int | None = None):
        self.game = game
        self.batch_size = batch_size
        self.key = jax.random.key(seed)
        self.sample_threshold = sample_threshold
        self.sampling_epsilon = sampling_epsilon
        self.num_players = game.num_players()
        self.num_actions = game.num_distinct_actions()
        # Each scan step is one transition, predicting the qualities of the NEXT state (chance
        # nodes are resolved within the same step as reaching them, not as their own iteration).
        # A no-chance trajectory of N rows (including the terminal row) therefore needs exactly
        # N-1 transitions -- matching ma_rssm.py's own `ac_trajectory_len` convention for its
        # structurally identical imagine_trajectory scan.
        self.ac_trajectory_len = (ac_trajectory_len if ac_trajectory_len and ac_trajectory_len > 0
                                  else game.max_trajectory_lenght_no_chance() - 1)
        self.encoder_compatible = None

    def get_next_key(self):
        immediate_key, self.key = jax.random.split(self.key)
        return immediate_key

    def check_encoder_compatible(self, ma_rssm: MARSSM) -> bool:
        """Detect once, eagerly (outside any jit/vmap trace), whether ma_rssm.get_encoder_no_jit
        accepts this game's real (recurrent, obs) shapes. Some checkpoints predate a change to
        the Encoder/ObservedPredictor I/O contract (recurrent_state moved from being an input to
        ObservedPredictor to being an input to Encoder) -- a genuine architecture incompatibility,
        not just a missing config field, that raises a shape-mismatch TypeError. Must be called
        before the first call to imagine_trajectories, since that is jitted and the result is
        baked into the trace as a static Python branch."""
        if self.encoder_compatible is not None:
            return self.encoder_compatible
        try:
            state, _ = self.game.initialize_structures()
            _, p1_obs, p2_obs, _ = self.game.get_info(state)
            obs = jnp.stack((p1_obs, p2_obs), axis=0)
            ma_rssm.get_encoder_no_jit(ma_rssm.get_init_recurrent(), obs)
            self.encoder_compatible = True
        except Exception:
            self.encoder_compatible = False
        return self.encoder_compatible

    @partial(nnx.jit, static_argnums=0)
    def imagine_trajectories(self, model: MARSSM, key) -> SampleData:
        keys = jax.random.split(key, self.batch_size)
        batch_sample_trajectory = nnx.vmap(self.imagine_trajectory, in_axes=(0, None), out_axes=1)
        data = batch_sample_trajectory(keys, model)
        return data

    def imagine_trajectory(self, key, ma_rssm: MARSSM) -> SampleData:
        init_sample_key, init_latent_sample_key, traj_key = jax.random.split(key, 3)
        trajectory_keys = jax.random.split(traj_key, self.ac_trajectory_len)

        def get_obs(state):
            _, p1_obs, p2_obs, _ = self.game.get_info(state)
            return jnp.stack((p1_obs, p2_obs), axis=0)

        def snap_to_outcome(game_state, model_obs):
            """Pick the real outcome of game_state's chance node whose observation is closest
            (L-inf) to model_obs -- keeps chance resolution self-consistent with the model's
            own (possibly blind/prior-based) prediction, rather than an independently/randomly
            selected outcome the model had no way to have predicted. Returns (state, terminal,
            reward, legals) for the selected outcome."""
            outcomes, probs = self.game.get_outcomes_and_probs(game_state)
            vmapped_apply = jax.vmap(self.game.apply_action, in_axes=(None, 0))
            o_states, o_term, o_reward, o_legals = vmapped_apply(game_state, outcomes)
            o_obs = jax.vmap(get_obs)(o_states)  # [N, players, obs_dim]
            dists = jnp.max(jnp.abs(model_obs[None] - o_obs), axis=(-1, -2))  # [N]
            dists = jnp.where(probs >= 1e-5, dists, jnp.inf)
            idx = jnp.argmin(dists)
            sel_state = jax.tree.map(lambda x: x[idx], o_states)
            return sel_state, o_term[idx], o_reward[idx], o_legals[idx]

        init_state, init_legals = self.game.initialize_structures()
        init_recurrent = ma_rssm.get_init_recurrent()

        if self.encoder_compatible:
            # Posterior grounding: the real root chance node is resolved independently at
            # random first, and the posterior faithfully encodes whatever real observation
            # results -- self-consistent, since the posterior directly conditions on the one
            # real, known observation (no snapping needed).
            def get_init_chance():
                outcomes, probs = self.game.get_outcomes_and_probs(init_state)
                sampled_outcome = jax.random.choice(init_sample_key, outcomes, p=probs)
                return self.game.apply_action(init_state, sampled_outcome)

            def get_init_no_chance():
                return init_state, jnp.array(False), jnp.array(0.0, dtype=jnp.float32), init_legals

            start_state, start_terminal, start_reward, start_legals = jax.lax.cond(
                self.game.is_chance(init_state), get_init_chance, get_init_no_chance)
            start_obs = get_obs(start_state)
            init_stoch = ma_rssm.get_encoder_no_jit(init_recurrent, start_obs)
            init_deter = sample_categorical(init_stoch, init_latent_sample_key, self.sample_threshold)
            model_root_obs = ma_rssm.get_decoder_no_jit(init_recurrent, init_deter, return_logits=False)
        else:
            # Encoder architecture incompatible with this checkpoint: fall back to the PRIOR,
            # exactly like every other step of the rollout, and resolve the real root chance
            # node via the SAME snapping the scan uses below -- never compare an independently/
            # randomly chosen real outcome against the model's unrelated blind prior guess.
            init_stoch = ma_rssm.get_dynamics_no_jit(init_recurrent)
            init_deter = sample_categorical(init_stoch, init_latent_sample_key,
                                            sample_threshold=ma_rssm.state_sample_threshold)
            model_root_obs = ma_rssm.get_decoder_no_jit(init_recurrent, init_deter, return_logits=False)

            def get_init_chance():
                return snap_to_outcome(init_state, model_root_obs)

            def get_init_no_chance():
                return init_state, jnp.array(False), jnp.array(0.0, dtype=jnp.float32), init_legals

            start_state, start_terminal, start_reward, start_legals = jax.lax.cond(
                self.game.is_chance(init_state), get_init_chance, get_init_no_chance)
            start_obs = get_obs(start_state)
        # Seed the latent infoset by "observing" the real root observation, consistent with the
        # in-scan update (get_next_infoset_all from the previous infoset + obs + action).
        dummy_infoset = jnp.zeros((self.num_players, ma_rssm.latent_infoset_size))
        dummy_action = jnp.zeros((self.num_players, self.num_actions))
        init_latent_infosets = ma_rssm.get_next_infoset_all_no_jit(
            dummy_infoset, start_obs, dummy_action, use_symlog=False)

        # --- Root: evaluate the model's own (init_recurrent, init_deter) against the real
        # first state, exactly like a scan step's post-transition comparison. When encoder-
        # compatible, init_deter came from the POSTERIOR of start_obs, so this doubles as an
        # encoder/decoder round-trip check; when falling back to the prior (see above), it is
        # a genuine blind prediction, same as every other step. Either way it pools into the
        # aggregate statistics with the same fields/masking as the rest of the trajectory. ---
        root_normalization = jnp.sum(start_legals, axis=-1, keepdims=True)
        root_valid = jnp.logical_and(jnp.logical_not(start_terminal), jnp.all(root_normalization > 0))
        root_valid_f = root_valid.astype(jnp.float32)
        root_non_terminal_f = 1.0 - start_terminal.astype(jnp.float32)

        pred_root_reward, pred_root_terminal, pred_root_legal = ma_rssm.get_predictor_no_jit(
            init_recurrent, init_deter)

        root_timestep = SampleData(
            valid=root_valid,
            next_terminal=start_terminal,
            obs_errors=jnp.max(jnp.abs(start_obs - model_root_obs)) * root_valid_f,
            reward_errors=jnp.abs(start_reward - pred_root_reward) * root_valid_f,
            terminal_errors=(pred_root_terminal != start_terminal).astype(jnp.float32) * root_valid_f,
            legal_errors=(jnp.any(pred_root_legal != start_legals).astype(jnp.float32)
                         * root_valid_f * root_non_terminal_f),
            pi=ma_rssm.get_policy_both_no_jit(model_root_obs, start_legals),
            action=-jnp.ones((self.num_players,), dtype=jnp.int32),
            pred_reward=pred_root_reward,
            real_reward=start_reward,
            pred_obs=model_root_obs,
            real_obs=start_obs,
        )

        @chex.dataclass(frozen=True)
        class SampleTrajectoryCarry:
            game_state: GameState
            reward: chex.Array
            terminal: chex.Array
            legal_actions: chex.Array
            recurrent_state: chex.Array
            deter_state: chex.Array
            joint_latent_infoset: chex.Array

        init_carry = SampleTrajectoryCarry(
            game_state=start_state,
            reward=start_reward,
            terminal=start_terminal,
            legal_actions=start_legals,
            recurrent_state=init_recurrent,
            deter_state=init_deter,
            joint_latent_infoset=init_latent_infosets,
        )

        def choice_wrapper(action_key, p):
            action = jax.random.choice(action_key, self.num_actions, p=p)
            action_oh = jax.nn.one_hot(action, self.num_actions)
            return action, action_oh

        vectorized_sample_action = nnx.vmap(choice_wrapper, in_axes=(0, 0), out_axes=0)

        @nnx.scan(in_axes=(nnx.Carry, 0, None), out_axes=(nnx.Carry, 0))
        def _imagine_trajectory(carry: SampleTrajectoryCarry, step_key,
                                ma_rssm: MARSSM) -> tuple[SampleTrajectoryCarry, SampleData]:
            # --- Policy on the model's (imagined) observation. ---
            decoded_obs, *_ = ma_rssm.get_infoset_decoder_all_no_jit(
                carry.joint_latent_infoset, use_symexp=False, return_logits=False)
            policy_obs = decoded_obs if ma_rssm.use_real_infoset else carry.joint_latent_infoset
            pi = ma_rssm.get_policy_both_no_jit(policy_obs, carry.legal_actions)
            normalization = jnp.sum(carry.legal_actions, axis=-1, keepdims=True)
            uniform_pi = carry.legal_actions / (normalization + (normalization == 0))
            pi = self.sampling_epsilon * uniform_pi + (1 - self.sampling_epsilon) * pi

            action_sample_key, state_sample_key = jax.random.split(step_key)
            action_sample_keys = jax.random.split(action_sample_key, self.num_players)
            action, action_oh = vectorized_sample_action(action_sample_keys, pi)

            # --- Advance the model with the sampled action + prior-sampled latent. ---
            next_recurrent = ma_rssm.get_next_recurrent_no_jit(
                carry.recurrent_state, carry.deter_state, action_oh)
            next_stoch = ma_rssm.get_dynamics_no_jit(next_recurrent)
            next_deter = sample_categorical(next_stoch, state_sample_key,
                                            sample_threshold=ma_rssm.state_sample_threshold)
            # Centralized decoder: the imagined observation AFTER playing the action.
            model_next_obs = ma_rssm.get_decoder_no_jit(next_recurrent, next_deter, return_logits=False)

            # --- Step the real game with the same action; snap chance nodes. ---
            def apply_branch(_):
                return self.game.apply_action(carry.game_state, action)

            def snap_branch(_):
                return snap_to_outcome(carry.game_state, model_next_obs)

            real_next_state, real_next_terminal, real_next_reward, real_next_legals = jax.lax.cond(
                self.game.is_chance(carry.game_state), snap_branch, apply_branch, operand=None)

            # --- Absorb any further chance node(s) reached by that action/snap into this same
            # transition, snapping each against the SAME model_next_obs (the model's dynamics
            # were themselves trained on chance-filtered trajectories, so one predicted
            # observation already stands for "at the next decision/terminal state", regardless
            # of how many raw chance hops separate it from carry.game_state -- see
            # replay_buffer.filter_chance_rewards). Without this, a game whose chance and play
            # nodes alternate (e.g. JaxRandomGoofspiel) leaves real_next_state sitting ON an
            # unresolved chance node, whose zeroed-out tensors get compared against the model's
            # genuine decision prediction one step later than they should.
            def chance_absorb_cond(loop_state):
                state, terminal, _, _ = loop_state
                return jnp.logical_and(self.game.is_chance(state), jnp.logical_not(terminal))

            def chance_absorb_body(loop_state):
                state, _, reward_acc, _ = loop_state
                next_state, next_terminal, next_reward, next_legals = snap_to_outcome(
                    state, model_next_obs)
                return next_state, next_terminal, reward_acc + next_reward, next_legals

            real_next_state, real_next_terminal, real_next_reward, real_next_legals = jax.lax.while_loop(
                chance_absorb_cond, chance_absorb_body,
                (real_next_state, real_next_terminal, real_next_reward, real_next_legals))
            real_next_obs = get_obs(real_next_state)

            # --- Reconstruct reward/terminal/legal from the model state. ---
            pred_reward, pred_terminal, pred_legal = ma_rssm.get_predictor_no_jit(
                next_recurrent, next_deter)

            # --- Per-category error magnitudes (masked by validity of this step). ---
            # obs/reward/terminal are meaningful even for a transition INTO a terminal state
            # (the observation, the reward for reaching it, and the terminal flag itself are
            # all well-defined there); the legal-action mask is not (nobody acts in a terminal
            # state), so legal_err is additionally masked out one step earlier than the others.
            valid = jnp.logical_and(jnp.logical_not(carry.terminal), jnp.all(normalization > 0))
            valid_f = valid.astype(jnp.float32)
            non_terminal_f = 1.0 - real_next_terminal.astype(jnp.float32)
            obs_err = jnp.max(jnp.abs(real_next_obs - model_next_obs)) * valid_f
            reward_err = jnp.abs(real_next_reward - pred_reward) * valid_f
            terminal_err = (pred_terminal != real_next_terminal).astype(jnp.float32) * valid_f
            legal_err = jnp.any(pred_legal != real_next_legals).astype(jnp.float32) * valid_f * non_terminal_f

            next_latent_infoset = ma_rssm.get_next_infoset_all_no_jit(
                carry.joint_latent_infoset, model_next_obs, action_oh, use_symlog=False)

            new_carry = SampleTrajectoryCarry(
                game_state=real_next_state,
                reward=real_next_reward,
                terminal=jnp.logical_or(carry.terminal, real_next_terminal),
                legal_actions=real_next_legals,
                recurrent_state=next_recurrent,
                deter_state=next_deter,
                joint_latent_infoset=next_latent_infoset,
            )
            timestep = SampleData(
                valid=valid,
                next_terminal=real_next_terminal,
                obs_errors=obs_err,
                reward_errors=reward_err,
                terminal_errors=terminal_err,
                legal_errors=legal_err,
                pi=pi,
                action=action,
                pred_reward=pred_reward,
                real_reward=real_next_reward,
                pred_obs=model_next_obs,
                real_obs=real_next_obs,
            )
            return new_carry, timestep

        _, timestep = _imagine_trajectory(init_carry, trajectory_keys, ma_rssm)
        # Prepend the root's own comparison as the first row, so it pools into aggregate()
        # exactly like every other step. [Trajectory, ...] -> [1 + Trajectory, ...]
        full_timestep = jax.tree.map(
            lambda root, rest: jnp.concatenate([root[None], rest], axis=0), root_timestep, timestep)
        return full_timestep


class RunningAverage:
    """Online weighted running average over a stream of (batch_sum, batch_n) pairs, without
    ever holding the raw data behind them. Letting n_t be the cumulative weight after merging
    batch t (n_0 = 0), the update is
        avg_t = avg_{t-1} * (n_{t-1} / n_t) + batch_sum / n_t,  n_t = n_{t-1} + batch_n
    which is algebraically the same running mean as summing every batch_sum and dividing by
    the total weight at the end (avg_t * n_t = avg_{t-1} * n_{t-1} + batch_sum by construction),
    just computed incrementally so only one batch's worth of data need exist at a time."""
    def __init__(self):
        self.avg = 0.0
        self.n = 0

    def update(self, batch_sum: float, batch_n: int):
        new_n = self.n + batch_n
        if new_n > 0:
            self.avg = self.avg * (self.n / new_n) + batch_sum / new_n
        self.n = new_n


def compute_batch_stats(data: SampleData, obs_threshold: float, reward_threshold: float) -> dict:
    """Per-batch local statistics -- the same per-batch quantities the old batch-concatenating
    aggregate() computed in one pass, just scoped to a single SampleData so the caller can
    discard it immediately afterward instead of accumulating every batch in memory."""
    valid = np.asarray(data.valid)                 # [T, B]
    obs = np.asarray(data.obs_errors)
    reward = np.asarray(data.reward_errors)
    terminal_err = np.asarray(data.terminal_errors)
    legal = np.asarray(data.legal_errors)
    next_terminal = np.asarray(data.next_terminal)

    # errors are already valid-masked to 0 on invalid steps, so thresholds never trip there
    # (and, being valid-masked, an erroneous step is necessarily a valid step). legal_errors
    # is further masked to 0 wherever next_terminal is True (no legal mask is defined for a
    # terminal state), so it never trips the threshold there either.
    erroneous_step = ((obs >= obs_threshold) | (reward > reward_threshold)
                      | (terminal_err > 0) | (legal > 0))                                     # [T, B]
    traj_erroneous = erroneous_step.any(axis=0)                                           # [B]

    # obs/reward/terminal are evaluated at every valid step, including the one transitioning
    # into a terminal state; legal is evaluated one step less (excluding that transition).
    legal_valid = valid * (1.0 - next_terminal.astype(np.float64))
    return dict(
        num_traj=int(valid.shape[1]),
        num_err_traj=int(traj_erroneous.sum()),
        total_valid_steps=int(valid.sum()),
        num_err_steps=int(erroneous_step.sum()),
        denom=float(valid[:, traj_erroneous].sum()),
        legal_denom=float(legal_valid[:, traj_erroneous].sum()),
        obs_sum=float(obs[:, traj_erroneous].sum()),
        reward_sum=float(reward[:, traj_erroneous].sum()),
        terminal_sum=float(terminal_err[:, traj_erroneous].sum()),
        legal_sum=float(legal[:, traj_erroneous].sum()),
    )


class RunningRolloutStats:
    """Merges a stream of per-batch compute_batch_stats() dicts into the same aggregate shape
    the old all-batches-at-once aggregate() returned, via RunningAverage for every fraction/
    mean-valued field (each with its own weight n: valid steps, trajectory count, or
    erroneous-trajectory valid/legal steps -- these differ per field, so each gets its own
    running (avg, n) pair rather than sharing one global step counter)."""
    def __init__(self):
        self.steps_avg = RunningAverage()      # fraction_erroneous_steps, n = total_valid_steps
        self.traj_avg = RunningAverage()       # fraction_erroneous_trajectories, n = num_trajectories
        self.obs_avg = RunningAverage()        # n = erroneous_trajectory_valid_steps
        self.reward_avg = RunningAverage()     # n = erroneous_trajectory_valid_steps
        self.terminal_avg = RunningAverage()   # n = erroneous_trajectory_valid_steps
        self.legal_avg = RunningAverage()      # n = erroneous_trajectory_legal_steps
        self.num_err_steps = 0
        self.num_err_traj = 0

    def update(self, stats: dict):
        self.steps_avg.update(stats["num_err_steps"], stats["total_valid_steps"])
        self.traj_avg.update(stats["num_err_traj"], stats["num_traj"])
        self.obs_avg.update(stats["obs_sum"], stats["denom"])
        self.reward_avg.update(stats["reward_sum"], stats["denom"])
        self.terminal_avg.update(stats["terminal_sum"], stats["denom"])
        self.legal_avg.update(stats["legal_sum"], stats["legal_denom"])
        self.num_err_steps += stats["num_err_steps"]
        self.num_err_traj += stats["num_err_traj"]

    def result(self, obs_threshold: float, reward_threshold: float) -> dict:
        return {
            # Step-level fraction: does not saturate the way the trajectory fraction does for
            # short games, so it is the more informative discriminator of rollout validity.
            "fraction_erroneous_steps": self.steps_avg.avg,
            "num_erroneous_steps": self.num_err_steps,
            "total_valid_steps": self.steps_avg.n,
            # Trajectory-level fraction (>=1 erroneous step in the trajectory), kept for reference.
            "fraction_erroneous_trajectories": self.traj_avg.avg,
            "num_erroneous_trajectories": self.num_err_traj,
            "num_trajectories": self.traj_avg.n,
            "erroneous_trajectory_valid_steps": int(self.obs_avg.n),
            "erroneous_trajectory_legal_steps": int(self.legal_avg.n),
            "per_category_error": {
                "obs": self.obs_avg.avg,
                "reward": self.reward_avg.avg,
                "terminal": self.terminal_avg.avg,
                "legal": self.legal_avg.avg,
            },
            "obs_threshold": obs_threshold,
            "reward_threshold": reward_threshold,
        }


parser = ArgumentParser(description="Evaluate world-model imagined-rollout validity by sampling "
                                    "batched rollouts and comparing reconstructions to the real game.")
parser.add_argument("--base_path", type=str, default="trained_networks",
                    help="Base directory; models load from base_path/algo_dir/game_dir/seed_{id}.")
parser.add_argument("--algo_dir", type=str, default="nash_dreamer_rnad", help="Algorithm subdirectory.")
parser.add_argument("--game_dir", type=str, default="leduc", help="Game subdirectory.")
parser.add_argument("--seeds", type=str, default="(42,)",
                    help="Seeds to evaluate, comma-terminated sequence like '(42, 99,)'.")
parser.add_argument("--restore_step", type=int, default=30000,
                    help="Saved step to restore. If < 0, every checkpoint is processed, descending "
                         "(0 is a valid checkpoint step, not a sentinel for 'process all').")
parser.add_argument("--batch_size", type=int, default=2058, help="Trajectories per sampled batch.")
parser.add_argument("--num_batches", type=int, default=4, help="Number of batches to sample per model.")
parser.add_argument("--upper_bound", type=float, default=0.7,
                    help="Observation reconstruction upper bound; obs error threshold = 1 - upper_bound.")
parser.add_argument("--reward_threshold", type=float, default=0.2,
                    help="A step is erroneous if |real reward - reconstructed reward| > reward_threshold.")
parser.add_argument("--sampling_epsilon", type=float, default=0.0,
                    help="Uniform mix into the sampling policy (0 = pure model policy).")
parser.add_argument("--sample_threshold", type=float, default=0.05,
                    help="Threshold for the posterior stochastic-state sampling at the root.")
parser.add_argument("--ac_trajectory_len", type=int, default=0,
                    help="Imagined trajectory length. <= 0 derives it from game.max_trajectory_length().")
parser.add_argument("--rng_seed", type=int, default=0, help="PRNG seed for sampling.")
parser.add_argument("--output_dir", type=str, default=None,
                    help="Directory to save results into, as output_dir/seed_{id}/rollout_validity.json. "
                         "Defaults to saving directly inside each seed's own model directory.")
parser.add_argument("--verbose", action="store_true", help="Print progress.")


def _parse_checkpoint_step(filename: str) -> int | None:
    """Trailing integer step of a `..._<step>.pkl` checkpoint filename, or None if it does not
    match (e.g. this script's own `*_rollout_validity.json` outputs / other .pkl files)."""
    parts = filename.split(".")
    if len(parts) != 2 or parts[1] != "pkl":
        return None
    try:
        return int(parts[0].split("_")[-1])
    except ValueError:
        return None


def discover_checkpoint_steps(model_dir: str) -> list[int]:
    steps = set()
    for filename in os.listdir(model_dir):
        if not os.path.isfile(os.path.join(model_dir, filename)):
            continue
        step = _parse_checkpoint_step(filename)
        if step is not None:
            steps.add(step)
    return sorted(steps)


def load_checkpoint(model_dir: str, step: int) -> DreamerMA:
    for filename in os.listdir(model_dir):
        if not os.path.isfile(os.path.join(model_dir, filename)):
            continue
        if _parse_checkpoint_step(filename) != step:
            continue
        model = load_model(os.path.join(model_dir, filename))
        assert isinstance(model, DreamerMA), \
            f"The saved model should be DreamerMA, got {model.__class__}"
        print(f"Restored model from {os.path.join(model_dir, filename)}")
        return model
    raise FileNotFoundError(f"No file matching step_{step}.pkl found in {model_dir}.")


def run_for_model(model: DreamerMA, args) -> dict:
    """Sample num_batches of imagined rollouts and aggregate the rollout-validity metrics."""
    sampler = Sampler(model.game, batch_size=args.batch_size, seed=args.rng_seed,
                      sample_threshold=args.sample_threshold, sampling_epsilon=args.sampling_epsilon,
                      ac_trajectory_len=args.ac_trajectory_len)
    ma_rssm = model.optimizer.model
    if not sampler.check_encoder_compatible(ma_rssm):
        print("  Encoder architecture incompatible with this checkpoint -- falling back to "
              "prior + snapping for root grounding.")
    obs_threshold = 1.0 - args.upper_bound
    reward_threshold = args.reward_threshold
    running = RunningRolloutStats()
    start_time = time.time()
    for b in range(args.num_batches):
        data = sampler.imagine_trajectories(ma_rssm, sampler.get_next_key())
        data = jax.tree.map(np.asarray, data)
        running.update(compute_batch_stats(data, obs_threshold, reward_threshold))
        if args.verbose:
            print(f"  batch {b + 1}/{args.num_batches}, {time.time() - start_time:.1f}s")
    result = running.result(obs_threshold, reward_threshold)
    result["batch_size"] = args.batch_size
    result["num_batches"] = args.num_batches
    result["time_seconds"] = time.time() - start_time
    return result


def summarize(result: dict):
    pc = result["per_category_error"]
    print(f"  erroneous steps: {result['num_erroneous_steps']}/{result['total_valid_steps']} "
          f"= {result['fraction_erroneous_steps']:.4f}")
    print(f"  erroneous trajectories: {result['num_erroneous_trajectories']}/{result['num_trajectories']} "
          f"= {result['fraction_erroneous_trajectories']:.4f}")
    print(f"  per-category mean error (erroneous trajs): obs {pc['obs']:.4f} | "
          f"reward {pc['reward']:.4f} | terminal {pc['terminal']:.4f} | legal {pc['legal']:.4f}")


def main():
    args = parser.parse_args()

    base_path = args.base_path
    if not base_path.startswith("/"):
        base_path = os.path.join(os.getcwd(), base_path)
    base_path = os.path.join(base_path, args.algo_dir, args.game_dir)
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Base path {base_path} does not exist.")

    seeds = sorted(parse_sequence(args.seeds))

    start_time = time.time()
    for seed in seeds:
        seed_dir = os.path.join(base_path, f"seed_{seed}")
        if not os.path.exists(seed_dir):
            raise FileNotFoundError(f"Seed directory {seed_dir} does not exist.")

        if args.restore_step < 0:
            steps = sorted(discover_checkpoint_steps(seed_dir), reverse=True)
            if not steps:
                raise FileNotFoundError(f"No checkpoint .pkl files found in {seed_dir}.")
        else:
            steps = [args.restore_step]
        multi_step = len(steps) > 1

        for step in steps:
            print(f"\n=== seed {seed}, step {step} ===")
            model = load_checkpoint(seed_dir, step)
            result = run_for_model(model, args)
            result["seed"] = seed
            result["step"] = step

            filename = f"step_{step}_rollout_validity.json" if multi_step else "rollout_validity.json"
            if args.output_dir:
                out_dir = os.path.join(args.output_dir, f"game_{args.game_dir}", f"seed_{seed}")
                os.makedirs(out_dir, exist_ok=True)
            else:
                out_dir = seed_dir
            output_path = os.path.join(out_dir, filename)
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  Saved metrics to {output_path}")
            summarize(result)

    print(f"\nTotal evaluation time: {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
