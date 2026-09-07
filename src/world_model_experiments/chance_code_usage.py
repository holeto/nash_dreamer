"""Discrete-code usage experiment: walking the real game tree under the model policy, at
every transition ask which discrete latent codes the POSTERIOR (not the prior -- this is a
representation-drift check) considers plausible, and accumulate a global count per code. The
resulting histogram's spread indicates whether the model's discrete latent space is well
utilized (codes spread out, doing real work) or collapsed (a handful of codes dominate
everywhere).

Continuation mechanism mirrors chance_marginal_eval.py's evaluate_marginal, uniformly at
every node (chance or deterministic): enumerate the posterior's --probability_eps-filtered
categorical combos, decode each, and continue with whichever decodes closest (L-inf) to the
real observation.

IMPORTANT: the posterior is computed with the CLUSTER's Encoder/Observer wiring --
tokens = enc(obs) (obs only), stochastic_state = observer(recurrent_state, tokens)
(contextual on recurrent) -- the inverse of this repo's current get_encoder_no_jit
(enc(recurrent_state, obs) then observer(tokens)). This is scoped to this script only (no
change to networks.py/ma_rssm.py), so it only works correctly against checkpoints/code
environments where that wiring is what was actually trained -- in general, run this under a
cluster-matched code environment, not the current repo's networks.py/ma_rssm.py.

Stores one aggregate JSON per (seed, step) -- not a per-node record list, since only the
final histograms matter here. Counts are split the same way posterior_collapse_eval.py's
data is later clustered for plotting: deterministic nodes get their own histogram, and
chance nodes are further split by their number of valid outcomes (since a chance node's
achievable code spread naturally depends on how many outcomes it has to distinguish) --
unlike posterior_collapse_eval.py, which only evaluates chance nodes, this script also
evaluates and stores deterministic nodes:
    {seed, step, encoded_classes, encoded_categories, num_transitions_evaluated,
     deterministic: {total_stochastic_states_seen, indices, counts, probabilities},
     chance: {"<num_outcomes>": {total_stochastic_states_seen, indices, counts, probabilities}, ...}}

Each combo's identity is captured by a bijective base-num_categories positional index
(combo_index), unique per one-hot-per-row (classes, categories) state, so distinct codes
never collide. Since the full code space (num_categories ** encoded_classes) is normally far
bigger than the number of codes actually observed, "indices" is NOT that raw bijective value
-- it is a dense 1-based rank (1, 2, 3, ...) over only the raw indices observed in that
bucket, ascending. This means a given rank number is only meaningful WITHIN one bucket of one
run: it is not a stable label for "the same code" across different steps/checkpoints or
across different buckets, since which raw indices get observed (and therefore their sort
order) varies run to run.
"""

from argparse import ArgumentParser
import json
import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

from nash_dreamer.dreamer_ma import DreamerMA, MARSSM
from nash_dreamer.train_utils import parse_sequence, symlog
from eval.eval_utils import unroll_chance_node, cartesian_product
from world_model_experiments.chance_marginal_eval import (
    load_checkpoint, discover_checkpoint_steps, _filter_stoch,
)


parser = ArgumentParser(description="Evaluate world-model discrete-code (posterior) usage by "
                                    "walking the game tree under the model policy.")
parser.add_argument("--base_path", type=str, default="trained_networks",
                    help="Base directory containing seed_{id} subdirectories.")
parser.add_argument("--algo_dir", type=str, default="nash_dreamer_rnad", help="The algorithm subdirectory")
parser.add_argument("--game_dir", type=str, default="leduc", help="Name of the game subdirectory")
parser.add_argument("--seeds", type=str, default="(42,)",
                    help="Seeds to evaluate, as a comma-terminated sequence like '(42, 99,)'. "
                         "Parsed with parse_sequence; each is loaded from base_path/seed_{id}.")
parser.add_argument("--restore_step", type=int, default=30000,
                    help="Saved step of the model to restore. If < 0, every checkpoint found "
                         "in the seed directory is processed, in descending order (most trained "
                         "first). 0 is a valid checkpoint step, not a sentinel for 'process all'.")
parser.add_argument("--probability_eps", type=float, default=0.05,
                    help="Threshold for filtering the posterior stochastic-state categories "
                         "per class.")
parser.add_argument("--policy_eps", type=float, default=0.05,
                    help="Actions with model policy < policy_eps are pruned; the policy is then "
                         "renormalized over the survivors before expanding.")
parser.add_argument("--max_depth", type=int, default=16,
                    help="Maximum game-tree depth to walk (safety cap).")
parser.add_argument("--output_dir", type=str, default=None,
                    help="Directory to save results into, as output_dir/game_dir/seed_{id}/"
                         "chance_code_usage.json. Defaults to saving directly inside each "
                         "seed's own model directory.")
parser.add_argument("--verbose", action="store_true", help="Print progress.")


def cluster_get_encoder(ma_rssm: MARSSM, recurrent_state, obs):
    """Cluster wiring: encoder takes obs only, observer is contextual on recurrent_state --
    the inverse of this repo's current get_encoder_no_jit. See module docstring."""
    obs = obs if ma_rssm.obs_loss_bce else symlog(obs)
    tokens = MARSSM.call_net(ma_rssm.enc, obs)
    return MARSSM.call_net(ma_rssm.observer, recurrent_state, tokens)


def combo_index(comb: np.ndarray, num_categories: int) -> int:
    """Bijective base-num_categories positional index for a one-hot-per-row
    (num_classes, num_categories) combo: comb[i] is the i-th digit. Unique per combo (unlike
    a plain sum over classes, which collides whenever two combos share the same digit sum)."""
    return int(sum(int(comb[i]) * (num_categories ** i) for i in range(len(comb))))


def build_histogram(counts: dict[int, int]) -> dict:
    """Sort by the raw bijective index and normalize into a probability histogram. "indices"
    holds dense 1-based ranks over only the raw indices observed here, ascending -- not the
    raw bijective value itself, which lives in a much larger, mostly-unobserved space. See
    module docstring for why ranks aren't comparable across buckets/runs."""
    sorted_raw_indices = sorted(counts.keys())
    total = sum(counts.values())
    return {
        "total_stochastic_states_seen": total,
        "indices": list(range(1, len(sorted_raw_indices) + 1)),
        "counts": [counts[i] for i in sorted_raw_indices],
        "probabilities": [counts[i] / total for i in sorted_raw_indices] if total > 0 else [],
    }


def run_for_model(model: DreamerMA, args) -> dict:
    """Walk the game tree under model's own policy, accumulating posterior discrete-code
    usage counts. Returns the aggregate result dict."""
    ma_rssm = model.optimizer.model
    game = model.game
    use_real_infoset = model.use_real_infoset
    num_players = game.num_players()
    num_actions = game.num_distinct_actions()
    prob_eps = args.probability_eps
    policy_eps = args.policy_eps

    def get_both_obs(state):
        _, p1_obs, p2_obs, _ = game.get_info(state)
        return jnp.stack([p1_obs, p2_obs], axis=0)

    vectorized_get_obs = jax.vmap(get_both_obs, in_axes=(0,), out_axes=(0))

    deterministic_counts: dict[int, int] = {}
    chance_counts: dict[int, dict[int, int]] = {}  # num_valid_outcomes -> {index: count}
    node_count = [0]
    start_time = time.time()

    def evaluate_and_count(posterior_logits, recurrent, target_obs, counts: dict[int, int]):
        """posterior_logits [classes, categories]; target_obs [players, obs_dim]. Enumerates
        the posterior's filtered combos, increments counts[index] for every one (counts is
        the caller-selected bucket -- deterministic_counts, or chance_counts[num_valid]),
        decodes each, and returns whichever decodes closest (L-inf) to target_obs."""
        filtered = _filter_stoch(posterior_logits, prob_eps)
        num_classes, num_categories = filtered.shape
        mask = filtered >= prob_eps
        per_class = []
        for c in range(num_classes):
            cats = np.nonzero(mask[c])[0]
            if cats.size == 0:  # keep at least the argmax so the product is never empty
                cats = np.array([int(np.argmax(filtered[c]))])
            per_class.append(cats)
        combos = cartesian_product(*per_class)  # [num_combos, num_classes]

        best_dist = np.inf
        best_deter = None
        for comb in combos:
            idx = combo_index(comb, num_categories)
            counts[idx] = counts.get(idx, 0) + 1
            deter = jax.nn.one_hot(comb, num_categories)  # [classes, categories]
            decoded = np.asarray(ma_rssm.get_decoder(recurrent, deter))  # [players, obs_dim]
            dist = float(np.max(np.abs(decoded - target_obs)))
            if dist < best_dist:
                best_dist = dist
                best_deter = deter
        return best_deter

    def process_transition(recurrent, action_oh, parent_infoset, reach_prob,
                           child_state, child_terminal, child_legals, depth, history):
        """At every transition (chance-outcome branch or deterministic step), compute the
        POSTERIOR of that transition's real observation, count its plausible codes, and
        return the recursion tuples for its non-terminal continuations."""
        node_count[0] += 1
        if args.verbose and node_count[0] % 200 == 0:
            print(f"  {node_count[0]} transitions, depth {depth}, {time.time() - start_time:.1f}s")

        if game.is_chance(child_state):
            num_valid = int(game.depth_chance_valid_outcomes(depth))
            bucket = chance_counts.setdefault(num_valid, {})
            out_states, out_term, _, out_legals, _ = unroll_chance_node(
                game, child_state, num_valid)
            out_term = np.asarray(out_term)
            out_legals = np.asarray(out_legals)
            target_obs = np.asarray(vectorized_get_obs(out_states))  # [K, players, obs_dim]

            following = []
            for i in range(num_valid):
                posterior_i = cluster_get_encoder(ma_rssm, recurrent, target_obs[i])
                best_deter_i = evaluate_and_count(posterior_i, recurrent, target_obs[i], bucket)
                if bool(out_term[i]):
                    continue
                child_i = jax.tree.map(lambda x: x[i], out_states)
                inf_i = ma_rssm.get_next_infoset_all(parent_infoset, target_obs[i], action_oh)
                following.append((recurrent, best_deter_i, child_i, out_legals[i], inf_i,
                              reach_prob, depth + 1, history + f"o{i}"))
            return following

        target_obs = np.asarray(get_both_obs(child_state))  # [players, obs_dim]
        posterior = cluster_get_encoder(ma_rssm, recurrent, target_obs)
        best_deter = evaluate_and_count(posterior, recurrent, target_obs, deterministic_counts)
        if bool(child_terminal):
            return []
        inf = ma_rssm.get_next_infoset_all(parent_infoset, target_obs, action_oh)
        return [(recurrent, best_deter, child_state, np.asarray(child_legals), inf,
                 reach_prob, depth, history)]

    def walk(recurrent, deter, state, legals, joint_infoset, reach_prob, depth, history):
        if depth > args.max_depth:
            return
        obs = get_both_obs(state)
        policy_obs = obs if use_real_infoset else ma_rssm.get_infoset(recurrent, deter, joint_infoset)
        pi = np.asarray(ma_rssm.get_policy_both(policy_obs, legals))

        pi_mask = pi >= policy_eps
        masked = pi * pi_mask
        row_sums = masked.sum(axis=-1, keepdims=True)
        renorm_pi = np.divide(masked, row_sums, out=np.zeros_like(masked), where=row_sums > 0)
        actions_grid = np.tile(np.arange(pi.shape[-1]), (num_players, 1)).reshape(pi.shape)
        valid_actions = [actions_grid[i][pi_mask[i]] for i in range(num_players)]
        if any(v.size == 0 for v in valid_actions):
            return
        joint_actions = cartesian_product(*valid_actions)
        for a in joint_actions:
            a_oh = jax.nn.one_hot(a, num_actions)
            a_recurrent = ma_rssm.get_next_recurrent(recurrent, deter, a_oh)
            child, child_term, _, child_legals = game.apply_action(state, a)
            following = process_transition(
                a_recurrent, a_oh, joint_infoset, reach_prob,
                child, child_term, np.asarray(child_legals), depth + 1, history + f"a{a}")
            for (rec, det, st, leg, inf, rp, d, hist) in following:
                walk(rec, det, st, leg, inf, rp, d, hist)

    # ---- Root: evaluate the root transition, then descend. ----
    recurrent0 = ma_rssm.get_init_recurrent()
    init_state, init_legals = game.initialize_structures()
    dummy_action = jnp.zeros((num_players, num_actions))
    dummy_infoset = jnp.zeros((num_players, ma_rssm.latent_infoset_size))
    root_following = process_transition(
        recurrent0, dummy_action, dummy_infoset, 1.0, init_state, False,
        np.asarray(init_legals), 0, "")
    for (rec, det, st, leg, inf, rp, d, hist) in root_following:
        walk(rec, det, st, leg, inf, rp, d, hist)

    result = {
        "encoded_classes": int(ma_rssm.encoded_classes),
        "encoded_categories": int(ma_rssm.encoded_categories),
        "num_transitions_evaluated": node_count[0],
        "deterministic": build_histogram(deterministic_counts),
        "chance": {str(n): build_histogram(c) for n, c in sorted(chance_counts.items())},
    }
    num_det_codes = len(deterministic_counts)
    num_chance_codes = sum(len(c) for c in chance_counts.values())
    print(f"  {node_count[0]} transitions evaluated -- deterministic: {num_det_codes} distinct "
          f"codes; chance: {num_chance_codes} distinct codes across "
          f"{len(chance_counts)} outcome-count bucket(s) {sorted(chance_counts.keys())}; "
          f"{time.time() - start_time:.2f}s")
    return result


def main():
    args = parser.parse_args()
    sys.setrecursionlimit(1_000_000)

    base_path = args.base_path
    if not base_path.startswith("/"):
        base_path = os.path.join(os.getcwd(), base_path)
    base_path = os.path.join(base_path, args.algo_dir)
    base_path = os.path.join(base_path, args.game_dir)
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Base path {base_path} does not exist.")

    seeds = sorted(parse_sequence(args.seeds))

    start_time = time.time()
    for seed in seeds:
        seed_dir = os.path.join(base_path, f"seed_{seed}")
        if not os.path.exists(seed_dir):
            raise FileNotFoundError(f"Seed directory {seed_dir} does not exist.")

        if args.restore_step < 0:
            # Most trained first, i.e. descending by step.
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

            filename = f"step_{step}_chance_code_usage.json" if multi_step else "chance_code_usage.json"
            if args.output_dir:
                out_dir = os.path.join(args.output_dir, args.game_dir, f"seed_{seed}")
                os.makedirs(out_dir, exist_ok=True)
            else:
                out_dir = seed_dir
            output_path = os.path.join(out_dir, filename)
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  Saved code-usage histogram to {output_path}")

    print(f"\nTotal evaluation time: {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
