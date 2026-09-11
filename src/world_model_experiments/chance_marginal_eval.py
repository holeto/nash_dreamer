"""General world-model accuracy experiment: at every node reached under the model's own
policy, compare the world model's predicted (prior) next-observation distribution against
the true one.

Generalizes the Leduc-specific `decode_outcomes` approach previously used for endgame
evaluation to any JaxGame. The tree is walked following
the model policy (actions below --policy_eps pruned and renormalized). At each transition
the model prior dynamics(recurrent) is enumerated (filtered stochastic continuations), each
continuation is decoded to an observation and snapped to the closest real reachable outcome
observation by max-per-element absolute (L-infinity) distance; a continuation whose closest
distance is >= (1 - upper_bound) is counted as an "error". Chance nodes are snapped to their
valid outcomes; deterministic nodes are treated as a single-outcome (one-hot) distribution.

Stores a pickle list with one record per transition:
    {depth, history, is_chance, reach_prob, gt_dist, model_dist, error, outcome_ids}.
"""

from argparse import ArgumentParser
import os
import pickle
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

from nash_dreamer.dreamer_ma import DreamerMA
from nash_dreamer.train_utils import load_model, parse_sequence
from eval.eval_utils import unroll_chance_node, cartesian_product
from nash_dreamer.distributions import joint_to_grid


parser = ArgumentParser(description="Evaluate world-model chance/next-observation marginal "
                                    "accuracy by walking the game tree under the model policy.")
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
parser.add_argument("--upper_bound", type=float, default=0.7,
                    help="Reconstruction upper bound; a continuation is an 'error' if its closest "
                         "real outcome has L-inf reconstruction distance >= 1 - upper_bound.")
parser.add_argument("--probability_eps", type=float, default=0.05,
                    help="Threshold for filtering the stochastic-state categories per class.")
parser.add_argument("--policy_eps", type=float, default=0.05,
                    help="Actions with model policy < policy_eps are pruned; the policy is then "
                         "renormalized over the survivors before expanding.")
parser.add_argument("--max_depth", type=int, default=16,
                    help="Maximum game-tree depth to walk (safety cap).")
parser.add_argument("--output_dir", type=str, default=None,
                    help="Directory to save results into, as output_dir/seed_{id}/chance_marginal_eval.pkl. "
                         "Defaults to saving directly inside each seed's own model directory.")
parser.add_argument("--verbose", action="store_true", help="Print progress.")


def _parse_checkpoint_step(filename: str) -> int | None:
    """Return the trailing integer step of a `..._<step>.pkl` checkpoint filename, or
    None if it doesn't match (e.g. this script's own `..._chance_marginal_eval.pkl`
    output files, which may sit alongside checkpoints in the same seed directory)."""
    parts = filename.split(".")
    if len(parts) != 2 or parts[1] != "pkl":
        return None
    try:
        return int(parts[0].split("_")[-1])
    except ValueError:
        return None


def discover_checkpoint_steps(model_dir: str) -> list[int]:
    """List the distinct saved steps of *_<step>.pkl checkpoint files in model_dir,
    sorted ascending."""
    steps = set()
    for filename in os.listdir(model_dir):
        if not os.path.isfile(os.path.join(model_dir, filename)):
            continue
        step = _parse_checkpoint_step(filename)
        if step is not None:
            steps.add(step)
    return sorted(steps)


def load_checkpoint(model_dir: str, step: int) -> DreamerMA:
    """Load the first *_<step>.pkl checkpoint file found in model_dir."""
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


def enumerate_prior_codes(ma_rssm, recurrent, probability_eps: float):
    """The prior at `recurrent` as explicit (one-hot grid, probability) pairs.

    The ONLY place in this file that needs to know which prior head is in use.

    A JOINT prior (--joint_prior) is already a distribution over all K ** C codes, so it is
    filtered directly. probability_eps is rescaled by the ratio of the two uniform masses,
    (1/K**C) / (1/K): a per class threshold compared against 1/K**C instead of 1/K would be
    wrong by K**(C-1) and, at the 0.05 default with a (2, 6) latent, would prune every code but
    the most likely one. The most likely code is always kept so the enumeration is never empty.

    A FACTORED prior is handled by the caller's original per class path, left untouched so
    previously computed metrics stay comparable."""
    classes, categories = ma_rssm.encoded_classes, ma_rssm.encoded_categories
    probs = np.asarray(jax.nn.softmax(ma_rssm.get_dynamics(recurrent), axis=-1))
    eps = probability_eps / (categories ** (classes - 1))
    keep = np.nonzero(probs >= eps)[0]
    if keep.size == 0:
        keep = np.array([int(np.argmax(probs))])
    total = probs[keep].sum()
    grids = np.asarray(joint_to_grid(jnp.asarray(keep), classes, categories))
    return list(zip(grids, probs[keep] / (total if total > 0 else 1.0)))


def _filter_stoch(logits, probability_eps: float):
    """Softmax -> zero out categories below eps -> renormalize (per class)."""
    s = np.asarray(jax.nn.softmax(logits, axis=-1))
    s = s * (s >= probability_eps)
    total = np.sum(s, axis=-1, keepdims=True)
    return s / np.where(total > 0, total, 1.0)


def run_for_model(model: DreamerMA, args) -> list[dict]:
    """Walk the game tree under model's own policy, evaluating chance/next-observation
    marginal accuracy at every transition. Returns the list of transition records."""
    ma_rssm = model.optimizer.model
    game = model.game
    use_real_infoset = model.use_real_infoset
    num_players = game.num_players()
    num_actions = game.num_distinct_actions()
    error_threshold = 1.0 - args.upper_bound
    prob_eps = args.probability_eps
    policy_eps = args.policy_eps
    #--joint_prior replaces the C independent categoricals with one distribution over all K ** C
    # codes, so the prior can no longer be filtered or expanded per class.
    joint_prior = bool(getattr(ma_rssm, "joint_prior", False))

    def get_both_obs(state):
        _, p1_obs, p2_obs, _ = game.get_info(state)
        return jnp.stack([p1_obs, p2_obs], axis=0)

    vectorized_get_obs = jax.vmap(get_both_obs, in_axes=(0,), out_axes=(0))

    records = []
    node_count = [0]
    start_time = time.time()

    def evaluate_marginal(prior_stoch, recurrent, target_obs):
        """prior_stoch [classes, categories], or None when the prior is a joint head;
        target_obs [K, players, obs_dim].
        Returns (model_dist [K], error, per_outcome_deter [K])."""
        if prior_stoch is None:
            #Joint head: the codes and their probabilities come straight off the joint.
            codes = enumerate_prior_codes(ma_rssm, recurrent, prob_eps)
        else:
            num_classes, num_categories = prior_stoch.shape
            mask = prior_stoch >= prob_eps
            per_class = []
            for c in range(num_classes):
                cats = np.nonzero(mask[c])[0]
                if cats.size == 0:  # keep at least the argmax so the product is never empty
                    cats = np.array([int(np.argmax(prior_stoch[c]))])
                per_class.append(cats)
            combos = cartesian_product(*per_class)  # [num_combos, num_classes]
            codes = [(jax.nn.one_hot(comb, num_categories),
                      float(np.prod(prior_stoch[np.arange(num_classes), comb])))
                     for comb in combos]

        K = target_obs.shape[0]
        model_dist = np.zeros(K)
        error = 0.0
        best_dist = np.full(K, np.inf)
        best_deter = [None] * K
        for deter, prob in codes:
            decoded = np.asarray(ma_rssm.get_decoder(recurrent, deter))  # [players, obs_dim]
            dists = np.max(np.abs(decoded[None] - target_obs), axis=(-1, -2))  # [K]
            j = int(np.argmin(dists))
            if dists[j] >= error_threshold:
                error += prob
            else:
                model_dist[j] += prob
            closer = dists < best_dist
            for k in np.nonzero(closer)[0]:
                best_dist[k] = dists[k]
                best_deter[k] = deter
        return model_dist, error, best_deter

    def process_transition(recurrent, action_oh, parent_infoset, reach_prob,
                           child_state, child_terminal, child_legals, depth, history):
        """Evaluate the prior at `recurrent` against the observation(s) reachable at
        child_state, record it, and return the recursion tuples for its non-terminal
        continuations."""
        node_count[0] += 1
        if args.verbose and node_count[0] % 200 == 0:
            print(f"  {node_count[0]} transitions, depth {depth}, {time.time() - start_time:.1f}s")
        #A joint prior is enumerated inside evaluate_marginal instead -- its codes cannot be
        # filtered per class, which is the entire point of the parameterization.
        prior = None if joint_prior else _filter_stoch(ma_rssm.get_dynamics(recurrent), prob_eps)

        if game.is_chance(child_state):
            num_valid = int(game.depth_chance_valid_outcomes(depth))
            out_states, out_term, _, out_legals, out_probs = unroll_chance_node(
                game, child_state, num_valid)
            out_term = np.asarray(out_term)
            out_legals = np.asarray(out_legals)
            gt_dist = np.asarray(out_probs)
            target_obs = np.asarray(vectorized_get_obs(out_states))  # [K, players, obs_dim]

            outcomes_all, probs_all = game.get_outcomes_and_probs(child_state)
            valid_idx = np.nonzero(np.asarray(probs_all) >= 1e-5)[0][:num_valid]
            outcome_ids = np.asarray(outcomes_all)[valid_idx, 1].astype(int)

            model_dist, error, best_deter = evaluate_marginal(prior, recurrent, target_obs)
            records.append(dict(depth=depth, history=history, is_chance=True,
                                reach_prob=reach_prob, gt_dist=gt_dist, model_dist=model_dist,
                                error=error, outcome_ids=outcome_ids))

            following = []
            for i in range(num_valid):
                if bool(out_term[i]):
                    continue
                child_i = jax.tree.map(lambda x: x[i], out_states)
                inf_i = ma_rssm.get_next_infoset_all(parent_infoset, target_obs[i], action_oh)
                following.append((recurrent, best_deter[i], child_i, out_legals[i], inf_i,
                              reach_prob, depth + 1, history + f"o{i}"))
            return following

        target_obs = np.asarray(get_both_obs(child_state))[None]  # [1, players, obs_dim]
        model_dist, error, best_deter = evaluate_marginal(prior, recurrent, target_obs)
        records.append(dict(depth=depth, history=history, is_chance=False,
                            reach_prob=reach_prob, gt_dist=np.array([1.0]),
                            model_dist=model_dist, error=error, outcome_ids=np.array([0])))
        if bool(child_terminal):
            return []
        inf = ma_rssm.get_next_infoset_all(parent_infoset, target_obs[0], action_oh)
        return [(recurrent, best_deter[0], child_state, np.asarray(child_legals), inf,
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
            action_prob = float(np.prod(renorm_pi[np.arange(a.shape[0]), a]))
            a_oh = jax.nn.one_hot(a, num_actions)
            a_recurrent = ma_rssm.get_next_recurrent(recurrent, deter, a_oh)
            child, child_term, _, child_legals = game.apply_action(state, a)
            following = process_transition(
                a_recurrent, a_oh, joint_infoset, reach_prob * action_prob,
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

    print(f"  {len(records)} transition records in {time.time() - start_time:.2f}s")
    return records


def summarize(records: list[dict]):
    """Print the mean error / total-variation summary, split by node type."""
    for is_chance, name in [(True, "chance"), (False, "deterministic")]:
        group = [r for r in records if r["is_chance"] == is_chance]
        if not group:
            continue
        mean_err = float(np.mean([r["error"] for r in group]))
        mean_tv = float(np.mean([
            0.5 * (np.sum(np.abs(r["gt_dist"] - r["model_dist"])) + r["error"]) for r in group]))
        print(f"  {name} nodes: {len(group)} | mean error {mean_err:.4f} | "
              f"mean total-variation {mean_tv:.4f}")


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
            records = run_for_model(model, args)

            filename = f"step_{step}_chance_marginal_eval.pkl" if multi_step else "chance_marginal_eval.pkl"
            if args.output_dir:
                out_dir = os.path.join(args.output_dir, args.game_dir, f"seed_{seed}")
                os.makedirs(out_dir, exist_ok=True)
            else:
                out_dir = seed_dir
            output_path = os.path.join(out_dir, filename)
            with open(output_path, "wb") as f:
                pickle.dump(records, f)
            print(f"  Saved {len(records)} transition records to {output_path}")
            summarize(records)

    print(f"\nTotal evaluation time: {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
