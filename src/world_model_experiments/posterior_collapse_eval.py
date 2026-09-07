"""Posterior-collapse experiment: at every chance node reached under the model's own
policy, check whether the encoder's posterior (which sees the true realized outcome)
actually differs from the prior (which doesn't). If the posterior collapses onto the
prior regardless of which outcome occurred, the model isn't using its stochastic state
to represent chance-outcome information at all -- a failure mode distinct from plain
reconstruction error (see chance_marginal_eval.py).

The tree is walked exactly like chance_marginal_eval.py (following the model policy,
actions below --policy_eps pruned and renormalized). At deterministic transitions the
walk simply continues with whichever prior stochastic combo decodes closest (L-inf) to
the real next observation -- same mechanism as chance_marginal_eval.py, just without
recording anything. At a chance node, every valid outcome's true observation is
encoded via the posterior (get_encoder_no_jit) and compared, via KL divergence, against
the (encoder-blind) prior; a uniform mixture (model.wm_config.uniform_mix) is added to
both distributions first, matching training's free-bits computation, so the two always
share full support and the KL is well-defined.

This test requires get_encoder_no_jit and therefore cannot run on checkpoints whose
encoder has a different I/O contract than the current code -- there is no meaningful
fallback for a posterior-vs-prior test when the posterior can't be computed, so it will
simply raise on such checkpoints.

Stores a pickle list with one record per chance node:
    {depth, history, reach_prob, gt_dist, kl_per_outcome, expected_kl, outcome_ids}.
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
from nash_dreamer.train_utils import parse_sequence
from nash_dreamer.distributions import add_uniform_mix, kl_divergence
from eval.eval_utils import unroll_chance_node, cartesian_product
from world_model_experiments.chance_marginal_eval import (
    load_checkpoint, discover_checkpoint_steps, _filter_stoch,
)


parser = ArgumentParser(description="Evaluate world-model posterior collapse at chance "
                                    "nodes by walking the game tree under the model policy.")
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
                    help="Threshold for filtering the prior stochastic-state categories per "
                         "class, used only to pick a continuation deter at deterministic nodes.")
parser.add_argument("--policy_eps", type=float, default=0.05,
                    help="Actions with model policy < policy_eps are pruned; the policy is then "
                         "renormalized over the survivors before expanding.")
parser.add_argument("--max_depth", type=int, default=16,
                    help="Maximum game-tree depth to walk (safety cap).")
parser.add_argument("--output_dir", type=str, default=None,
                    help="Directory to save results into, as "
                         "output_dir/game_dir/seed_{id}/posterior_collapse_eval.pkl. Defaults to "
                         "saving directly inside each seed's own model directory.")
parser.add_argument("--verbose", action="store_true", help="Print progress.")


def run_for_model(model: DreamerMA, args) -> list[dict]:
    """Walk the game tree under model's own policy, evaluating posterior-vs-prior KL
    divergence at every chance node. Returns the list of chance-node records."""
    ma_rssm = model.optimizer.model
    game = model.game
    use_real_infoset = model.use_real_infoset
    num_players = game.num_players()
    num_actions = game.num_distinct_actions()
    uniform_mix = model.wm_config.uniform_mix
    prob_eps = args.probability_eps
    policy_eps = args.policy_eps

    def get_both_obs(state):
        _, p1_obs, p2_obs, _ = game.get_info(state)
        return jnp.stack([p1_obs, p2_obs], axis=0)

    vectorized_get_obs = jax.vmap(get_both_obs, in_axes=(0,), out_axes=(0))

    records = []
    node_count = [0]
    start_time = time.time()

    def select_closest_deter(prior_stoch, recurrent, target_obs):
        """Enumerate prior_stoch's filtered categorical combos, decode each, and
        return whichever one decodes closest (L-inf) to the single real target_obs
        [players, obs_dim]. Mirrors chance_marginal_eval.py's evaluate_marginal, minus
        the error/model_dist bookkeeping this script doesn't need."""
        num_classes, num_categories = prior_stoch.shape
        mask = prior_stoch >= prob_eps
        per_class = []
        for c in range(num_classes):
            cats = np.nonzero(mask[c])[0]
            if cats.size == 0:  # keep at least the argmax so the product is never empty
                cats = np.array([int(np.argmax(prior_stoch[c]))])
            per_class.append(cats)
        combos = cartesian_product(*per_class)  # [num_combos, num_classes]

        best_dist = np.inf
        best_deter = None
        for comb in combos:
            deter = jax.nn.one_hot(comb, num_categories)  # [classes, categories]
            decoded = np.asarray(ma_rssm.get_decoder(recurrent, deter))  # [players, obs_dim]
            dist = float(np.max(np.abs(decoded - target_obs)))
            if dist < best_dist:
                best_dist = dist
                best_deter = deter
        return best_deter

    def evaluate_posterior_collapse(recurrent, target_obs):
        """target_obs [K, players, obs_dim], the true observations of a chance node's K
        valid outcomes. Returns (kl_per_outcome [K], expected_kl, per_outcome_deter [K])
        where expected_kl still needs to be weighted by the caller's gt_dist."""
        prior_probs = np.asarray(jax.nn.softmax(
            add_uniform_mix(ma_rssm.get_dynamics_no_jit(recurrent), uniform_mix), axis=-1))

        K = target_obs.shape[0]
        kl_per_outcome = np.zeros(K)
        per_outcome_deter = [None] * K
        for i in range(K):
            posterior_logits = ma_rssm.get_encoder_no_jit(recurrent, target_obs[i])
            posterior_probs = jax.nn.softmax(add_uniform_mix(posterior_logits, uniform_mix), axis=-1)
            kl_per_outcome[i] = float(kl_divergence(posterior_probs, prior_probs))
            per_outcome_deter[i] = jax.nn.one_hot(
                jnp.argmax(posterior_probs, axis=-1), posterior_probs.shape[-1])
        return kl_per_outcome, per_outcome_deter

    def process_transition(recurrent, action_oh, parent_infoset, reach_prob,
                           child_state, child_terminal, child_legals, depth, history):
        """At a chance node, evaluate posterior-vs-prior KL for every valid outcome and
        record it; at a deterministic node, just pick a continuation deter. Returns the
        recursion tuples for non-terminal continuations."""
        node_count[0] += 1
        if args.verbose and node_count[0] % 200 == 0:
            print(f"  {node_count[0]} transitions, depth {depth}, {time.time() - start_time:.1f}s")

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

            kl_per_outcome, per_outcome_deter = evaluate_posterior_collapse(recurrent, target_obs)
            expected_kl = float(np.sum(gt_dist * kl_per_outcome))
            records.append(dict(depth=depth, history=history, reach_prob=reach_prob,
                                gt_dist=gt_dist, kl_per_outcome=kl_per_outcome,
                                expected_kl=expected_kl, outcome_ids=outcome_ids))

            following = []
            for i in range(num_valid):
                if bool(out_term[i]):
                    continue
                child_i = jax.tree.map(lambda x: x[i], out_states)
                inf_i = ma_rssm.get_next_infoset_all(parent_infoset, target_obs[i], action_oh)
                following.append((recurrent, per_outcome_deter[i], child_i, out_legals[i], inf_i,
                              reach_prob, depth + 1, history + f"o{i}"))
            return following

        target_obs = np.asarray(get_both_obs(child_state))  # [players, obs_dim]
        prior = _filter_stoch(ma_rssm.get_dynamics(recurrent), prob_eps)
        best_deter = select_closest_deter(prior, recurrent, target_obs)
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

    print(f"  {len(records)} chance-node records in {time.time() - start_time:.2f}s")
    return records


def summarize(records: list[dict]):
    """Print the mean/median/min/max expected-KL summary over chance nodes."""
    if not records:
        print("  no chance nodes encountered")
        return
    expected_kls = np.array([r["expected_kl"] for r in records])
    print(f"  chance nodes: {len(records)} | mean expected KL {expected_kls.mean():.4f} | "
          f"median {np.median(expected_kls):.4f} | min {expected_kls.min():.4f} | "
          f"max {expected_kls.max():.4f}")


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

            filename = f"step_{step}_posterior_collapse_eval.pkl" if multi_step else "posterior_collapse_eval.pkl"
            if args.output_dir:
                out_dir = os.path.join(args.output_dir, args.game_dir, f"seed_{seed}")
                os.makedirs(out_dir, exist_ok=True)
            else:
                out_dir = seed_dir
            output_path = os.path.join(out_dir, filename)
            with open(output_path, "wb") as f:
                pickle.dump(records, f)
            print(f"  Saved {len(records)} chance-node records to {output_path}")
            summarize(records)

    print(f"\nTotal evaluation time: {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
