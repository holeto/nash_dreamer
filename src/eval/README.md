# Evaluation

Detailed evaluation reference for `src/eval/`. See the root [README.md](../../README.md) for install
instructions. Every direct call below assumes:

```bash
export PYTHONPATH="$(pwd)/src"
```

Three protocols are covered here. A fourth — exact Phantom TTT exploitability via the external `eas`
treeplex solver — lives in [pttt_exploitability/](pttt_exploitability/) and has its own
[CLAUDE.md](pttt_exploitability/CLAUDE.md); it is not covered on this page.

## NashConv and policy quality

[policy_eval_utils.py](policy_eval_utils.py) computes `nash_conv` / best responses by an **exhaustive
tree walk** that materializes every history at a depth at once and expands each by all `A²` joint
actions before filtering. This is only viable for small games (Goofspiel-3, Leduc, RPS) — it OOMs
within a few plies on large ones (Phantom TTT dies at depth 4 of 17). [actor_critic_evaluate.py](actor_critic_evaluate.py)
evaluates NashConv (`nash_conv`), expected utility (`expected_util`), or smoothed training returns
(`env_return`) for one or more algorithms across seeds, using that tree walk.

```bash
# Evaluate NashDreamer and RNaD on Goofspiel-3 at step 10 000 (default)
./nash_dreamer_evaluate.sh

# Evaluate on Leduc Poker, NashConv metric
GAME_NAME="leduc" METRIC="nash_conv" SCALE_FACTOR=13 ./nash_dreamer_evaluate.sh

# Or call directly:
uv run python -m eval.actor_critic_evaluate \
  --base_path trained_networks \
  --game_name goofspiel_3 \
  --seeds "(42, 99, 160)" \
  --restore_step 10000 \
  loaded --metric nash_conv \
  --algo_dirs "NashDreamer=nash_dreamer_rnad RNaD=rnad"
```

Pass `--restore_step -1` to evaluate all saved checkpoints in the directory. For Leduc Poker, set
`--scale_factor 13` to match the reward scaling from -13 to 13.

`goofspiel_nash.pkl` and `leduc_nash.pkl` are reference equilibria for the `nash` subcommand.
Everything under `metrics/`, `precomputed_metrics/`, `world_model_metrics/` is generated output.

## Head-to-head evaluation

[head_to_head_evaluate.py](head_to_head_evaluate.py) plays two trained policies against each other
and records win rates — an approximate empirical alternative to NashConv for games too large for the
exact tree walk above.

```bash
# NashDreamer (P1) vs RNaD (P2) on Goofspiel-4 at step 10 000
GAME_NAME="goofspiel_4" RESTORE_STEP=10000 ./head_to_head.sh

# Custom algorithm pairing:
GAME_NAME="goofspiel_3" \
  ALGO_DIRS="NashDreamer=nash_dreamer_rnad NashDreamerREINFORCE=nash_dreamer_reinforce" \
  ./head_to_head.sh

# Or call directly:
uv run python -m eval.head_to_head_evaluate \
  --base_path trained_networks \
  --game_name goofspiel_4 \
  --seeds "(42, 99, 160)" \
  --restore_step_a 10000 \
  --restore_step_b 10000 \
  --algo_dirs "NashDreamer=nash_dreamer_rnad RNaD=rnad" \
  --num_games 1024 \
  --metric_store_dir metrics/
```

## Approximate exploitability with PPO (`ppo_exploitability.py`)

[policy_eval_utils.py](policy_eval_utils.py)'s exact best response only scales to tiny games.
[ppo_exploitability.py](ppo_exploitability.py) is the learned stand-in for anything bigger: it trains
single-agent `SimPPO` best responses against a **frozen** opponent checkpoint and immediately plays
each one head-to-head against that opponent, all in one process — so nothing is written to
`trained_networks/` and no PPO checkpoints accumulate, only metrics.

```bash
# Sweep every seed of one algorithm at one fixed step
ALGO_DIR=mmd GAME_NAME=goofspiel_3 RESTORE_STEP=1000 ./ppo_exploitability.sh

# Or evaluate a single checkpoint directly:
OPPONENT_PATH=trained_networks/mmd/goofspiel_3/seed_42/step_1000.pkl ./ppo_exploitability.sh

# Or call the module directly:
uv run python -m eval.ppo_exploitability --algo_dir mmd --game_name goofspiel_3 \
  --opponent_seeds "(42, 99)" --restore_step 1000 --num_steps 1000
```

It sweeps the **seed** directories of one algorithm at one fixed `--restore_step` (or takes a single
`--opponent_path`), training `PPO_SEEDS × PLAYERS` best responses per opponent checkpoint (10 seeds ×
2 players by default), and reports `max_seed BR(p0) + max_seed BR(p1)` as the headline `nash_conv`.
Results are appended to `<metric_store_dir>/ppo_exploitability/<game_name>/approx_exploitability.tsv`.

**The PPO budget (`NUM_STEPS`, `PPO_SEEDS`, `BATCH_SIZE`, `NUM_EPOCHS`) is part of the metric, not a
shortcoming of it.** Best responding to a frozen opponent is far easier than solving the game, so an
unbounded budget saturates — step 0 and step 30000 of a hard game would both score the maximum and
the metric would carry no information. Consequences:

- The number is a **budgeted lower bound**, rather than value comparable with true exploitability.
- The budget must be identical across everything you compare. Every row records it, and the script
  warns when appending to a results file written under a different one.
- `--entropy_coeff` defaults to **0** on purpose: entropy regularization softens the best response,
  which *under-estimates* exploitability .

`--num_steps` defaults per game (1000 for Goofspiel, 3000 for Leduc/Battleships/Strength Duel, 5000
for Phantom TTT — bigger games need more steps to find any exploit at all, without going so far that
the budget saturates). A negative `expl_p` means the search never matched the profile's own value at
that budget; it is reported, not clamped. Watch out as too large budget
against not yet strong enough policies can result in all being maximally 
exploitable and the metric being uninformative.
