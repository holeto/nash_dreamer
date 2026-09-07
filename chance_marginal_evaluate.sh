#!/usr/bin/env bash

# src/ layout: put it on PYTHONPATH (resolved from cwd, since this must be run from
# the project root -- checkpoint/metric paths are also relative to the caller's cwd)
# so `python -m <pkg>.<module>` resolves.
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"

# ------------------------------------------------------------------
# USAGE INSTRUCTIONS
# ------------------------------------------------------------------

# Pass environment variables *before* the script command.
# Assumption: You are already in the project folder. No manual venv activation
# needed -- `uv run` resolves this project's own locked environment regardless of
# what else is active in the shell.
#
# Example (Standard run):
# ./chance_marginal_evaluate.sh
#
# Example (Override game and seeds):
# GAME_NAME="goofspiel_3" SEEDS="(42, 99)" ./chance_marginal_evaluate.sh
#
# Example (Loosen the reconstruction upper bound and cap tree depth):
# UPPER_BOUND=0.5 MAX_DEPTH=8 ./chance_marginal_evaluate.sh
#
# Example (Point directly at a seed-containing directory):
# BASE_PATH="trained_networks/nash_dreamer_rnad/leduc" ./chance_marginal_evaluate.sh
#
# Example (Process every checkpoint in each seed directory, most trained first):
# RESTORE_STEP=-1 ./chance_marginal_evaluate.sh

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------



#Game name, including the parameters.
# As provided by game.to_compact_str
: ${GAME_NAME:="leduc"}
#Root folder of the trained networks
: ${TRAINED_NETWORKS_ROOT:="trained_networks"}
#Directory name of the algorithm to evaluate
: ${ALGO_DIR:="nash_dreamer_rnad"}
#Directory containing seed_{id} subdirectories. Defaults to
# TRAINED_NETWORKS_ROOT/ALGO_DIR/GAME_NAME; set explicitly to override.
: ${BASE_PATH:="$TRAINED_NETWORKS_ROOT"}

: ${SEEDS:="(42,)"}
#Saved step of the model to restore. If < 0, every checkpoint found in each
# seed directory is processed, in descending order (most trained first).
# 0 is a valid checkpoint step, not a sentinel for "process all".
: ${RESTORE_STEP:=30000}

#Reconstruction upper bound; a model continuation is an "error" if its closest
# real outcome has L-inf reconstruction distance >= 1 - UPPER_BOUND.
: ${UPPER_BOUND:=0.7}
#Threshold for filtering the stochastic-state categories per class.
: ${PROBABILITY_EPS:=0.05}
#Actions with model policy below this are pruned (then the policy is renormalized).
: ${POLICY_EPS:=0.05}
#Maximum game-tree depth to walk (safety cap).
: ${MAX_DEPTH:=16}

#Directory to save results into, as OUTPUT_DIR/seed_{id}/chance_marginal_eval.pkl.
# Leave empty to save directly inside each seed's own model directory instead.
: ${OUTPUT_DIR:="world_model_metrics/chance_distribution"}
#Set to any non-empty value to print per-transition progress.
: ${VERBOSE:=""}

# Debug Output
echo "------------------------------------------------"
echo "Mode:             Local"
echo "Game:             $GAME_NAME"
echo "Seeds:            $SEEDS"
echo "Algo dir          $ALGO_DIR"
echo "Restore step:     $RESTORE_STEP"
echo "Upper bound:      $UPPER_BOUND"
echo "Probability eps:  $PROBABILITY_EPS"
echo "Policy eps:       $POLICY_EPS"
echo "Max depth:        $MAX_DEPTH"
echo "Base path:        $BASE_PATH"
echo "Output dir:       ${OUTPUT_DIR:-<per-seed directory>}"
echo "------------------------------------------------"

args=(
  --base_path "$BASE_PATH"
  --seeds "$SEEDS"
  --restore_step "$RESTORE_STEP"
  --game_dir "$GAME_NAME"
  --algo_dir "$ALGO_DIR"
  --upper_bound "$UPPER_BOUND"
  --probability_eps "$PROBABILITY_EPS"
  --policy_eps "$POLICY_EPS"
  --max_depth "$MAX_DEPTH"
)
[ -n "$OUTPUT_DIR" ] && args+=(--output_dir "$OUTPUT_DIR")
[ -n "$VERBOSE" ] && args+=(--verbose)

uv run python -m world_model_experiments.chance_marginal_eval "${args[@]}"
