#!/usr/bin/env bash

# ------------------------------------------------------------------
# USAGE INSTRUCTIONS
# ------------------------------------------------------------------

# Pass environment variables *before* the script command.
# Assumption: You are already in the project folder and venv is active.
#
# Example (Standard run):
# ./posterior_collapse_evaluate.sh
#
# Example (Override game and seeds):
# GAME_NAME="goofspiel_3" SEEDS="(42, 99)" ./posterior_collapse_evaluate.sh
#
# Example (Point directly at a seed-containing directory):
# BASE_PATH="trained_networks/nash_dreamer_rnad/leduc" ./posterior_collapse_evaluate.sh
#
# Example (Process every checkpoint in each seed directory, most trained first):
# RESTORE_STEP=-1 ./posterior_collapse_evaluate.sh

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

#Threshold for filtering the prior stochastic-state categories per class, used
# only to pick a continuation deter at deterministic (non-chance) nodes.
: ${PROBABILITY_EPS:=0.05}
#Actions with model policy below this are pruned (then the policy is renormalized).
: ${POLICY_EPS:=0.05}
#Maximum game-tree depth to walk (safety cap).
: ${MAX_DEPTH:=16}

#Directory to save results into, as OUTPUT_DIR/seed_{id}/posterior_collapse_eval.pkl.
# Leave empty to save directly inside each seed's own model directory instead.
: ${OUTPUT_DIR:="world_model_metrics/posterior_collapse"}
#Set to any non-empty value to print per-transition progress.
: ${VERBOSE:=""}

# Debug Output
echo "------------------------------------------------"
echo "Mode:             Local"
echo "Game:             $GAME_NAME"
echo "Seeds:            $SEEDS"
echo "Algo dir          $ALGO_DIR"
echo "Restore step:     $RESTORE_STEP"
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
  --probability_eps "$PROBABILITY_EPS"
  --policy_eps "$POLICY_EPS"
  --max_depth "$MAX_DEPTH"
)
[ -n "$OUTPUT_DIR" ] && args+=(--output_dir "$OUTPUT_DIR")
[ -n "$VERBOSE" ] && args+=(--verbose)

python -m world_model_experiments.posterior_collapse_eval "${args[@]}"
