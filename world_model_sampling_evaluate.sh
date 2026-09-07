#!/usr/bin/env bash

# ------------------------------------------------------------------
# USAGE INSTRUCTIONS
# ------------------------------------------------------------------

# Pass environment variables *before* the script command.
# Assumption: You are already in the project folder and venv is active.
#
# Example (Standard run):
# ./world_model_sampling_evaluate.sh
#
# Example (Override game and seeds):
# GAME_NAME="goofspiel_3" SEEDS="(42, 99)" ./world_model_sampling_evaluate.sh
#
# Example (Loosen the reconstruction/reward thresholds):
# UPPER_BOUND=0.5 REWARD_THRESHOLD=0.3 ./world_model_sampling_evaluate.sh
#
# Example (More/bigger batches):
# BATCH_SIZE=4096 NUM_BATCHES=8 ./world_model_sampling_evaluate.sh
#
# Example (Process every checkpoint in each seed directory, most trained first):
# RESTORE_STEP=-1 ./world_model_sampling_evaluate.sh

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
#Base path passed to the python script; it appends ALGO_DIR/GAME_NAME itself
# to form base_path/algo_dir/game_dir/seed_{id}. Set explicitly to override.
: ${BASE_PATH:="$TRAINED_NETWORKS_ROOT"}

: ${SEEDS:="(42,)"}
#Saved step of the model to restore. If < 0, every checkpoint found in each
# seed directory is processed, in descending order (most trained first).
# 0 is a valid checkpoint step, not a sentinel for "process all".
: ${RESTORE_STEP:=30000}

#Trajectories sampled per batch, and number of batches per model.
: ${BATCH_SIZE:=2058}
: ${NUM_BATCHES:=4}

#Observation reconstruction upper bound; a step is an "error" if the L-inf
# distance between the real and reconstructed observation is >= 1 - UPPER_BOUND.
: ${UPPER_BOUND:=0.7}
#A step is an "error" if |real reward - reconstructed reward| > REWARD_THRESHOLD.
: ${REWARD_THRESHOLD:=0.2}
#Uniform mix into the sampling policy (0 = pure model policy).
: ${SAMPLING_EPSILON:=0.0}
#Threshold for the posterior stochastic-state sampling at the root.
: ${SAMPLE_THRESHOLD:=0.05}
#Imagined trajectory length. <= 0 derives it from game.max_trajectory_length().
: ${AC_TRAJECTORY_LEN:=0}
#RNG seed for sampling.
: ${RNG_SEED:=0}

#Directory to save results into, as OUTPUT_DIR/seed_{id}/rollout_validity.json.
# Leave empty to save directly inside each seed's own model directory instead.
: ${OUTPUT_DIR:="world_model_metrics/rollout_validity"}
#Set to any non-empty value to print per-batch progress.
: ${VERBOSE:=""}

# Debug Output
echo "------------------------------------------------"
echo "Mode:              Local"
echo "Game:              $GAME_NAME"
echo "Seeds:             $SEEDS"
echo "Algo dir:          $ALGO_DIR"
echo "Restore step:      $RESTORE_STEP"
echo "Batch size:        $BATCH_SIZE"
echo "Num batches:       $NUM_BATCHES"
echo "Upper bound:       $UPPER_BOUND"
echo "Reward threshold:  $REWARD_THRESHOLD"
echo "Sampling epsilon:  $SAMPLING_EPSILON"
echo "Sample threshold:  $SAMPLE_THRESHOLD"
echo "AC trajectory len: $AC_TRAJECTORY_LEN"
echo "RNG seed:          $RNG_SEED"
echo "Base path:         $BASE_PATH"
echo "Output dir:        ${OUTPUT_DIR:-<per-seed directory>}"
echo "------------------------------------------------"

args=(
  --base_path "$BASE_PATH"
  --seeds "$SEEDS"
  --restore_step "$RESTORE_STEP"
  --game_dir "$GAME_NAME"
  --algo_dir "$ALGO_DIR"
  --batch_size "$BATCH_SIZE"
  --num_batches "$NUM_BATCHES"
  --upper_bound "$UPPER_BOUND"
  --reward_threshold "$REWARD_THRESHOLD"
  --sampling_epsilon "$SAMPLING_EPSILON"
  --sample_threshold "$SAMPLE_THRESHOLD"
  --ac_trajectory_len "$AC_TRAJECTORY_LEN"
  --rng_seed "$RNG_SEED"
)
[ -n "$OUTPUT_DIR" ] && args+=(--output_dir "$OUTPUT_DIR")
[ -n "$VERBOSE" ] && args+=(--verbose)

python -m world_model_experiments.world_model_sampling_eval "${args[@]}"
