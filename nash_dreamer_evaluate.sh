#!/usr/bin/env bash

# ------------------------------------------------------------------
# USAGE INSTRUCTIONS
# ------------------------------------------------------------------

# Pass environment variables *before* the script command.
# Assumption: You are already in the project folder and venv is active.
#
# Example (Standard run):
# ./nash_dreamer_evaluate.sh
#
# Example (Override Game and metric):
# GAME_NAME="goofspiel_3" METRIC="env_return" ./nash_dreamer_evaluate.sh
#
# Example (Override seeds):
# GAME_NAME="goofspiel_3" SEEDS="(42,)" ./nash_dreamer_evaluate.sh

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------

#Game name, including the parameters.
# As provided by game.to_compact_str
: ${GAME_NAME:="goofspiel_3"}
# Evaluation metric. Either nash_conv
# expected_util or env_return
: ${METRIC:="nash_conv"}
#Multiplicand of the results
scale_factor=1
if [ "$GAME_NAME" == "leduc" ]; then
  scale_factor=13
fi
scale_factor=${SCALE_FACTOR:-$scale_factor}
#Root folder of the trained networks
: ${BASE_PATH:="trained_networks"}

: ${ALGO_DIRS:="NashDreamer=nash_dreamer_rnad RNaD=rnad"}

: ${SEEDS:="(42, 99, 160, 308, 150, 999, 616, 19, 1000, 513)"}

# Debug Output
echo "------------------------------------------------"
echo "Mode:           Local"
echo "Game:           $GAME_NAME"
echo "Seeds:          $SEEDS"
echo "Metric:         $METRIC"
echo "Scale factor:   $scale_factor"
echo "Base path:      $BASE_PATH"
echo "Algo dirs:      $ALGO_DIRS"
echo "------------------------------------------------"

#The experiment type is always "loaded" with --restore_step -1,
# since that ensures checking the whole folder for the metric.
python -m experiments.actor_critic_evaluate --base_path "$BASE_PATH" --game_name "$GAME_NAME" \
--seeds "$SEEDS" --restore_step "-1" --scale_factor "$scale_factor" loaded --metric "$METRIC" \
--algo_dirs "$ALGO_DIRS"
