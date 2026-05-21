#!/usr/bin/env bash

# ------------------------------------------------------------------
# USAGE INSTRUCTIONS
# ------------------------------------------------------------------

# Pass environment variables *before* the script command.
# Assumption: You are already in the project folder and venv is active.
#
# Example (Standard run):
# ./head_to_head.sh
#
# Example (Override game and restore step):
# GAME_NAME="goofspiel_3" RESTORE_STEP=5000 ./head_to_head.sh
#
# Example (Custom algo dirs):
# ALGO_DIRS="NashDreamer=nash_dreamer_rnad RNaD=rnad" ./head_to_head.sh

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------

# Game name, including the parameters.
# As provided by game.to_compact_str
: ${GAME_NAME:="goofspiel_4"}

# Root folder of the trained networks
: ${BASE_PATH:="trained_networks"}

# Algorithm name=directory pairs (space-separated).
# First algorithm plays as player 1, second as player 2.
: ${ALGO_DIRS:="NashDreamer=nash_dreamer_rnad RNaD=rnad"}

# Seeds of the stored models to evaluate
: ${SEEDS:="(42, 99, 160, 308, 150, 999, 616, 19, 1000, 513)"}

# Saved step of the model to restore (per algorithm; RESTORE_STEP sets both if the individual vars are unset)
: ${RESTORE_STEP:=10000}
: ${RESTORE_STEP_A:=$RESTORE_STEP}
: ${RESTORE_STEP_B:=$RESTORE_STEP}

# Number of games to play per configuration
: ${NUM_GAMES:=1024}

# Uniform policy mix per player. Set to 1.0 for a fully random player.
: ${EPSILON_A:=0.0}
: ${EPSILON_B:=0.0}

# RNG seed for game sampling
: ${RNG_SEED:=42}

# Number of games per vmap chunk (to avoid OOM)
: ${CHUNK_SIZE:=1024}

# Directory to store metrics
: ${METRIC_STORE_DIR:="metrics/"}

# Debug Output
echo "------------------------------------------------"
echo "Mode:             Local"
echo "Game:             $GAME_NAME"
echo "Seeds:            $SEEDS"
echo "Restore step A:   $RESTORE_STEP_A"
echo "Restore step B:   $RESTORE_STEP_B"
echo "Algo dirs:        $ALGO_DIRS"
echo "Num games:        $NUM_GAMES"
echo "Epsilon A:        $EPSILON_A"
echo "Epsilon B:        $EPSILON_B"
echo "RNG seed:         $RNG_SEED"
echo "Chunk size:       $CHUNK_SIZE"
echo "Base path:        $BASE_PATH"
echo "Metric store dir: $METRIC_STORE_DIR"
echo "------------------------------------------------"

python -m experiments.head_to_head_evaluate \
  --base_path "$BASE_PATH" \
  --game_name "$GAME_NAME" \
  --seeds "$SEEDS" \
  --restore_step_a "$RESTORE_STEP_A" \
  --restore_step_b "$RESTORE_STEP_B" \
  --algo_dirs "$ALGO_DIRS" \
  --num_games "$NUM_GAMES" \
  --epsilon_a "$EPSILON_A" \
  --epsilon_b "$EPSILON_B" \
  --rng_seed "$RNG_SEED" \
  --chunk_size "$CHUNK_SIZE" \
  --metric_store_dir "$METRIC_STORE_DIR"
