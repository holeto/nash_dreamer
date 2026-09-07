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
# ./rnad_train.sh
#
# Example (Override Game):
# GAME=leduc ./rnad_train.sh
#
# Example (With complex flags - quote the entire string!):
# GAME=goofspiel OPT_FLAGS="--lr 0.01" ./rnad_train.sh

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------


#Which type of actor-critic
# to use. Either RNaD or Reinforce
: ${ALGO:="rnad"}

: ${GAME:="goofspiel"}

#Configure the step and
# logging/saving parameters
steps=1000
print_each=100
save_each=100

base_cat_flags="--encoded_classes 1 --encoded_categories 3"

if [ "$GAME" == "leduc" ]; then
    steps=10000
    print_each=1000
    save_each=1000
    base_cat_flags="--encoded_classes 1 --encoded_categories 30"
fi

#If user provided the macros,
# override the default values
steps=${NUM_STEPS:=$steps}
save_each=${SAVE_EACH:=$save_each}
print_each=${PRINT_EACH:=$print_each}

#  Game specific parameters/flags
: ${GAME_FLAGS:=""}
# Additional experiment flags/parameters
# by default deletes the previous directory
# and retrains anew. 
# Additional world model/replay_buffer flags/parameters
# by default logs the smoothed environment
# returns
: ${EXPERIMENT_ADD_FLAGS:="--clean_dir --save_first"}
# Additional actor critic flags
: ${ALGO_FLAGS:=""}
: ${AC_ADD_FLAGS:=""}
: ${AC_FLAGS:="--eta 0.2 --sampling_epsilon 0.2 --rho_vtrace -1"}
# Additional optimizer flags
: ${OPT_FLAGS:=""}
# Replay Buffer flags. By default, fully online manner (assuming batch_size 64) and we log returns
: ${REPLAY_FLAGS="--buffer_size 64 --replay_ratio -1 --smoothing_window 64 --log_returns"}


: ${SEEDS:="(42, 99, 160, 308, 150, 999, 616, 19, 1000, 513)"}

exp_flags="--num_steps $steps --save_each $save_each --print_each $print_each $EXPERIMENT_ADD_FLAGS"
joint_ac_flags="$AC_ADD_FLAGS $AC_FLAGS"

# Debug Output
echo "------------------------------------------------"
echo "Mode:           Local"
echo "Algorithm:      $ALGO"
echo "Game:           $GAME"
echo "Seeds:          $SEEDS"
echo "Game Flags:     $GAME_FLAGS"
echo "Exp Flags:      $exp_flags"
echo "Algorithm Flags: $ALGO_FLAGS"
echo "Replay Flags:   $REPLAY_FLAGS"
echo "Opt Flags:      $OPT_FLAGS"
echo "Running         uv run python -m train."$GAME"_train $GAME_FLAGS rnad $exp_flags --seeds $SEEDS \
  $joint_ac_flags $OPT_FLAGS $REPLAY_FLAGS"
echo "------------------------------------------------"

uv run python -m train."$GAME"_train $GAME_FLAGS rnad $exp_flags --seeds "$SEEDS" \
  $joint_ac_flags $OPT_FLAGS $REPLAY_FLAGS
