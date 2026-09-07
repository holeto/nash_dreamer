#!/usr/bin/env bash

# ------------------------------------------------------------------
# USAGE INSTRUCTIONS
# ------------------------------------------------------------------

# Pass environment variables *before* the script command.
# Assumption: You are already in the project folder and venv is active.
#
# Example (Standard run):
# ./nash_dreamer_train.sh
#
# Example (Override Game and Algo):
# GAME=rps ALGO=reinforce ./nash_dreamer_train.sh
#
# Example (With complex flags - quote the entire string!):
# GAME=goofspiel OPT_FLAGS="--lr 0.01" ./nash_dreamer_train.sh

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
# The latent categorical distribution flags
: ${CAT_FLAGS:=$base_cat_flags}
: ${WM_ADD_FLAGS:=""}
: ${WM_FLAGS:="--free_bits_threshold 1.0 --beta_representation 0.1 --batch_size 64"}
# Additional actor critic flags. These are the flags of the chosen
# train_mode subcommand, so the sensible default differs per algorithm:
# MMD has no --eta, and reinforce/rnad have none of the MMD parameters.
# Setting ALGO_FLAGS explicitly still overrides this.
if [ "$ALGO" == "mmd" ]; then
    algo_default_flags="--num_epochs 4 --clip_epsilon 0.2 --kl_coeff 0.1 --magnet_coeff 0.05"
else
    algo_default_flags="--eta 0.2"
fi
: ${ALGO_FLAGS:=$algo_default_flags}
: ${AC_ADD_FLAGS:=""}
: ${AC_FLAGS:="--beta_imagination 1.0 --beta_real 0.3"}
# Additional optimizer flags
: ${OPT_FLAGS:=""}
# Replay Buffer flags. By default, fully online manner (assuming batch_size 64) and we log returns
: ${REPLAY_FLAGS="--buffer_size 64 --replay_ratio -1 --smoothing_window 64 --log_returns"}


: ${SEEDS:="(42, 99, 160, 308, 150, 999, 616, 19, 1000, 513)"}

exp_flags="--num_steps $steps --save_each $save_each --print_each $print_each $EXPERIMENT_ADD_FLAGS"
wm_flags="$WM_ADD_FLAGS $WM_FLAGS $REPLAY_FLAGS $CAT_FLAGS"
joint_ac_flags="$AC_ADD_FLAGS $AC_FLAGS"

# Debug Output
echo "------------------------------------------------"
echo "Mode:           Local"
echo "Algorithm:      $ALGO"
echo "Game:           $GAME"
echo "Seeds:          $SEEDS"
echo "Game Flags:     $GAME_FLAGS"
echo "Exp Flags:      $exp_flags"
echo "WM Flags:       $wm_flags"
echo "AC Flags:       $joint_ac_flags"
echo "Algorithm Flags: $ALGO_FLAGS"
echo "Opt Flags:      $OPT_FLAGS"
echo "Running         python -m experiments."$GAME"_train $GAME_FLAGS nash_dreamer $exp_flags --seeds $SEEDS \
  $wm_flags $joint_ac_flags $OPT_FLAGS $ALGO $ALGO_FLAGS"
echo "------------------------------------------------"

python -m experiments."$GAME"_train $GAME_FLAGS nash_dreamer $exp_flags --seeds "$SEEDS" \
  $wm_flags $joint_ac_flags $OPT_FLAGS "$ALGO" $ALGO_FLAGS
