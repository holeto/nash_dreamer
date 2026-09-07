#!/usr/bin/env bash

# ------------------------------------------------------------------
# USAGE INSTRUCTIONS
# ------------------------------------------------------------------

# Pass environment variables *before* the script command.
# Assumption: You are already in the project folder and venv is active.
#
# Example (Standard run):
# ./mmd_train.sh
#
# Example (Override Game):
# GAME=leduc ./mmd_train.sh
#
# Example (With complex flags - quote the entire string!):
# GAME=goofspiel OPT_FLAGS="--lr 0.01" ./mmd_train.sh
#
# Note on the magnet strength. --magnet_coeff is THE parameter to sweep per game,
# it sets the temperature of the quantal response equilibrium MMD converges to.
# A game with a pure equilibrium wants a small magnet, since the uniform magnet
# pulls away from every pure strategy (goofspiel_3 reaches nash_conv 0.0000
# anywhere in [0, 0.2], but 0.9534 at 1.0). A game with a mixed equilibrium wants
# a large one, since the regularization is what stops the policy gradient from
# cycling away from it (RPS reaches 0.105 at 1.0, but 1.784 at 0.2). It lives on
# the scale of the normalized advantage, NOT on the scale of the much smaller
# RNaD --eta. Override it per game:
# GAME=rps AC_FLAGS="--magnet_coeff 1.0" ./mmd_train.sh

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------


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
: ${EXPERIMENT_ADD_FLAGS:="--clean_dir --save_first"}
# Additional actor critic flags
: ${AC_ADD_FLAGS:=""}
: ${AC_FLAGS:="--num_epochs 4 --clip_epsilon 0.2 --kl_coeff 0.1 --magnet_coeff 0.05 --sampling_epsilon 0.0"}
# Additional optimizer flags
: ${OPT_FLAGS:=""}
# Replay Buffer flags. MMD is strictly on-policy, so replay_ratio must stay -1
# (fully online). Anything else mixes off-policy trajectories into the batch,
# which MMD carries no importance sampling correction for.
: ${REPLAY_FLAGS="--buffer_size 64 --replay_ratio -1 --smoothing_window 64 --log_returns"}


: ${SEEDS:="(42, 99, 160, 308, 150, 999, 616, 19, 1000, 513)"}

exp_flags="--num_steps $steps --save_each $save_each --print_each $print_each $EXPERIMENT_ADD_FLAGS"
joint_ac_flags="$AC_ADD_FLAGS $AC_FLAGS"

# Debug Output
echo "------------------------------------------------"
echo "Mode:           Local"
echo "Algorithm:      mmd"
echo "Game:           $GAME"
echo "Seeds:          $SEEDS"
echo "Game Flags:     $GAME_FLAGS"
echo "Exp Flags:      $exp_flags"
echo "AC Flags:       $joint_ac_flags"
echo "Replay Flags:   $REPLAY_FLAGS"
echo "Opt Flags:      $OPT_FLAGS"
echo "Running         python -m experiments."$GAME"_train $GAME_FLAGS mmd $exp_flags --seeds $SEEDS \
  $joint_ac_flags $OPT_FLAGS $REPLAY_FLAGS"
echo "------------------------------------------------"

python -m experiments."$GAME"_train $GAME_FLAGS mmd $exp_flags --seeds "$SEEDS" \
  $joint_ac_flags $OPT_FLAGS $REPLAY_FLAGS
