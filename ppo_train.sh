#!/usr/bin/env bash

# src/ layout: put it on PYTHONPATH (resolved from cwd, since this must be run from
# the project root -- checkpoint/metric paths are also relative to the caller's cwd)
# so `python -m <pkg>.<module>` resolves.
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"

# ------------------------------------------------------------------
# USAGE INSTRUCTIONS
# ------------------------------------------------------------------

# Trains a single agent PPO approximate best response against a FROZEN opponent
# checkpoint. The mean return it reaches is an approximate exploitability of that
# opponent, usable on games too large for the exact tree walk in
# policy_eval_utils.model_best_response.
#
# OPPONENT is required. Example:
#
# OPPONENT=trained_networks/mmd/goofspiel_3/seed_42/step_1000.pkl ./ppo_train.sh
#
# For a full exploitability estimate, train one best response per player and add
# the two values:
#
# OPPONENT=... PLAYER_ID=0 ./ppo_train.sh
# OPPONENT=... PLAYER_ID=1 ./ppo_train.sh
#
# then play each against the opponent with head_to_head.sh.
#
# Note on --entropy_coeff: it defaults to 0 on purpose. Entropy regularization
# softens the best response and therefore UNDER-estimates exploitability.

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------

: ${GAME:="goofspiel"}
: ${PLAYER_ID:=0}

if [ -z "$OPPONENT" ]; then
    echo "Error: OPPONENT is required. Set it to a trained checkpoint, e.g."
    echo "  OPPONENT=trained_networks/mmd/goofspiel_3/seed_42/step_1000.pkl ./ppo_train.sh"
    exit 1
fi

#Configure the step and
# logging/saving parameters
steps=1000
print_each=100
save_each=100

if [ "$GAME" == "leduc" ]; then
    steps=10000
    print_each=1000
    save_each=1000
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
: ${AC_FLAGS:="--num_epochs 4 --clip_epsilon 0.2 --entropy_coeff 0.0 --sampling_epsilon 0.0"}
# Additional optimizer flags
: ${OPT_FLAGS:=""}
# PPO is strictly on-policy and has NO replay buffer, so --buffer_size, --replay_ratio
# and --log_returns are accepted but ignored. --smoothing_window and
# --return_log_frequency do apply: they control the best-response value estimate.
: ${REPLAY_FLAGS="--smoothing_window 64 --return_log_frequency 10"}

: ${SEEDS:="(42, )"}

exp_flags="--num_steps $steps --save_each $save_each --print_each $print_each $EXPERIMENT_ADD_FLAGS"
joint_ac_flags="$AC_ADD_FLAGS $AC_FLAGS"

# Debug Output
echo "------------------------------------------------"
echo "Mode:           Local"
echo "Algorithm:      ppo (approximate best response)"
echo "Game:           $GAME"
echo "Opponent:       $OPPONENT"
echo "Player id:      $PLAYER_ID"
echo "Seeds:          $SEEDS"
echo "Game Flags:     $GAME_FLAGS"
echo "Exp Flags:      $exp_flags"
echo "AC Flags:       $joint_ac_flags"
echo "Opt Flags:      $OPT_FLAGS"
echo "Running         uv run python -m train."$GAME"_train $GAME_FLAGS ppo $exp_flags --seeds $SEEDS \
  --opponent_path $OPPONENT --player_id $PLAYER_ID $joint_ac_flags $OPT_FLAGS $REPLAY_FLAGS"
echo "------------------------------------------------"

uv run python -m train."$GAME"_train $GAME_FLAGS ppo $exp_flags --seeds "$SEEDS" \
  --opponent_path "$OPPONENT" --player_id "$PLAYER_ID" $joint_ac_flags $OPT_FLAGS $REPLAY_FLAGS
