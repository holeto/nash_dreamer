#!/usr/bin/env bash


# Override specific parameters
#GAME_NAME=leduc_poker METRIC=expected_util ./plot_metrics.sh

# Override algos (quote multi-word value)
#ALGOS="NashDreamer" GAME_NAME=goofspiel_4 ./plot_metrics.sh


METRIC_STORE_DIR="${METRIC_STORE_DIR:-metrics/}"
GAME_NAME="${GAME_NAME:-goofspiel_3}"
METRIC="${METRIC:-nash_conv}"
ALGOS="${ALGOS:-NashDreamer RNaD}"

python -m experiments.plot_metrics \
    --metric_store_dir "$METRIC_STORE_DIR" \
    --game_name "$GAME_NAME" \
    --metric "$METRIC" \
    --algos "$ALGOS"
