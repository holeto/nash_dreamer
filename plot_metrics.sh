#!/usr/bin/env bash

# src/ layout: put it on PYTHONPATH (resolved from cwd, since this must be run from
# the project root -- checkpoint/metric paths are also relative to the caller's cwd)
# so `python -m <pkg>.<module>` resolves.
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"


# Override specific parameters
#GAME_NAME=leduc_poker METRIC=expected_util ./plot_metrics.sh

# Override algos (quote multi-word value)
#ALGOS="NashDreamer" GAME_NAME=goofspiel_4 ./plot_metrics.sh


METRIC_STORE_DIR="${METRIC_STORE_DIR:-metrics/}"
GAME_NAME="${GAME_NAME:-goofspiel_3}"
METRIC="${METRIC:-nash_conv}"
ALGOS="${ALGOS:-NashDreamer RNaD}"

uv run python -m plotting.plot_metrics \
    --metric_store_dir "$METRIC_STORE_DIR" \
    --game_name "$GAME_NAME" \
    --metric "$METRIC" \
    --algos "$ALGOS"
