# Plotting

Plotting reference for `src/plotting/`. See the root [README.md](../../README.md) for install
instructions.

```bash
export PYTHONPATH="$(pwd)/src"
```

## Training curves (`plot_metrics.py`)

Plots NashConv / return curves written by [eval/actor_critic_evaluate.py](../eval/actor_critic_evaluate.py)
(a per-seed faint dashed line plus a bold per-algorithm mean, consistent colors across algorithms).

```bash
./plot_metrics.sh

# Override game/metric/algorithms
GAME_NAME=leduc_poker METRIC=expected_util ./plot_metrics.sh
ALGOS="NashDreamer" GAME_NAME=goofspiel_4 ./plot_metrics.sh

# Or call directly:
uv run python -m plotting.plot_metrics \
    --metric_store_dir metrics/ \
    --game_name goofspiel_3 \
    --metric nash_conv \
    --algos "NashDreamer RNaD"
```

## World-model quality plots

Three scripts plot the pickled/JSON output of the matching check under
[world_model_experiments/](../world_model_experiments/) (see that page for what each check itself
measures), comparing multiple models with the same mean-bold/per-seed-faint-dashed style as
`plot_metrics.py`. All three take `--model_paths` as comma-separated `label=path` pairs, where each
path contains `seed_{id}` subdirectories with the corresponding check's output file, and
`--output_dir` (default `world_model_plots`):

| Script | Reads | Plots |
|---|---|---|
| `chance_distribution_plot.py` | `chance_marginal_eval.pkl` | Reconstruction L1 error and unmatched-probability magnitude, chance vs. deterministic nodes, four PDFs under `{spec}/chance_distribution/` |
| `posterior_collapse_plot.py` | `posterior_collapse_eval.pkl` | Expected KL(posterior‖prior), clustered by chance-node outcome count `N`, one PDF per `N` under `{spec}/posterior_collapse/`, with a `log(N)` uniform-entropy baseline |
| `rollout_validity_plot.py` | `rollout_validity.json` | Per-category (obs/reward/terminal/legal) rollout error, four PDFs under `{spec}/rollout_validity/` |

```bash
uv run python -m plotting.chance_distribution_plot \
  --model_paths "no_representation=leduc_no_rep, full_loss=leduc" \
  --output_dir world_model_plots
```

`plot_utils.py` is the shared helper module behind all four scripts here (`plot_metrics.py` included)
— path/seed/step discovery and the common plotting style.
