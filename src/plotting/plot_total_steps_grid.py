"""Merge plot_total_steps_comparison.py's per-game NashConv-vs-total-steps plots
(plots/steps_seen_comparison/<game>/nash_conv_comparison.pdf) into a single figure: one
row of subplots, one per game, with a single shared legend below. Reuses
plot_total_steps_comparison.py's loader, step-transform and color logic so the per-game
panels match the standalone plots; only the figure/legend layout differs.

Default games/algos/labels reproduce the existing goofspiel_5, leduc and phantom_ttt
plots exactly (per-game algo-dir overrides for leduc handled the same way):

    python -m plotting.plot_total_steps_grid

Output: plots/steps_seen_comparison/combined/nash_conv_comparison.pdf
"""

import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np

from plotting.plot_metrics import assign_colors, load_algo_metrics
from plotting.plot_total_steps_comparison import (
    classify_mode,
    recompute_wm_warmup_env_step,
    transform_steps,
    trajectory_length,
)

parser = ArgumentParser()
parser.add_argument("--metric_store_dir", type=str, default="precomputed_metrics/nash_conv/",
                     help="Base path of the directory where extracted metrics are stored. "
                          "The full path per algo is metric_store_dir/algo_name/game_name.")
parser.add_argument("--games", type=str, default="goofspiel_5 leduc phantom_ttt",
                     help="Space-separated game names, one subplot per entry, left to right.")
parser.add_argument("--titles", type=str, default="Goofspiel-5,Leduc,Phantom TTT",
                     help="Comma-separated subplot titles, one per --games entry.")
parser.add_argument("--algos", type=str, default="nash_dreamer_rnad rnad mmd nash_dreamer_mmd replayed_rnad",
                     help="Which algorithms (directory names) to plot, shared across all games "
                          "(subject to --leduc overrides below).")
parser.add_argument("--labels", type=str, default="NashDreamerRNaD,RNaD,MMD,NashDreamerMMD,RNaD with replay",
                     help="Comma-separated legend labels, one per --algos entry, overriding "
                          "the algo_name recorded in the metrics file.")
parser.add_argument("--replayed_algo_dirs", type=str, default="replayed_rnad",
                     help="Space-separated --algos entries to treat as replayed_full "
                          "(no warm-up, fixed traj_len+1 multiplier applied throughout).")
parser.add_argument("--log_x", action="store_true", default=True, help="Plot x-axes in log scale (on by default).")
parser.add_argument("--no_log_x", dest="log_x", action="store_false")
parser.add_argument("--no_log_y", action="store_true", help="Disable y-axis log scale (on by default).")
parser.add_argument("--no_warmup_line", action="store_true", default=True,
                     help="Don't plot the WM warm-up vertical line (on by default -- matches "
                          "the existing per-game plots, none of which show it).")
parser.add_argument("--warmup_line", dest="no_warmup_line", action="store_false")
parser.add_argument("--skip_first_checkpoint", action="store_true", default=True,
                     help="Drop each seed's first (step=0) checkpoint before plotting -- on by "
                          "default, since on a log x-axis it would otherwise collapse to the "
                          "+1 offset and bunch every algo's first point at x=1.")
parser.add_argument("--keep_first_checkpoint", dest="skip_first_checkpoint", action="store_false")
parser.add_argument("--save_dir", type=str, default="plots/steps_seen_comparison/combined",
                     help="Directory to save the merged figure into.")

# leduc has no nash_dreamer_{rnad,mmd}/leduc data -- both NashDreamer runs only exist
# under the "_factored" ablation dirs. Substituted in per-game so --algos/--labels can
# otherwise stay identical across all three games.
GAME_ALGO_OVERRIDES = {
    "leduc": {"nash_dreamer_rnad": "nash_dreamer_rnad_factored", "nash_dreamer_mmd": "nash_dreamer_mmd_factored"},
}


def plot_one_game(ax, metric_store_dir, game_name, algo_names, labels, replayed_algo_dirs,
                   log_x, log_y, skip_first_checkpoint, no_warmup_line):
    """Loads, transforms and draws one game's panel onto ax. Mirrors
    plot_total_steps_comparison.main()'s per-algo loop, minus legend/save (handled once
    for the whole figure by the caller)."""
    traj_len = trajectory_length(game_name)
    results = {}
    algo_strs = {}
    uniform_nash_conv = None
    algo_warmup_steps = {}

    for i, algo_name in enumerate(algo_names):
        seed_data, _, algo_str, _, new_uniform_nash_conv, _ = load_algo_metrics(
            metric_store_dir, algo_name, game_name, "nash_conv"
        )
        algo_strs[algo_name] = labels[i] if labels else algo_str
        if seed_data is None:
            continue

        mode = classify_mode(algo_name, replayed_algo_dirs)
        wm_warmup_env_step = -1
        if mode == "warmup_split":
            wm_warmup_env_step = recompute_wm_warmup_env_step(metric_store_dir, algo_name, game_name, traj_len)

        seed_data = {
            s: (transform_steps(steps, mode, wm_warmup_env_step, traj_len), metrics)
            for s, (steps, metrics) in seed_data.items()
        }
        if skip_first_checkpoint:
            seed_data = {s: (steps[1:], metrics[1:]) for s, (steps, metrics) in seed_data.items()}

        results[algo_name] = seed_data
        if wm_warmup_env_step > 0:
            algo_warmup_steps[algo_name] = wm_warmup_env_step
        if new_uniform_nash_conv is not None:
            uniform_nash_conv = new_uniform_nash_conv

    colors = assign_colors(algo_names, algo_strs)

    for algo_name, seed_data in results.items():
        if not seed_data:
            continue
        algo_str = algo_strs[algo_name]
        first_seed = list(seed_data.keys())[0]
        ref_steps = seed_data[first_seed][0]

        stacked_metrics = [metrics for s, (steps, metrics) in seed_data.items() if np.array_equal(steps, ref_steps)]
        if not stacked_metrics:
            continue
        matrix = np.vstack(stacked_metrics)
        mean = np.mean(matrix, axis=0)

        color = colors.get(algo_str, 'black')
        for i in range(matrix.shape[0]):
            ax.plot(ref_steps, matrix[i], color=color, alpha=0.3, linestyle='--', linewidth=1)
        ax.plot(ref_steps, mean, label=algo_str, color=color, linestyle='-', linewidth=2.5,
                 marker='o', markersize=4)

    if uniform_nash_conv is not None:
        ax.axhline(y=uniform_nash_conv, color='orange', linestyle='--', label="Uniform Policy", alpha=0.7)

    if algo_warmup_steps and not no_warmup_line:
        warmup_step = next(iter(algo_warmup_steps.values()))
        ax.axvline(x=warmup_step, color='black', linestyle=':', linewidth=1.5, alpha=0.7, label="WM warm-up end")

    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(min(ymin, 0.1), max(ymax, 1.0))
    ax.tick_params(axis='both', labelsize=13)
    ax.grid(True, alpha=0.3)


def main():
    args = parser.parse_args()
    games = args.games.split()
    titles = args.titles.split(",")
    algo_names = args.algos.split()
    labels = args.labels.split(",") if args.labels else None
    replayed_algo_dirs = set(args.replayed_algo_dirs.split())
    assert len(titles) == len(games), f"--titles must have one entry per --games entry ({len(games)}), got {len(titles)}"

    fig, axes = plt.subplots(1, len(games), figsize=(6 * len(games), 5), sharey=False)
    if len(games) == 1:
        axes = [axes]

    for ax, game_name, title in zip(axes, games, titles):
        overrides = GAME_ALGO_OVERRIDES.get(game_name, {})
        game_algo_names = [overrides.get(a, a) for a in algo_names]
        plot_one_game(ax, args.metric_store_dir, game_name, game_algo_names, labels, replayed_algo_dirs,
                      args.log_x, not args.no_log_y, args.skip_first_checkpoint, args.no_warmup_line)
        ax.set_title(title, fontsize=18)
        ax.set_xlabel("Total steps seen by actor-critic", fontsize=16)

    axes[0].set_ylabel("NashConv", fontsize=18)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    ncol = min(len(legend_labels), 4)
    legend_rows = -(-len(legend_labels) // ncol)  # ceil division
    bottom_margin = 0.1 + 0.06 * legend_rows
    fig.tight_layout(rect=(0, bottom_margin, 1, 1))
    fig.legend(handles, legend_labels, fontsize=15, loc='lower center',
               bbox_to_anchor=(0.5, 0), ncol=ncol)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    filename = "nash_conv_comparison.pdf"
    plt.savefig(os.path.join(args.save_dir, filename), bbox_inches='tight')
    print(f"Plot saved to {os.path.join(args.save_dir, filename)}")


if __name__ == "__main__":
    main()
