from argparse import ArgumentParser
import os
from matplotlib import ticker
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.pyplot as plt


parser = ArgumentParser()
parser.add_argument("--metric_store_dir", type=str, default="metrics/", help="Base path of the directory where extracted metrics are stored. The full path per algo is metric_store_dir/algo_name/game_name.")
parser.add_argument("--game_name", type=str, default="goofspiel_3", help="Name and parameter string of the game to plot for.")
parser.add_argument("--metric", type=str, default="nash_conv", choices=("nash_conv", "expected_util", "env_return"), help="Type of metric to plot.")
parser.add_argument("--algos", type=str, default="NashDreamer RNaD", help="Which algorithms to plot.")
parser.add_argument("--log_x", action="store_true", help="Plot x-axis in log scale.")
parser.add_argument("--log_y", action="store_true", help="Plot y-axis in log scale.")
parser.add_argument("--divide_by", type=float, default=1.0, help="Divide all metric values (and the uniform-policy baseline) by this constant before plotting.")
parser.add_argument("--skip_divide", type=str, default="", help="Space-separated subset of --algos entries to exclude from --divide_by (e.g. baselines that are already normalized).")
parser.add_argument("--no_warmup_line", action="store_true", help="Don't plot the WM warm-up vertical line (useful when the warm-up length itself is the thing being ablated).")
parser.add_argument("--save_dir", type=str, default=None, help="Directory to save the plot into. Defaults to plots/comparison/<game_name>; override to avoid colliding with an existing plot for the same game.")
parser.add_argument("--labels", type=str, default=None, help="Optional comma-separated list of legend labels, one per entry in --algos (in order), overriding the algo_name recorded in the metrics file.")
parser.add_argument("--clamp_checkpoints", type=str, default="", help="Space-separated subset of --algos entries whose steps/values are truncated to the first --clamp_to checkpoints (e.g. a baseline logged at a finer checkpoint cadence than the rest).")
parser.add_argument("--clamp_to", type=int, default=None, help="Number of checkpoints to truncate --clamp_checkpoints entries to.")
parser.add_argument("--skip_first_checkpoint", action="store_true", help="Drop each seed's first (step=0, untrained-network) checkpoint before plotting.")


PRESET_COLORS = {
    'NashDreamer': 'tab:blue',
    'NashDreamerRNaD': 'tab:blue',
    'MMD' : 'tab:purple',
    'NashDreamerMMD' : 'tab:cyan',
    'NashDreamer without enc loss': 'tab:brown',
    'NashDreamerREINFORCE': 'tab:orange',
    'RNaD': 'tab:red',
    'RNaD with replay': 'tab:green',
}


def assign_colors(algo_names, algo_strs):
    """Maps each label in algo_strs (in algo_names order) to a plot color: the
    PRESET_COLORS entry if the label has one, else the next free color from
    default_cycle+tab20-tints that isn't reserved by PRESET_COLORS. default_cycle
    alone (tab10, 10 colors) is mostly claimed by PRESET_COLORS, so it's extended with
    tab20's lighter tints (odd indices; the even ones just duplicate tab10) to leave
    enough distinct, non-reserved colors for algos with no preset entry -- e.g. a
    multi-value ablation (wm0/250/500/2000/factored) has more such algos than tab10 has
    unreserved colors. Comparison is done via to_hex() since PRESET_COLORS uses named
    strings ('tab:blue') while default_cycle yields hex strings ('#1f77b4') -- a plain
    string comparison between the two never matches, so reserved colors would silently
    leak into the "unreserved" pool without this normalization.
    """
    default_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    extra_cycle = list(plt.get_cmap('tab20').colors[1::2])
    reserved_hex = {mcolors.to_hex(c) for c in PRESET_COLORS.values()}
    available_colors = [c for c in default_cycle + extra_cycle if mcolors.to_hex(c) not in reserved_hex]

    colors = {}
    next_available = 0
    for name in algo_names:
        algo_str = algo_strs.get(name)
        if not algo_str or algo_str in colors:
            continue
        if algo_str in PRESET_COLORS:
            colors[algo_str] = PRESET_COLORS[algo_str]
        else:
            colors[algo_str] = available_colors[next_available % len(available_colors)]
            next_available += 1
    return colors


def load_algo_metrics(metric_store_dir, algo_name, game_name, metric):
    """Load metrics from a text file for one algorithm.
    Returns: (seed_data, game_str, smoothing_window, uniform_nash_conv)
      seed_data: {seed: (steps_array, metrics_array)}
    """
    filepath = os.path.join(metric_store_dir, algo_name, game_name, f"{metric}.txt")
    if not os.path.exists(filepath):
        print(f"Metrics file not found: {filepath}")
        return None, None, None, None, None, None

    seed_data = {}
    game_str = ""
    algo_str = ""
    smoothing_window = -1
    uniform_nash_conv = None
    wm_warmup_env_step = -1

    current_seed = None
    current_steps = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, _, value = line.partition(': ')
            if key == 'algo_name':
                algo_str = value
            if key == 'game_str':
                game_str = value
            elif key == 'smoothing_window':
                smoothing_window = int(value)
            elif key == 'uniform_nash_conv':
                uniform_nash_conv = float(value)
            elif key == 'wm_warmup_env_step':
                wm_warmup_env_step = float(value)
            elif key == 'seed':
                current_seed = int(value)
                current_steps = None
            elif key == 'steps':
                current_steps = np.array([float(v) for v in value.split()])
            elif key == 'values':
                assert current_seed is not None and current_steps is not None
                values = np.array([float(v) for v in value.split()])
                seed_data[current_seed] = (current_steps, values)
    return seed_data, game_str, algo_str, smoothing_window, uniform_nash_conv, wm_warmup_env_step


def plot_comparison(args):
    """
    Plot NashDreamer vs RNaD metrics loaded from stored text files.
    """
    game_path = args.game_name
    algo_names = args.algos.split()
    algo_strs = {}
    labels = args.labels.split(",") if args.labels else None
    skip_divide = set(args.skip_divide.split())
    clamp_checkpoints = set(args.clamp_checkpoints.split())

    results = {}
    game_str = ""
    smoothing_window = -1
    uniform_nash_conv = None
    uniform_nash_conv_divisor = 1.0
    algo_warmup_steps = {}
    max_steps = 0

    # 1. Load Data
    for i, algo_name in enumerate(algo_names):
        seed_data, new_game_str, algo_str, new_smoothing_window, new_uniform_nash_conv, wm_warmup = load_algo_metrics(
            args.metric_store_dir, algo_name, args.game_name, args.metric
        )
        algo_strs[algo_name] = labels[i] if labels else algo_str
        if seed_data is None:
            continue
        if algo_name in clamp_checkpoints:
            seed_data = {s: (steps[:args.clamp_to], metrics[:args.clamp_to]) for s, (steps, metrics) in seed_data.items()}
        results[algo_name] = seed_data
        if wm_warmup > 0:
            algo_warmup_steps[algo_name] = wm_warmup
        for s, (steps, metrics) in seed_data.items():
            max_steps = max(max_steps, len(steps))
        #Check if all experiments used the same game
        if not game_str:
            game_str = new_game_str
        else:
            assert new_game_str == game_str, f"Expected all models to use the same game {game_str}. Found {new_game_str} for algorithm {algo_name} instead!"
        #Check if all experiments
        # used the same smoothing window
        # (relevant for the environment returns only)
        if smoothing_window < 0:
            smoothing_window = new_smoothing_window
        else:
            assert smoothing_window == new_smoothing_window, f"Expected all models to use the same smoothing window {smoothing_window}. Found {new_smoothing_window} for algorithm {algo_name} instead!"
        if new_uniform_nash_conv is not None:
            uniform_nash_conv = new_uniform_nash_conv
            uniform_nash_conv_divisor = 1.0 if algo_name in skip_divide else args.divide_by

    if not results:
        print("No data found for either algorithm.")
        return

    render_plot(results, algo_strs, algo_names, game_str, smoothing_window,
                uniform_nash_conv, uniform_nash_conv_divisor, algo_warmup_steps,
                max_steps, args, skip_first_checkpoint=args.skip_first_checkpoint)


def render_plot(results, algo_strs, algo_names, game_str, smoothing_window,
                uniform_nash_conv, uniform_nash_conv_divisor, algo_warmup_steps,
                max_steps, args, x_label="Environment steps", markers=False,
                skip_first_checkpoint=False):
    """Renders and saves the comparison plot from already-loaded (and possibly
    x-axis-transformed) data. Split out of plot_comparison so other scripts -- e.g. one
    plotting against a different x-axis quantity -- can reuse the exact same styling/
    save logic after loading and transforming data their own way. x_label defaults to
    plot_metrics.py's own original hardcoded label, so its call site needs no change.
    markers=False reproduces plot_metrics.py's original line-only look exactly; callers
    whose x-axis is a nonlinear transform of the evaluated steps (so evenly-spaced-looking
    segments are not evenly spaced in reality) can pass markers=True to mark each actually-
    evaluated point explicitly, distinguishing real data from the straight-line interpolation
    matplotlib draws in between. skip_first_checkpoint=True drops each seed's first (step=0,
    untrained-network) checkpoint before plotting -- useful on log_x, where it would
    otherwise need the +1 offset below anyway, and on linear axes where an untrained
    uniform-policy value can dwarf the rest of the curve's range."""
    game_path = args.game_name
    skip_divide = set(args.skip_divide.split())

    # 2. Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    x_name = "Gradient steps"

    if args.metric == 'nash_conv':
        metric_str = "NashConv"
        plot_str = "nash_conv"
    elif args.metric == 'expected_util':
        metric_str = "Expected Utility"
        plot_str = "expected_utility"
    else:
        metric_str = f"Env returns smoothed with a {smoothing_window} window"
        plot_str = f"env_return_window_{smoothing_window}"

    # Plot Algorithm Curves
    colors = assign_colors(algo_names, algo_strs)

    for algo_name, seed_data in results.items():
        if not seed_data:
            continue

        algo_str = algo_strs[algo_name]
        # seed_data is {seed: (steps, metrics)}
        # We need to aggregate them.

        if skip_first_checkpoint:
            seed_data = {s: (steps[1:], metrics[1:]) for s, (steps, metrics) in seed_data.items()}

        # Assumption: All seeds have the same steps.
        first_seed = list(seed_data.keys())[0]
        ref_steps = seed_data[first_seed][0]

        # Create list of metric arrays
        stacked_metrics = []
        for s, (steps, metrics) in seed_data.items():
            if np.array_equal(steps, ref_steps):
                stacked_metrics.append(metrics)
            else:
                print(f"Warning: Step mismatch for {algo_name} seed {s}. Skipping aggregation for this seed.")

        if not stacked_metrics:
            continue

        # Convert to matrix: (N_seeds, N_steps)
        divisor = 1.0 if algo_name in skip_divide else args.divide_by
        matrix = np.vstack(stacked_metrics) / divisor

        # Calculate Statistics
        mean = np.mean(matrix, axis=0)

        if args.log_x:
            ref_steps += 1

        # Plot Mean Line
        color = colors.get(algo_str, 'black')
        for i in range(matrix.shape[0]):
            ax.plot(ref_steps, matrix[i],
                    color=color,
                    alpha=0.3,       # Make it faint
                    linestyle='--',   # Dotted/Dashed line
                    linewidth=1)      # Thinner line

        # 2. Plot Mean Line
        # We plot this LAST so it appears on top of the individual seeds.
        # We add the label here so it appears in the legend once.
        ax.plot(ref_steps, mean,
                #label="Smoothed returns with rolling average over 32 trajectories",
                #color='blue',
                label=algo_str,
                color=color,
                linestyle='-',
                linewidth=2.5,    # Thicker, solid line
                marker='o' if markers else None,
                markersize=4)

    # Plot Uniform Baseline (Dashed Line)
    if args.metric == "nash_conv" and uniform_nash_conv is not None:
        ax.axhline(y=uniform_nash_conv / uniform_nash_conv_divisor, xmin=0, xmax=max_steps, color='orange', linestyle='--', label="Uniform Policy", alpha=0.7)
        #ax.set_yscale('log')

    # Plot world model warm-up boundary as a single vertical line
    if algo_warmup_steps and not args.no_warmup_line:
        warmup_step = next(iter(algo_warmup_steps.values()))
        ax.axvline(x=warmup_step, color='black', linestyle=':', linewidth=1.5, alpha=0.7,
                   label="WM warm-up end")

    # Styling
    # A long legend (e.g. a multi-value ablation) in its default 'upper right' spot
    # obscures the plot; any in-axes corner still risks overlapping curves depending on
    # where the data sits, so once it grows past a handful of entries, move it fully
    # outside the axes to the right instead. This needs bbox_inches='tight' at savefig
    # time below, or the legend gets clipped off the saved file.
    handles, labels = ax.get_legend_handles_labels()
    if len(labels) > 5:
        ax.legend(handles, labels, fontsize=15, loc='center left', bbox_to_anchor=(1.02, 0.5))
    else:
        ax.legend(handles, labels, fontsize=15, loc='upper right')
    ax.set_xlabel(x_label, fontsize=20)
    if args.log_x:
        ax.set_xscale('log')
    if args.log_y:
        ax.set_yscale('log')
        if args.metric == 'nash_conv':
            # NashConv is in [0, uniform_nash_conv], often well under 1 -- extend the view
            # to always include the 10^0/10^-1 decades, so the default LogLocator places
            # (and actually renders) ticks there instead of leaving it to the data's own
            # range, which for well-trained runs can sit entirely below 0.1 and show only a
            # single tick. Deliberately NOT calling ax.set_yticks() to force this: LogLocator
            # always returns tick candidates padded a decade beyond [ymin, ymax], and handing
            # that padded list to set_yticks() pulls the padding decades into the view too
            # (set_yticks expands ylim to fit whatever it's given), which is what previously
            # also dragged in unwanted 10^1/10^-2/10^-3 clutter. Widening ylim alone is
            # sufficient -- the existing LogLocator already places a tick at every decade
            # inside the (now-widened) view and nothing outside it gets rendered.
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(min(ymin, 0.1), max(ymax, 1.0))
    # ymin, ymax = ax.get_ylim()
    # ax.set_ylim(bottom=min(ymin, 1e-1), top=max(ymax, 1.0))
    # ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))
    # ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
    ax.set_ylabel(metric_str, fontsize=20)
    #ax.set_ylabel("Episode return", fontsize=15)
    ax.tick_params(axis='both', labelsize=15)
    ax.xaxis.get_offset_text().set_fontsize(15)
    ax.grid(True, alpha=0.3)

    # if args.metric == "nash_conv":
    #     ax.set_yscale("log")

    # Save
    save_dir = args.save_dir if args.save_dir else f"plots/comparison/{game_path}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filename = f"{plot_str}_comparison.pdf"
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, filename.replace(".pdf", ".pdf")), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(save_dir, filename)}")


def main():
    args = parser.parse_args()
    plot_comparison(args)


if __name__ == "__main__":
    main()
