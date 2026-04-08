from argparse import ArgumentParser
import os
import numpy as np
import matplotlib.pyplot as plt


parser = ArgumentParser()
parser.add_argument("--metric_store_dir", type=str, default="metrics/", help="Base path of the directory where extracted metrics are stored. The full path per algo is metric_store_dir/algo_name/game_name.")
parser.add_argument("--game_name", type=str, default="goofspiel_3", help="Name and parameter string of the game to plot for.")
parser.add_argument("--metric", type=str, default="nash_conv", choices=("nash_conv", "expected_util", "env_return"), help="Type of metric to plot.")
parser.add_argument("--algos", type=str, default="NashDreamer RNaD", help="Which algorithms to plot.")


def load_algo_metrics(metric_store_dir, algo_name, game_name, metric):
    """Load metrics from a text file for one algorithm.
    Returns: (seed_data, game_str, smoothing_window, uniform_nash_conv)
      seed_data: {seed: (steps_array, metrics_array)}
    """
    filepath = os.path.join(metric_store_dir, algo_name, game_name, f"{metric}.txt")
    if not os.path.exists(filepath):
        print(f"Metrics file not found: {filepath}")
        return None, None, None, None

    seed_data = {}
    game_str = ""
    algo_str = ""
    smoothing_window = -1
    uniform_nash_conv = None

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
            elif key == 'seed':
                current_seed = int(value)
                current_steps = None
            elif key == 'steps':
                current_steps = np.array([float(v) for v in value.split()])
            elif key == 'values':
                assert current_seed is not None and current_steps is not None
                values = np.array([float(v) for v in value.split()])
                seed_data[current_seed] = (current_steps, values)

    return seed_data, game_str, algo_str, smoothing_window, uniform_nash_conv


def plot_comparison(args):
    """
    Plot NashDreamer vs RNaD metrics loaded from stored text files.
    """
    game_path = args.game_name
    algo_names = args.algos.split()
    algo_strs = {}

    results = {}
    game_str = ""
    smoothing_window = -1
    uniform_nash_conv = None
    max_steps = 0

    # 1. Load Data
    for algo_name in algo_names:
        seed_data, new_game_str, algo_str, new_smoothing_window, new_uniform_nash_conv = load_algo_metrics(
            args.metric_store_dir, algo_name, args.game_name, args.metric
        )
        algo_strs[algo_name] = algo_str
        if seed_data is None:
            continue
        results[algo_name] = seed_data
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

    if not results:
        print("No data found for either algorithm.")
        return

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
    colors = {'NashDreamer': 'tab:red', 'RNaD': 'tab:blue', 'REINFORCE': 'tab:green'}

    for algo_name, seed_data in results.items():
        if not seed_data:
            continue

        algo_str = algo_strs[algo_name]
        # seed_data is {seed: (steps, metrics)}
        # We need to aggregate them.

        # Assumption: All seeds have the same steps.
        # If not, we take the intersection or reference the first one.
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
        matrix = np.vstack(stacked_metrics)

        # Calculate Statistics
        mean = np.mean(matrix, axis=0)

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
                linewidth=2.5)    # Thicker, solid line

    # Plot Uniform Baseline (Dashed Line)
    if args.metric == "nash_conv" and uniform_nash_conv is not None:
        ax.axhline(y=uniform_nash_conv, xmin=0, xmax=max_steps, color='orange', linestyle='--', label="Uniform Policy", alpha=0.7)
        #ax.set_yscale('log')

    # Styling
    ax.legend(fontsize=15)
    ax.set_xlabel("Environment steps", fontsize=15)
    #ax.set_xscale('log')
    ax.set_ylabel(metric_str)
    #ax.set_ylabel("Episode return", fontsize=15)
    ax.set_title(f"Comparison {metric_str} on {args.game_name}", fontsize=20)
    #ax.set_title(f"NashDreamer obtained returns", fontsize=20)
    ax.grid(True, alpha=0.3)

    # if args.metric == "nash_conv":
    #     ax.set_yscale("log")

    # Save
    save_dir = f"plots/comparison/{game_path}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filename = f"{plot_str}_comparison.pdf"
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename))
    print(f"Plot saved to {os.path.join(save_dir, filename)}")


def main():
    args = parser.parse_args()
    plot_comparison(args)


if __name__ == "__main__":
    main()
