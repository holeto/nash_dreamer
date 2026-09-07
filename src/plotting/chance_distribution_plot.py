"""Plot chance_marginal_eval.py's reconstruction accuracy over training steps, comparing
multiple models. Produces four files under output_dir/{spec}/chance_distribution/:
chance_l1.pdf, chance_magnitude.pdf, deterministic_l1.pdf, deterministic_magnitude.pdf.

Per node (chance or deterministic), two metrics are averaged across all nodes of that
type at a given step:
  - L1: sum(abs(gt_dist - model_dist)) + error. The '+ error' term accounts for the
    probability mass evaluate_marginal couldn't assign to any real outcome (gt_dist and
    model_dist+error both sum to ~1) -- the same correction chance_marginal_eval.py's own
    summarize() applies to its total-variation figure.
  - magnitude: error (the unmatched probability mass), stored directly per record.
"""

from argparse import ArgumentParser
import pickle

import numpy as np
import matplotlib.pyplot as plt

from plotting.plot_utils import (
    parse_model_paths, compute_spec_name, discover_seed_dirs, discover_step_files,
    plot_mean_and_seeds, save_figure, get_model_colors,
)

parser = ArgumentParser(description="Plot chance_marginal_eval.py's L1/magnitude reconstruction "
                                     "error over training steps, comparing multiple models.")
parser.add_argument("--model_paths", type=str, required=True,
                     help="Comma-separated label=path pairs, e.g. "
                          "'no_representation=leduc_no_rep, full_loss=leduc'. Each path must "
                          "contain seed_{id} subdirectories with chance_marginal_eval.pkl files.")
parser.add_argument("--output_dir", type=str, default="world_model_plots",
                     help="Plots are saved to output_dir/{spec}/chance_distribution/.")

SUFFIX = "chance_marginal_eval.pkl"


def load_model_data(model_dir: str) -> dict[int, dict[int, list[dict]]]:
    """Returns {seed: {step: records_list}}."""
    data = {}
    for seed, seed_dir in discover_seed_dirs(model_dir).items():
        step_files = discover_step_files(seed_dir, SUFFIX)
        if not step_files:
            continue
        data[seed] = {}
        for step, path in step_files.items():
            with open(path, "rb") as f:
                data[seed][step] = pickle.load(f)
    return data


def compute_metric_series(model_data: dict[int, dict[int, list[dict]]], is_chance: bool, metric: str):
    """Returns {seed: (steps_array, values_array)}, one point per step where at least one
    node of the requested type was recorded."""
    per_seed = {}
    for seed, step_records in model_data.items():
        steps, values = [], []
        for step, records in sorted(step_records.items()):
            group = [r for r in records if bool(r["is_chance"]) == is_chance]
            if not group:
                continue
            if metric == "l1":
                per_node = [float(np.sum(np.abs(r["gt_dist"] - r["model_dist"])))
                            for r in group]
            else:
                per_node = [float(r["error"]) for r in group]
            steps.append(step)
            values.append(float(np.mean(per_node)))
        if steps:
            per_seed[seed] = (np.array(steps), np.array(values))
    return per_seed


def make_plot(all_model_data, model_paths, colors, is_chance, metric, output_dir, spec):
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, _ in model_paths:
        per_seed = compute_metric_series(all_model_data[label], is_chance, metric)
        plot_mean_and_seeds(ax, per_seed, colors[label], label)

    node_str = "chance" if is_chance else "deterministic"
    if metric == "l1":
        metric_str, filename = "L1 distance to true distribution", f"{node_str}_l1.pdf"
    else:
        metric_str, filename = "Unmatched probability mass (error)", f"{node_str}_magnitude.pdf"

    ax.set_xlabel("Training step", fontsize=15)
    ax.set_ylabel(metric_str, fontsize=15)
    ax.set_title(f"{node_str.capitalize()} nodes: {metric_str.lower()}", fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    save_figure(fig, output_dir, spec, "chance_distribution", filename)
    plt.close(fig)


def main():
    args = parser.parse_args()
    model_paths = parse_model_paths(args.model_paths)
    spec = compute_spec_name(model_paths)
    colors = get_model_colors(model_paths)

    all_model_data = {}
    for label, path in model_paths:
        all_model_data[label] = load_model_data(path)
        if not all_model_data[label]:
            print(f"Warning: no data found for model '{label}' at {path}")

    for is_chance in (True, False):
        for metric in ("l1", "magnitude"):
            make_plot(all_model_data, model_paths, colors, is_chance, metric, args.output_dir, spec)


if __name__ == "__main__":
    main()
