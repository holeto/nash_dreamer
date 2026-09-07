"""Plot world_model_sampling_eval.py's per-category rollout error over training steps,
comparing multiple models. Each rollout_validity.json already stores an explicit 'step'
and a 'per_category_error' dict averaged over all its batches, so no further aggregation
within a checkpoint is needed. Produces four files under
output_dir/{spec}/rollout_validity/: obs.pdf, reward.pdf, terminal.pdf, legal.pdf.
"""

from argparse import ArgumentParser
import json

import numpy as np
import matplotlib.pyplot as plt

from plotting.plot_utils import (
    parse_model_paths, compute_spec_name, discover_seed_dirs, discover_step_files,
    plot_mean_and_seeds, save_figure, get_model_colors,
)

parser = ArgumentParser(description="Plot world_model_sampling_eval.py's per-category rollout "
                                     "error over training steps, comparing multiple models.")
parser.add_argument("--model_paths", type=str, required=True,
                     help="Comma-separated label=path pairs, e.g. "
                          "'no_representation=leduc_no_rep, full_loss=leduc'. Each path must "
                          "contain seed_{id} subdirectories with rollout_validity.json files.")
parser.add_argument("--output_dir", type=str, default="world_model_plots",
                     help="Plots are saved to output_dir/{spec}/rollout_validity/.")

SUFFIX = "rollout_validity.json"
CATEGORIES = ("obs", "reward", "terminal", "legal")


def load_model_data(model_dir: str) -> dict[int, dict[int, dict]]:
    """Returns {seed: {step: per_category_error dict}}."""
    data = {}
    for seed, seed_dir in discover_seed_dirs(model_dir).items():
        step_files = discover_step_files(seed_dir, SUFFIX)
        if not step_files:
            continue
        data[seed] = {}
        for step, path in step_files.items():
            with open(path) as f:
                payload = json.load(f)
            data[seed][step] = payload["per_category_error"]
    return data


def compute_category_series(model_data: dict[int, dict[int, dict]], category: str):
    """Returns {seed: (steps_array, values_array)}."""
    per_seed = {}
    for seed, step_data in model_data.items():
        steps = sorted(step_data.keys())
        if not steps:
            continue
        values = [step_data[step][category] for step in steps]
        per_seed[seed] = (np.array(steps), np.array(values))
    return per_seed


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

    for category in CATEGORIES:
        fig, ax = plt.subplots(figsize=(10, 6))
        for label, _ in model_paths:
            per_seed = compute_category_series(all_model_data[label], category)
            plot_mean_and_seeds(ax, per_seed, colors[label], label)

        ax.set_xlabel("Training step", fontsize=15)
        ax.set_ylabel(f"{category.capitalize()} avg error", fontsize=15)
        ax.set_title(f"Rollout validity: {category} error", fontsize=15)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        save_figure(fig, args.output_dir, spec, "rollout_validity", f"{category}.pdf")
        plt.close(fig)


if __name__ == "__main__":
    main()
