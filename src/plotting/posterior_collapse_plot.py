"""Plot posterior_collapse_eval.py's expected KL(posterior || prior) over training steps,
comparing multiple models. Chance nodes are clustered by their number of valid outcomes N
(computed per record as the count of nonzero gt_dist entries -- robust regardless of
whether the stored distribution is padded), since a posterior that merely fails to
distinguish outcomes will have a different achievable KL depending on how many outcomes
there were to distinguish. One file is produced per distinct N found across all models,
under output_dir/{spec}/posterior_collapse/posterior_collapse_K{N}.pdf, with a horizontal
baseline at log(N): the true entropy of a uniform N-outcome chance node.
"""

from argparse import ArgumentParser
import pickle

import numpy as np
import matplotlib.pyplot as plt

from plotting.plot_utils import (
    parse_model_paths, compute_spec_name, discover_seed_dirs, discover_step_files,
    plot_mean_and_seeds, save_figure, get_model_colors,
)

parser = ArgumentParser(description="Plot posterior_collapse_eval.py's expected "
                                     "KL(posterior || prior) over training steps, clustered by "
                                     "chance-node outcome count, comparing multiple models.")
parser.add_argument("--model_paths", type=str, required=True,
                     help="Comma-separated label=path pairs, e.g. "
                          "'no_representation=leduc_no_rep, full_loss=leduc'. Each path must "
                          "contain seed_{id} subdirectories with posterior_collapse_eval.pkl files.")
parser.add_argument("--output_dir", type=str, default="world_model_plots",
                     help="Plots are saved to output_dir/{spec}/posterior_collapse/.")

SUFFIX = "posterior_collapse_eval.pkl"


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


def num_outcomes(record: dict) -> int:
    return int(np.sum(record["gt_dist"] > 0))


def compute_kl_series(model_data: dict[int, dict[int, list[dict]]], n_outcomes: int):
    """Returns {seed: (steps_array, values_array)} of mean expected_kl across chance nodes
    with exactly n_outcomes valid outcomes, one point per step with such nodes."""
    per_seed = {}
    for seed, step_records in model_data.items():
        steps, values = [], []
        for step, records in sorted(step_records.items()):
            group = [r for r in records if num_outcomes(r) == n_outcomes]
            if not group:
                continue
            steps.append(step)
            values.append(float(np.mean([r["expected_kl"] for r in group])))
        if steps:
            per_seed[seed] = (np.array(steps), np.array(values))
    return per_seed


def main():
    args = parser.parse_args()
    model_paths = parse_model_paths(args.model_paths)
    spec = compute_spec_name(model_paths)
    colors = get_model_colors(model_paths)

    all_model_data = {}
    all_n_outcomes = set()
    for label, path in model_paths:
        all_model_data[label] = load_model_data(path)
        if not all_model_data[label]:
            print(f"Warning: no data found for model '{label}' at {path}")
        for step_records in all_model_data[label].values():
            for records in step_records.values():
                all_n_outcomes.update(num_outcomes(r) for r in records)

    for n in sorted(all_n_outcomes):
        fig, ax = plt.subplots(figsize=(10, 6))
        any_data = False
        for label, _ in model_paths:
            per_seed = compute_kl_series(all_model_data[label], n)
            any_data = any_data or bool(per_seed)
            plot_mean_and_seeds(ax, per_seed, colors[label], label)
        if not any_data:
            plt.close(fig)
            continue

        baseline = float(np.log(n))
        ax.axhline(y=baseline, color="gray", linestyle="-", linewidth=1.5, alpha=0.7,
                   label=f"Uniform baseline: log({n}) = {baseline:.3f}")

        ax.set_xlabel("Training step", fontsize=15)
        ax.set_ylabel("Expected KL(posterior || prior)", fontsize=15)
        ax.set_title(f"Posterior collapse: chance nodes with {n} outcomes", fontsize=15)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        save_figure(fig, args.output_dir, spec, "posterior_collapse", f"posterior_collapse_K{n}.pdf")
        plt.close(fig)


if __name__ == "__main__":
    main()
