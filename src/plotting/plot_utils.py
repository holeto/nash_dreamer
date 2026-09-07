"""Shared helpers for the world-model plot scripts (chance_distribution_plot.py,
posterior_collapse_plot.py, rollout_validity_plot.py): parsing --model_paths,
discovering seed/step data on disk, and the mean-bold/per-seed-faint-dashed comparison
style established in plotting/plot_metrics.py."""

import os
import re

import numpy as np
import matplotlib.pyplot as plt


def parse_model_paths(spec: str) -> list[tuple[str, str]]:
    """Parse '--model_paths' of the form 'label1=path1, label2=path2' into
    [(label1, path1), (label2, path2)]."""
    model_paths = []
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        label, sep, path = entry.partition("=")
        if not sep:
            raise ValueError(f"Malformed --model_paths entry {entry!r}, expected 'label=path'.")
        model_paths.append((label.strip(), path.strip()))
    if not model_paths:
        raise ValueError("--model_paths must contain at least one 'label=path' entry.")
    return model_paths


def compute_spec_name(model_paths: list[tuple[str, str]]) -> str:
    """Single model -> its directory's basename. Multiple models -> each one's basename,
    concatenated with '_' in the given order."""
    basenames = [os.path.basename(os.path.normpath(path)) for _, path in model_paths]
    return "_".join(basenames)


def discover_seed_dirs(model_dir: str) -> dict[int, str]:
    """Returns {seed: seed_dir_path} for every seed_{id} subdirectory of model_dir."""
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model path {model_dir} does not exist or is not a directory.")
    seed_dirs = {}
    for name in os.listdir(model_dir):
        full = os.path.join(model_dir, name)
        if not os.path.isdir(full) or not name.startswith("seed_"):
            continue
        try:
            seed = int(name[len("seed_"):])
        except ValueError:
            continue
        seed_dirs[seed] = full
    return seed_dirs


def discover_step_files(seed_dir: str, suffix: str) -> dict[int, str]:
    """Returns {step: filepath} for files named 'step_{N}_{suffix}' in seed_dir. A bare
    '{suffix}' file (no step number, written by a single-checkpoint eval run) has no
    x-value to plot against and is skipped with a warning."""
    pattern = re.compile(rf"^step_(\d+)_{re.escape(suffix)}$")
    step_files = {}
    for filename in os.listdir(seed_dir):
        match = pattern.match(filename)
        if match:
            step_files[int(match.group(1))] = os.path.join(seed_dir, filename)
        elif filename == suffix:
            print(f"Warning: {os.path.join(seed_dir, filename)} has no step number in its "
                  f"filename (single-checkpoint run) -- skipping, nothing to plot on a step axis.")
    return step_files


def get_model_colors(model_paths: list[tuple[str, str]]) -> dict[str, str]:
    """Assigns each model label a color from matplotlib's default cycle, by order of
    appearance in --model_paths."""
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return {label: cycle[i % len(cycle)] for i, (label, _) in enumerate(model_paths)}


def plot_mean_and_seeds(ax, per_seed_data: dict[int, tuple[np.ndarray, np.ndarray]],
                         color: str, label: str):
    """Draws one model's data on ax: a faint dashed line per seed, and a bold solid mean
    line on top (averaged per step over whichever seeds have data at that step -- seeds
    need not share the exact same set of steps). Mirrors experiments/plot_metrics.py's
    plot_comparison aggregation style."""
    if not per_seed_data:
        return

    step_to_values: dict[int, list[float]] = {}
    for steps, values in per_seed_data.values():
        for step, value in zip(steps, values):
            step_to_values.setdefault(int(step), []).append(float(value))

    for steps, values in per_seed_data.values():
        order = np.argsort(steps)
        ax.plot(np.asarray(steps)[order], np.asarray(values)[order],
                color=color, alpha=0.3, linestyle="--", linewidth=1)

    mean_steps = sorted(step_to_values)
    mean_values = [np.mean(step_to_values[step]) for step in mean_steps]
    ax.plot(mean_steps, mean_values, color=color, linestyle="-", linewidth=2.5, label=label)


def save_figure(fig, output_dir: str, spec: str, plot_name: str, filename: str) -> str:
    """Saves fig to output_dir/{spec}/{plot_name}/{filename}, creating directories as needed."""
    save_dir = os.path.join(output_dir, spec, plot_name)
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    fig.tight_layout()
    fig.savefig(path)
    print(f"Plot saved to {path}")
    return path
