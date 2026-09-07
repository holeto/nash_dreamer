"""Plot NashConv against total steps SEEN BY THE ACTOR-CRITIC (real + imagined/replayed),
rather than plot_metrics.py's environment steps. Reuses plot_metrics.py's loader and
renderer unchanged; only the x-axis values fed into the shared renderer differ.

    python -m plotting.plot_total_steps_comparison \
        --metric_store_dir precomputed_metrics/nash_conv --game_name leduc \
        --algos "nash_dreamer_rnad_factored nash_dreamer_mmd_factored rnad mmd replayed_rnad"

Transform, applied per algo in already-stored env_step space (no need to recover raw
gradient steps or batch_size):

    total_step(env_step) = env_step                                            if env_step <= wm_warmup_env_step
    total_step(env_step) = wm_warmup_env_step + (env_step - wm_warmup_env_step)
                            * (max_trajectory_lenght_no_chance() + 1)           if env_step >  wm_warmup_env_step

Three modes, classified by algo_dir name (not config.json structure -- replayed_rnad
and rnad are both SimRNaD-shaped, so there is no structural signal to tell them apart):
  - env_steps_only: default. total_step = env_step, unchanged.
  - warmup_split:   algo_dir contains "nash_dreamer" (covers *_factored variants too).
                     wm_warmup_env_step is NOT trusted from the .txt file's own header --
                     confirmed unreliable for non-DreamerMA algos (e.g.
                     precomputed_metrics/nash_conv/rnad/leduc/nash_conv.txt carries a
                     nonzero wm_warmup_env_step copied from whatever NashDreamer run was
                     evaluated in the same historical batch invocation, despite RNaD
                     having no warm-up concept). Recomputed instead from the algo's own
                     sibling config.json, the same formula actor_critic_evaluate.py's
                     _wm_warmup_env_step uses, evaluated on the config dict directly.
  - replayed_full:  --replayed_algo_dirs (default "replayed_rnad"). No warm-up phase, so
                     the post-warmup branch applies to the whole curve
                     (wm_warmup_env_step = 0), per the assumption that its replay buffer
                     replays one trajectory per real trajectory -- a FIXED traj_len+1
                     multiplier, never derived from the algo's actual configured
                     replay_ratio (which can be wrong: replayed_rnad/leduc's config.json
                     shows replay_ratio=8 against the correct traj_len+1=10, a confirmed
                     off-by-one in how that run was trained, not a reason to use 8 here).

Output: plots/steps_seen_comparison/<game_name>/nash_conv_comparison.{pdf,pdf}, to avoid
colliding with plot_metrics.py's own plots/comparison/<game_name>.
"""

import json
import os
from argparse import ArgumentParser

import numpy as np

from plotting.plot_metrics import load_algo_metrics, render_plot

parser = ArgumentParser()
parser.add_argument("--metric_store_dir", type=str, default="metrics/",
                    help="Base path of the directory where extracted metrics are stored. "
                         "The full path per algo is metric_store_dir/algo_name/game_name.")
parser.add_argument("--game_name", type=str, default="goofspiel_3",
                    help="Name and parameter string of the game to plot for.")
parser.add_argument("--algos", type=str, default="NashDreamer RNaD",
                    help="Which algorithms (directory names) to plot.")
parser.add_argument("--replayed_algo_dirs", type=str, default="replayed_rnad",
                    help="Space-separated --algos entries to treat as replayed_full "
                         "(no warm-up, fixed traj_len+1 multiplier applied throughout).")
parser.add_argument("--log_x", action="store_true", help="Plot x-axis in log scale.")
parser.add_argument("--log_y", action="store_true", help="Plot y-axis in log scale.")
parser.add_argument("--divide_by", type=float, default=1.0,
                    help="Divide all metric values (and the uniform-policy baseline) by "
                         "this constant before plotting.")
parser.add_argument("--skip_divide", type=str, default="",
                    help="Space-separated subset of --algos entries to exclude from "
                         "--divide_by (e.g. baselines that are already normalized).")
parser.add_argument("--no_warmup_line", action="store_true",
                    help="Don't plot the WM warm-up vertical line.")
parser.add_argument("--save_dir", type=str, default=None,
                    help="Directory to save the plot into. Defaults to "
                         "plots/steps_seen_comparison/<game_name>.")
parser.add_argument("--labels", type=str, default=None,
                    help="Optional comma-separated list of legend labels, one per entry "
                         "in --algos (in order), overriding the algo_name recorded in "
                         "the metrics file.")
parser.add_argument("--clamp_checkpoints", type=str, default="",
                    help="Space-separated subset of --algos entries whose steps/values "
                         "are truncated to the first --clamp_to checkpoints.")
parser.add_argument("--clamp_to", type=int, default=None,
                    help="Number of checkpoints to truncate --clamp_checkpoints entries to.")
parser.add_argument("--skip_first_checkpoint", action="store_true",
                    help="Drop each seed's first (step=0, untrained-network) checkpoint "
                         "before plotting.")

# metric is always nash_conv for this script: total-steps-seen only makes sense for a
# quantity plotted against training progress, and every (algo, game) this bridges to
# was produced by either actor_critic_evaluate.py or aggregate_results.py, both of
# which only ever write nash_conv.txt.
METRIC = "nash_conv"


def trajectory_length(game_name):
    """max_trajectory_lenght_no_chance() for game_name, derived from the real game
    class rather than hardcoded, so this stays correct if a game's constants change."""
    if game_name == "leduc":
        from envs.jax_leduc import JaxLeduc
        return JaxLeduc().max_trajectory_lenght_no_chance()
    if game_name.startswith("goofspiel"):
        # goofspiel_5, goofspiel_5_obs, goofspiel_3, ... -- cards is the leading digits
        # right after "goofspiel_", observation_only doesn't affect trajectory length.
        cards = int(game_name.split("_")[1])
        from envs.jax_goofspiel import JaxGoofspiel
        return JaxGoofspiel(cards=cards).max_trajectory_lenght_no_chance()
    if game_name == "phantom_ttt":
        from envs.jax_phantom_ttt import JaxPhantomTTT
        return JaxPhantomTTT().max_trajectory_lenght_no_chance()
    raise SystemExit(f"Don't know how to derive trajectory length for game {game_name!r}. "
                      f"Add a case to trajectory_length() in this script.")


def classify_mode(algo_dir, replayed_algo_dirs):
    if "nash_dreamer" in algo_dir:
        return "warmup_split"
    if algo_dir in replayed_algo_dirs:
        return "replayed_full"
    return "env_steps_only"


def recompute_wm_warmup_env_step(metric_store_dir, algo_dir, game_name, traj_len):
    """Reads the algo's own sibling config.json and recomputes wm_warmup_env_step
    directly, the same formula as actor_critic_evaluate.py's _wm_warmup_env_step,
    evaluated on the config dict since there is no live model object here. Does NOT
    trust the .txt file's own wm_warmup_env_step header -- confirmed unreliable for at
    least some non-DreamerMA algos in this corpus (see module docstring)."""
    config_path = os.path.join(metric_store_dir, algo_dir, game_name, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    warm_up_grad_steps = config["ac_config"]["wm_warm_up_period"]
    if warm_up_grad_steps <= 0:
        return -1
    batch_size = config["wm_config"]["batch_size"]
    ratio = config["buffer_config"]["replay_ratio"]
    ratio = 1 if ratio is None or ratio <= 0 else ratio
    env_steps_per_grad_step = (batch_size * traj_len) / ratio
    return float(warm_up_grad_steps * env_steps_per_grad_step)


def transform_steps(steps, mode, wm_warmup_env_step, traj_len):
    """env_step -> total_step, the piecewise transform. wm_warmup_env_step=0 for
    replayed_full collapses this to steps * (traj_len + 1) uniformly, so replayed_full
    and warmup_split share one implementation. Continuous at the boundary either way
    (both branches evaluate to exactly `boundary` when steps == boundary, since the
    post-branch's (steps - boundary) term is 0 there), so <= vs < makes no numeric
    difference -- <= just reads naturally as "still during warm-up" at that point."""
    if mode == "env_steps_only":
        return steps
    boundary = wm_warmup_env_step if mode == "warmup_split" else 0.0
    boundary = max(boundary, 0.0)  # wm_warmup_env_step is -1 when there is no warm-up
    return np.where(steps <= boundary, steps, boundary + (steps - boundary) * (traj_len + 1))


def main():
    args = parser.parse_args()
    algo_names = args.algos.split()
    labels = args.labels.split(",") if args.labels else None
    clamp_checkpoints = set(args.clamp_checkpoints.split())
    replayed_algo_dirs = set(args.replayed_algo_dirs.split())
    traj_len = trajectory_length(args.game_name)

    results = {}
    algo_strs = {}
    game_str = ""
    smoothing_window = -1
    uniform_nash_conv = None
    uniform_nash_conv_divisor = 1.0
    skip_divide = set(args.skip_divide.split())
    algo_warmup_steps = {}
    max_steps = 0

    for i, algo_dir in enumerate(algo_names):
        seed_data, new_game_str, algo_str, new_smoothing_window, new_uniform_nash_conv, _ = load_algo_metrics(
            args.metric_store_dir, algo_dir, args.game_name, METRIC
        )
        algo_strs[algo_dir] = labels[i] if labels else algo_str
        if seed_data is None:
            continue

        mode = classify_mode(algo_dir, replayed_algo_dirs)
        wm_warmup_env_step = -1
        if mode == "warmup_split":
            wm_warmup_env_step = recompute_wm_warmup_env_step(
                args.metric_store_dir, algo_dir, args.game_name, traj_len)
        print(f"{algo_dir}: mode={mode} wm_warmup_env_step={wm_warmup_env_step} traj_len={traj_len}")

        seed_data = {
            s: (transform_steps(steps, mode, wm_warmup_env_step, traj_len), metrics)
            for s, (steps, metrics) in seed_data.items()
        }
        if algo_dir in clamp_checkpoints:
            seed_data = {s: (steps[:args.clamp_to], metrics[:args.clamp_to]) for s, (steps, metrics) in seed_data.items()}

        results[algo_dir] = seed_data
        if wm_warmup_env_step > 0:
            algo_warmup_steps[algo_dir] = wm_warmup_env_step
        for s, (steps, metrics) in seed_data.items():
            max_steps = max(max_steps, len(steps))

        if not game_str:
            game_str = new_game_str
        else:
            assert new_game_str == game_str, (
                f"Expected all models to use the same game {game_str}. "
                f"Found {new_game_str} for algorithm {algo_dir} instead!"
            )
        if smoothing_window < 0:
            smoothing_window = new_smoothing_window
        else:
            assert smoothing_window == new_smoothing_window
        if new_uniform_nash_conv is not None:
            uniform_nash_conv = new_uniform_nash_conv
            uniform_nash_conv_divisor = 1.0 if algo_dir in skip_divide else args.divide_by

    if not results:
        print("No data found for any algorithm.")
        return

    if not args.save_dir:
        args.save_dir = f"plots/steps_seen_comparison/{args.game_name}"
    # render_plot reads args.metric for the y-axis label/filename; this script always
    # scores nash_conv, so pin it rather than require --metric on every invocation.
    args.metric = METRIC

    render_plot(results, algo_strs, algo_names, game_str, smoothing_window,
                uniform_nash_conv, uniform_nash_conv_divisor, algo_warmup_steps,
                max_steps, args, x_label="Total steps seen by actor-critic", markers=True,
                skip_first_checkpoint=args.skip_first_checkpoint)


if __name__ == "__main__":
    main()
