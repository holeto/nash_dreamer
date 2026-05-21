from argparse import ArgumentParser
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np


parser = ArgumentParser(description="Plot head-to-head evaluation results")
parser.add_argument("--metrics_dir", type=str, required=True,
    help="Directory containing per-seed subdirectories (seed_*/head_to_head.json)")
parser.add_argument("--output", type=str, default=None, help="Output image path. If not set, displays the plot.")


def main():
  args = parser.parse_args()

  seed_files = sorted(glob.glob(os.path.join(args.metrics_dir, "seed_*", "head_to_head.json")))
  if not seed_files:
    print(f"No seed_*/head_to_head.json files found in {args.metrics_dir}")
    return

  seed_data = []
  for path in seed_files:
    with open(path, 'r') as f:
      seed_data.append(json.load(f))

  # Use metadata from the first file
  first = seed_data[0]
  algo_a = first['algo_a']
  algo_b = first['algo_b']
  game_name = first['game_name']
  num_games = first['num_games']

  seed_keys = [str(d['seed']) for d in seed_data]

  # Collect per-seed averaged win/draw/loss ratios (averaged over both player roles)
  a_avg_ratios = []
  b_avg_ratios = []
  advantages = []

  for d in seed_data:
    a_p1 = np.array([d['a_as_p1_wins'], d['a_as_p1_draws'], d['a_as_p1_losses']]) / num_games
    a_p2 = np.array([d['b_as_p1_losses'], d['b_as_p1_draws'], d['b_as_p1_wins']]) / num_games
    a_avg = (a_p1 + a_p2) / 2
    a_avg_ratios.append(a_avg)
    b_avg_ratios.append(np.array([a_avg[2], a_avg[1], a_avg[0]]))
    advantages.append(d['advantage'])

  a_avg_ratios = np.array(a_avg_ratios)  # [seeds, 3]
  b_avg_ratios = np.array(b_avg_ratios)  # [seeds, 3]
  advantages = np.array(advantages)

  mean_adv = float(advantages.mean())
  std_adv = float(advantages.std())

  print(f"Loaded {len(seed_data)} seed(s): {seed_keys}")
  print(f"Overall {algo_a} vs {algo_b}:")
  print(f"  Mean advantage: {mean_adv:.4f} +/- {std_adv:.4f}")

  # Mean ratios across seeds
  a_mean = a_avg_ratios.mean(axis=0)
  b_mean = b_avg_ratios.mean(axis=0)

  fig, ax = plt.subplots(1, 1, figsize=(5, 5))

  bar_width = 0.5
  xpos = 0

  colors = {'win': '#4CAF50', 'draw': '#9E9E9E', 'loss': '#F44336'}

  # Plot per-seed ratios as transparent thin bars
  thin_width = bar_width * 0.6
  for a_r in a_avg_ratios:
    bottom = 0
    for val, color_key in zip(a_r, ['win', 'draw', 'loss']):
      ax.bar(xpos, val, thin_width, bottom=bottom, color=colors[color_key], alpha=0.15, edgecolor='none')
      bottom += val

  # Plot mean ratios as bold bar
  bottom = 0
  for val, color_key, label in zip(a_mean, ['win', 'draw', 'loss'],
                                   [f'{algo_a} wins', 'Draws', f'{algo_b} wins']):
    ax.bar(xpos, val, bar_width, bottom=bottom, color=colors[color_key],
           alpha=0.85, edgecolor='white', linewidth=0.5, label=label)
    if val > 0.05:
      ax.text(xpos, bottom + val / 2, f'{val:.1%}', ha='center', va='center', fontsize=9, fontweight='bold')
    bottom += val

  ax.set_xticks([xpos])
  ax.set_xticklabels([f'{algo_a} vs {algo_b}'], fontsize=11)
  ax.set_ylabel('Ratio', fontsize=11)
  ax.set_ylim(0, 1)
  ax.set_title(f'{algo_a} vs {algo_b} — {game_name}\n({num_games} games/matchup, {len(seed_data)} seed{"s" if len(seed_data) > 1 else ""})',
               fontsize=12)
  ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=9)

  ax.text(0.5, -0.12, f'{algo_a} mean advantage: {mean_adv:.4f} +/- {std_adv:.4f}',
          ha='center', va='top', transform=ax.transAxes, fontsize=10)

  plt.tight_layout()
  if args.output:
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {args.output}")
  else:
    plt.show()


if __name__ == "__main__":
  main()
