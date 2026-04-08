from argparse import ArgumentParser
import json

import matplotlib.pyplot as plt
import numpy as np


parser = ArgumentParser(description="Plot head-to-head evaluation results")
parser.add_argument("--metrics_path", type=str, required=True, help="Path to head_to_head.json file")
parser.add_argument("--output", type=str, default=None, help="Output image path. If not set, displays the plot.")


def main():
  args = parser.parse_args()

  with open(args.metrics_path, 'r') as f:
    data = json.load(f)

  algo_a = data['algo_a']
  algo_b = data['algo_b']
  game_name = data['game_name']
  num_games = data['num_games']
  seeds = data['seeds']

  # Collect per-seed win/draw/loss ratios for both configurations
  seed_keys = sorted(seeds.keys(), key=int)
  a_as_p1_ratios = []  # Each: (win_ratio, draw_ratio, loss_ratio)
  b_as_p1_ratios = []

  for s in seed_keys:
    r = seeds[s]
    a_as_p1_ratios.append([r['a_as_p1_wins'] / num_games, r['a_as_p1_draws'] / num_games, r['a_as_p1_losses'] / num_games])
    b_as_p1_ratios.append([r['b_as_p1_losses'] / num_games, r['b_as_p1_draws'] / num_games, r['b_as_p1_wins'] / num_games])

  a_as_p1_ratios = np.array(a_as_p1_ratios)  # [seeds, 3]
  b_as_p1_ratios = np.array(b_as_p1_ratios)  # [seeds, 3]

  # Mean ratios across seeds
  a_mean = a_as_p1_ratios.mean(axis=0)
  b_mean = b_as_p1_ratios.mean(axis=0)

  fig, ax = plt.subplots(1, 1, figsize=(8, 5))

  categories = [f'{algo_a} as P1', f'{algo_b} as P1']
  x = np.arange(len(categories))
  bar_width = 0.5

  colors = {'win': '#4CAF50', 'draw': '#9E9E9E', 'loss': '#F44336'}

  # Plot per-seed ratios as transparent thin bars
  thin_width = bar_width * 0.6
  for i, (a_r, b_r) in enumerate(zip(a_as_p1_ratios, b_as_p1_ratios)):
    for j, (ratios, xpos) in enumerate([(a_r, x[0]), (b_r, x[1])]):
      bottom = 0
      for k, (val, color_key) in enumerate(zip(ratios, ['win', 'draw', 'loss'])):
        ax.bar(xpos, val, thin_width, bottom=bottom, color=colors[color_key], alpha=0.15, edgecolor='none')
        bottom += val

  # Plot mean ratios as bold bars
  for j, (mean_ratios, xpos) in enumerate([(a_mean, x[0]), (b_mean, x[1])]):
    bottom = 0
    for k, (val, color_key, label) in enumerate(zip(mean_ratios, ['win', 'draw', 'loss'],
                                                      [f'{algo_a} wins', 'Draws', f'{algo_b} wins'])):
      bar = ax.bar(xpos, val, bar_width, bottom=bottom, color=colors[color_key],
                   alpha=0.85, edgecolor='white', linewidth=0.5,
                   label=label if j == 0 else None)
      if val > 0.05:
        ax.text(xpos, bottom + val / 2, f'{val:.1%}', ha='center', va='center', fontsize=9, fontweight='bold')
      bottom += val

  ax.set_xticks(x)
  ax.set_xticklabels(categories, fontsize=11)
  ax.set_ylabel('Ratio', fontsize=11)
  ax.set_ylim(0, 1)
  ax.set_title(f'{algo_a} vs {algo_b} — {game_name}\n({num_games} games/matchup, {len(seed_keys)} seed{"s" if len(seed_keys) > 1 else ""})',
               fontsize=12)
  ax.legend(loc='upper right', fontsize=9)

  # Add overall advantage text
  mean_adv = data['overall_mean_advantage']
  std_adv = data['overall_std_advantage']
  ax.text(0.5, -0.12, f'{algo_a} mean advantage: {mean_adv:.4f} +/- {std_adv:.4f}',
          ha='center', va='top', transform=ax.transAxes, fontsize=10)

  plt.tight_layout()
  if args.output:
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {args.output}")
  else:
    plt.show()


if __name__ == "__main__":
  main()
