from argparse import ArgumentParser
import dataclasses
import json
import os

import jax
import numpy as np

from train_utils import load_model, parse_sequence, track
from dreamer_ma import DreamerMA
from sim_rnad import SimRNaD
from sim_mmd import SimMMD
from sim_ppo import SimPPO
from experiments.policy_eval_utils import head_to_head_play


parser = ArgumentParser(description="Head-to-head evaluation between two algorithms across multiple seeds")
parser.add_argument("--base_path", type=str, default="trained_networks", help="Path to the directory of saved models")
parser.add_argument("--game_name", type=str, default="goofspiel_4", help="Name and parameter string of the game to evaluate for")
parser.add_argument("--seeds", type=str, default='(42, )', help="Seeds of the stored models to check. Supplied as a string (seed_1, seed_2, ..., seed_n)")
parser.add_argument("--restore_step_a", type=int, default=10000, help="Saved step of the first algorithm's model to restore")
parser.add_argument("--restore_step_b", type=int, default=10000, help="Saved step of the second algorithm's model to restore")
parser.add_argument("--algo_dirs", type=str, default="NashDreamer=nash_dreamer_rnad RNaD=rnad",
    help="Mapping of algorithm names to their directory names under base_path. Space-separated AlgoName=dir_name pairs. "
         "First algorithm plays as player 1, second as player 2.")
parser.add_argument("--num_games", type=int, default=1024, help="Number of games to play per configuration")
parser.add_argument("--epsilon_a", type=float, default=0.0, help="Uniform policy mix for the first algorithm (player 1 in A-vs-B). Set to 1.0 for a fully random player.")
parser.add_argument("--epsilon_b", type=float, default=0.0, help="Uniform policy mix for the second algorithm (player 2 in A-vs-B). Set to 1.0 for a fully random player.")
parser.add_argument("--rng_seed", type=int, default=42, help="RNG seed for game sampling")
parser.add_argument("--chunk_size", type=int, default=1024, help="Number of games per vmap chunk (to avoid OOM)")
parser.add_argument("--metric_store_dir", type=str, default="metrics/", help="Base path of the directory where to store the extracted metrics")


def _config_to_dict(model):
    """Extract config fields as a JSON-serializable dict based on model type."""
    if isinstance(model, DreamerMA):
        return {
            'wm_config': dataclasses.asdict(model.wm_config),
            'ac_config': dataclasses.asdict(model.ac_config),
            'buffer_config': dataclasses.asdict(model.buffer_config),
            'optimizer_config': dataclasses.asdict(model.opt_config),
        }
    elif isinstance(model, (SimRNaD, SimMMD, SimPPO)):
        out = {'config': dataclasses.asdict(model.config)}
        #SimPPO has no replay buffer, hence no buffer_config
        if hasattr(model, "buffer_config"):
            out['buffer_config'] = dataclasses.asdict(model.buffer_config)
        return out
    return {}

@track
def main():
  args = parser.parse_args()
  seeds = parse_sequence(args.seeds)

  # Parse algo_dirs in order — first is player 1, second is player 2
  algo_pairs = [pair.split("=", 1) for pair in args.algo_dirs.split()]
  assert len(algo_pairs) == 2, f"Expected exactly 2 algorithms in algo_dirs, got {len(algo_pairs)}"
  algo_name_a, dir_a = algo_pairs[0]
  algo_name_b, dir_b = algo_pairs[1]

  matchup_dir_name = f"{dir_a}_vs_{dir_b}"

  print(f"Head-to-head: {algo_name_a} (p1) vs {algo_name_b} (p2)")
  print(f"Game: {args.game_name}, Step A: {args.restore_step_a}, Step B: {args.restore_step_b}, Seeds: {seeds}")
  print(f"Games per matchup: {args.num_games}, Epsilon A: {args.epsilon_a}, Epsilon B: {args.epsilon_b}")
  print()

  # Load all models for both algorithms across seeds
  models_a = {}
  models_b = {}
  for seed in seeds:
    for models, algo_dir, algo_name, restore_step in [
        (models_a, dir_a, algo_name_a, args.restore_step_a),
        (models_b, dir_b, algo_name_b, args.restore_step_b),
    ]:
      model_path = os.path.join(args.base_path, algo_dir, args.game_name, f"seed_{seed}", f"step_{restore_step}.pkl")
      if not os.path.exists(model_path):
        print(f"Skipping {model_path} (not found)")
        continue
      model = load_model(model_path)
      assert isinstance(model, (DreamerMA, SimRNaD, SimMMD, SimPPO)), f"Expected DreamerMA, SimRNaD, SimMMD or SimPPO, got {type(model)}"
      models[seed] = model

  if not models_a or not models_b:
    print("No models loaded for one or both algorithms.")
    return

  # Get game from first loaded model
  first_model = next(iter(models_a.values()))
  game = first_model.game
  game_str = str(game)

  key = jax.random.key(args.rng_seed)

  for seed in seeds:
    if seed not in models_a or seed not in models_b:
      continue

    net_a = models_a[seed].optimizer.model
    net_b = models_b[seed].optimizer.model

    key, key_ab, key_ba = jax.random.split(key, 3)

    # A as player 0, B as player 1
    rewards_ab = np.asarray(head_to_head_play(game, net_a, net_b, args.num_games, key_ab, args.epsilon_a, args.epsilon_b, args.chunk_size))
    # B as player 0, A as player 1
    rewards_ba = np.asarray(head_to_head_play(game, net_b, net_a, args.num_games, key_ba, args.epsilon_b, args.epsilon_a, args.chunk_size))

    advantage = (rewards_ab.mean() - rewards_ba.mean()) / 2

    print(f"  Seed {seed}:")
    print(f"    {algo_name_a}(p1) vs {algo_name_b}(p2): mean={rewards_ab.mean():.4f}, "
          f"A wins={np.sum(rewards_ab > 0)}, draws={np.sum(rewards_ab == 0)}, B wins={np.sum(rewards_ab < 0)}")
    print(f"    {algo_name_b}(p1) vs {algo_name_a}(p2): mean={rewards_ba.mean():.4f}, "
          f"B wins={np.sum(rewards_ba > 0)}, draws={np.sum(rewards_ba == 0)}, A wins={np.sum(rewards_ba < 0)}")
    print(f"    {algo_name_a} advantage: {advantage:.4f}")

    # Save per-seed results to its own directory
    seed_dir = os.path.join(args.metric_store_dir, matchup_dir_name, args.game_name, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    metrics = {
      'algo_a': algo_name_a,
      'algo_a_dir': dir_a,
      'algo_b': algo_name_b,
      'algo_b_dir': dir_b,
      'game_name': args.game_name,
      'game_str': game_str,
      'restore_step_a': args.restore_step_a,
      'restore_step_b': args.restore_step_b,
      'num_games': args.num_games,
      'epsilon_a': args.epsilon_a,
      'epsilon_b': args.epsilon_b,
      'seed': int(seed),
      'a_as_p1_mean': float(rewards_ab.mean()),
      'a_as_p1_std': float(rewards_ab.std()),
      'a_as_p1_wins': int(np.sum(rewards_ab > 0)),
      'a_as_p1_draws': int(np.sum(rewards_ab == 0)),
      'a_as_p1_losses': int(np.sum(rewards_ab < 0)),
      'b_as_p1_mean': float(rewards_ba.mean()),
      'b_as_p1_std': float(rewards_ba.std()),
      'b_as_p1_wins': int(np.sum(rewards_ba > 0)),
      'b_as_p1_draws': int(np.sum(rewards_ba == 0)),
      'b_as_p1_losses': int(np.sum(rewards_ba < 0)),
      'advantage': float(advantage),
    }
    metrics_path = os.path.join(seed_dir, "head_to_head.json")
    with open(metrics_path, 'w') as f:
      json.dump(metrics, f, indent=2)
    print(f"  Metrics saved to {metrics_path}")

    configs = {
      algo_name_a: _config_to_dict(models_a[seed]),
      algo_name_b: _config_to_dict(models_b[seed]),
    }
    config_path = os.path.join(seed_dir, "config.json")
    with open(config_path, 'w') as f:
      json.dump(configs, f, indent=2)
    print(f"  Config saved to {config_path}")


if __name__ == "__main__":
  main()
