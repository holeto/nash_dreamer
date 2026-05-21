from argparse import ArgumentParser
import dataclasses
import json
import os
import time
from experiments.policy_eval_utils import *
from games.model_game import InformedRealGame
from train_utils import load_model, parse_sequence, track


parser = ArgumentParser()

parser.add_argument("--base_path", type=str, default="trained_networks", help="Path to the directory of saved models")
parser.add_argument("--game_name", type=str, default="goofspiel_3", help="Name and parameter string of the game to evaluate for")
parser.add_argument("--seeds", type=str, default='(42, )', help="Seeds of the stored models to check. Supplied as a string (seed_1, seed_2, ..., seed_n)")
parser.add_argument("--restore_step", type=int, default=10000, help="Saved step of the model to restore. If checking entire directory, -1 is also supported for all steps")

parser.add_argument("--scale_factor", type=float, default=1.0, help="Scale factor to multiply all rewards by. Useful if the game implementation scaled rewards in a different way than traditional implementations."
                    "Then, this should be the inverse of the game scaling factor. For example, JaxLeduc divides all rewards by 13, so to get values appriopriately scaled as in literature, this should be set to 13.")

experiment_parsers = parser.add_subparsers(dest="experiment_type", required=True, help="Which experiment type to run. Currently available are: loaded"
                                          "evaluate best responses against, or expected values of particular loaded model, or all models in the directory if restore_step is -1" \
                                          "nash: evaluate expected values of the model, best response values against it and also of a saved reference nash equilibrium strategy.")

loaded_parser = experiment_parsers.add_parser(name="loaded", help="Evaluate best responses against particular loaded model, or all models in the directory if restore_step is -1")
loaded_parser.add_argument("--metric", type=str, default="nash_conv", choices=("nash_conv", "expected_util", "env_return"), help="Type of metric to plot. Either NashConv, expected_utility, or smoothed environment returns during training.")
loaded_parser.add_argument("--metric_store_dir", type=str, default="metrics/", help="Base path of the directory where to store the extracted metrics. The full path will be metric_store_dir/algo_name/game_name.")
loaded_parser.add_argument("--algo_dirs", type=str, default="NashDreamer=nash_dreamer_rnad RNaD=rnad NashDreamerREINFORCE=nash_dreamer_reinforce", help="Mapping of algorithm names to their directory names under base_path. Space-separated AlgoName=dir_name pairs. Example: 'NashDreamer=nash_dreamer_rnad RNaD=rnad_replayed'")

nash_parser = experiment_parsers.add_parser(name="nash", help="Evaluate expected values of the model, best response values against it and also of a saved reference nash equilibrium strategy.")
nash_parser.add_argument("--nash_strategy_path", type=str, default="experiments/goofspiel_nash.pkl", help="Path to the saved nash strategy in pickle format. Must be formatted as a tuple of behavioral strategies per tree depth and infoset map per tree_depth.")

   
def parse_env_returns(model_dir):
    """ Parse the environment return smoothed averages logged
    in the directory and prepare them into an array for plotting.
    Returns: (steps, return_values, game_str, smoothing_window)
    """
    return_file = model_dir + "/env_returns.txt"
    if not os.path.exists(return_file):
        print(f"File {return_file} does not exist!")
        return None, None, None, None
    with open(return_file, 'r') as f:
       lines = f.readlines()
    #The first line contains the game_string
    game_str = lines[0].strip()
    #Second line contains the smoothing window
    smoothing_window = int(lines[1].split(':')[1].strip())
    steps = []
    returns = []
    for l in lines[2:]:
       step_part, return_part = l.split(',')
       step, ret = [float(p.split(':')[1].strip()) for p in (step_part, return_part)]
       steps.append(step)
       returns.append(ret)
    return np.asarray(steps, dtype=np.int32), np.asarray(returns), game_str, smoothing_window
    

@track
def get_metrics_from_dir(model_dir, args):
    """
    Scans a directory for checkpoints and calculates either
    expected return or NashConv metrics.
    Returns: (steps, metrics)
    """
    metrics = []
    steps = []
    
    if not model_dir.startswith("/"):
        model_dir = os.path.join(os.getcwd(), model_dir)
        
    if not os.path.exists(model_dir):
        print(f"Skipping {model_dir} (Not found)")
        return None, None, None

    print(f"Starting evaluation for: {model_dir}")
    start_time = time.time()
    
    first = True
    model = None
    game = None
    
    # Get all .pkl files and sort them by step to avoid jumping around
    files = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]
    
    for filename in files:
        step = int(filename.split("_")[-1].split(".")[0])

        model_path = os.path.join(model_dir, filename)

        if args.restore_step >=0 and (not step == args.restore_step):
            continue
        
        #try:
        if first:
            model = load_model(model_path)
            if isinstance(model, DreamerMA):
                pass
            elif isinstance(model, SimRNaD):
                pass
            else:
                assert False, f"Expected DreamerMA or SimRNAD, got {model.__class__}"
            
            # Initialize Game
            if isinstance(model, DreamerMA) and not model.optimizer.model.use_real_infoset:
                game = InformedRealGame(model)
            else:
                game = model.game
            first = False
        else:
            temp_model = load_model(model_path)
            nnx.update(model.optimizer, nnx.state(temp_model.optimizer))
            if isinstance(model, DreamerMA):
                model.actor_critic.learner_steps = temp_model.actor_critic.learner_steps
            model.learner_steps = temp_model.learner_steps
            
            if isinstance(model, DreamerMA) and not model.optimizer.model.use_real_infoset:
                game = InformedRealGame(model)

        # Calculate Metric
        if args.metric == "nash_conv":
            metric = nash_conv(model, game)
            #jax.debug.breakpoint()
        else:
            model_map_and_behaviorals = extract_model_policy(model, game)
            metric, _ = policy_expected_value(game, model_map_and_behaviorals)
        
        metric = args.scale_factor * metric
        metrics.append(metric)
        steps.append(step)
            
        # except Exception as e:
        #     breakpoint()
        #     print(f"Failed to process {filename}: {e}")
        #     continue

    print(f"Evaluation for {model_dir} took {time.time() - start_time:.2f} seconds.")
    
    # Sort results
    steps = np.asarray(steps)
    metrics = np.asarray(metrics)
    if len(steps) > 0:
        sort_indices = np.argsort(steps)
        sorted_steps = steps[sort_indices]
        batch_size = model.wm_config.batch_size if isinstance(model, DreamerMA) else model.batch_size
        ratio = 1 if model.buffer_config.replay_ratio <= 0 else model.buffer_config.replay_ratio
        env_steps = sorted_steps * ((batch_size * model.game.max_trajectory_lenght_no_chance()) / ratio)
        return env_steps, metrics[sort_indices], game, _wm_warmup_env_step(model)
    else:
        return [], [], game, -1


def _config_to_dict(model):
    """Extract config fields as a JSON-serializable dict based on model type."""
    if isinstance(model, DreamerMA):
        return {
            'wm_config': dataclasses.asdict(model.wm_config),
            'ac_config': dataclasses.asdict(model.ac_config),
            'buffer_config': dataclasses.asdict(model.buffer_config),
            'optimizer_config': dataclasses.asdict(model.opt_config),
        }
    elif isinstance(model, SimRNaD):
        return {
            'config': dataclasses.asdict(model.config),
            'buffer_config': dataclasses.asdict(model.buffer_config),
        }
    return {}


def _wm_warmup_env_step(model):
    """Return the env step at which the world model warm-up period ends.
    Returns -1 if the model has no world model or warm-up period is 0."""
    if not isinstance(model, DreamerMA):
        return -1
    warm_up_grad_steps = model.ac_config.wm_warm_up_period
    if warm_up_grad_steps <= 0:
        return -1
    batch_size = model.wm_config.batch_size
    ratio = 1 if model.buffer_config.replay_ratio <= 0 else model.buffer_config.replay_ratio
    env_step_per_grad_step = (batch_size * model.game.max_trajectory_lenght_no_chance()) / ratio
    return float(warm_up_grad_steps * env_step_per_grad_step)


def _wm_warmup_env_step_from_dir(model_dir):
    """Load the first checkpoint from model_dir and return its warmup env step."""
    if not model_dir.startswith("/"):
        model_dir = os.path.join(os.getcwd(), model_dir)
    if not os.path.exists(model_dir):
        return -1
    files = sorted(f for f in os.listdir(model_dir) if f.endswith(".pkl"))
    if not files:
        return -1
    model = load_model(os.path.join(model_dir, files[0]))
    return _wm_warmup_env_step(model)


def extract_config_from_dir(model_dir, restore_step):
    """Load one checkpoint from model_dir and return its config as a dict."""
    if not model_dir.startswith("/"):
        model_dir = os.path.join(os.getcwd(), model_dir)
    if not os.path.exists(model_dir):
        return None
    files = sorted(f for f in os.listdir(model_dir) if f.endswith(".pkl"))
    for filename in files:
        step = int(filename.split("_")[-1].split(".")[0])
        if restore_step >= 0 and step != restore_step:
            continue
        model = load_model(os.path.join(model_dir, filename))
        return _config_to_dict(model)
    return None


def save_metrics(results, game_str, smoothing_window, uniform_nash_conv, algo_configs, plotting_names, algo_warmup_steps, args):
    """Save extracted metrics to text files under metric_store_dir/algo_name/game_name/metric.txt.
    Also saves a config.json in the same directory when config data is available."""
    for algo_name, seed_data in results.items():
        save_dir = os.path.join(args.metric_store_dir, algo_name, args.game_name)
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, f"{args.metric}.txt")
        with open(filepath, 'w') as f:
            f.write(f"algo_name: {plotting_names[algo_name]}\n")
            f.write(f"game_name: {args.game_name}\n")
            f.write(f"game_str: {game_str}\n")
            f.write(f"metric: {args.metric}\n")
            f.write(f"smoothing_window: {smoothing_window}\n")
            f.write(f"wm_warmup_env_step: {algo_warmup_steps.get(algo_name, -1)}\n")
            if uniform_nash_conv is not None:
                f.write(f"uniform_nash_conv: {uniform_nash_conv}\n")
            for seed, (steps, metrics) in seed_data.items():
                f.write(f"seed: {seed}\n")
                f.write("steps: " + " ".join(str(s) for s in steps) + "\n")
                f.write("values: " + " ".join(str(v) for v in metrics) + "\n")
        print(f"Metrics for {algo_name} saved to {filepath}")
        if algo_name in algo_configs:
            config_path = os.path.join(save_dir, "config.json")
            with open(config_path, 'w') as f:
                json.dump(algo_configs[algo_name], f, indent=2)
            print(f"Config for {algo_name} saved to {config_path}")


def plot_comparison(args):
    """
    Main function to extract metrics for NashDreamer vs RNaD for a specific game
    and save them to disk.
    """
    base_path = args.base_path
    game_path = args.game_name
    seeds = parse_sequence(args.seeds)
    seed_paths = [f"seed_{s}" for s in seeds]

    # Parse algo_dirs: "AlgoName=dir_name ..." -> {"AlgoName": "dir_name", ...}
    algo_dir_map = dict(pair.split("=", 1) for pair in args.algo_dirs.split())

    # Define the algorithms to compare, using the mapped directory names
    algos = {
        algo_name: [os.path.join(base_path, dir_name, game_path, p) for p in seed_paths]
        for algo_name, dir_name in algo_dir_map.items()
    }
    names = {v: k for k, v in algo_dir_map.items()}
    algos = {algo_dir_map[k]: v for k, v in algos.items()}
    print(f"Evaluating for {algos}")

    results = {v: {} for k, v in algo_dir_map.items()}
    algo_configs = {}
    algo_warmup_steps = {}
    game_str = ""
    game = None
    smoothing_window = -1
    max_steps = 0

    def metrics_wrapper(directory, args):
        """Just a wrapper function to
        handle the interface discrepancy between
        the environment returns, and game theoretic metric
        computation."""
        if args.metric == 'env_return':
            steps, metrics, game_str, smoothing_window = parse_env_returns(directory)
            game = None
            wm_warmup_env_step = _wm_warmup_env_step_from_dir(directory)
        else:
            steps, metrics, game, wm_warmup_env_step = get_metrics_from_dir(d, args)
            game_str = str(game)
            smoothing_window = -1
        return steps, metrics, game, game_str, smoothing_window, wm_warmup_env_step



    # 1. Collect Data
    for algo_name, dir_paths in algos.items():
        for s, d in zip(seeds,dir_paths):
            steps, metrics, game, new_game_str, new_smoothing_window, wm_warmup = metrics_wrapper(d, args)
            if algo_name not in algo_configs and args.metric != 'env_return':
                config = extract_config_from_dir(d, args.restore_step)
                if config:
                    algo_configs[algo_name] = config
            if algo_name not in algo_warmup_steps and wm_warmup >= 0:
                algo_warmup_steps[algo_name] = wm_warmup
            if steps is not None and len(steps) > 0:
                max_steps = max(max_steps, len(steps))
                results[algo_name][s] = (steps, metrics)
                #Check if all experiments used the same game
                if not game_str:
                    game_str = new_game_str
                else:
                    assert new_game_str == game_str, f"Expected all models to use the same game {game_str}. Found {new_game_str} for algorithm {algo_name} seed {s} instead!"
                #Check if all experiments
                # used the same smoothing window
                # (relevant for the environment returns only)
                if smoothing_window < 0:
                    smoothing_window = new_smoothing_window
                else:
                    assert smoothing_window == new_smoothing_window, f"Expected all models to use the same smoothing window {smoothing_window}. Found {new_smoothing_window} for algorithm {algo_name} seed {s} instead!"

    if not results:
        print("No data found for either algorithm.")
        return

    # 2. Compute uniform nash conv baseline before saving (requires game object)
    uniform_nash_conv = None
    if args.metric == "nash_conv" and game:
        #Using the overloaded functionality of extract model policy
        # to get uniform policy for the game
        uniform_policy = extract_model_policy(None, game, uniform=True)
        uniform_nash_conv = args.scale_factor * nash_conv(None, game, uniform_policy)

    # 3. Save metrics
    save_metrics(results, game_str, smoothing_window, uniform_nash_conv, algo_configs, names, algo_warmup_steps, args)

def test_nash(args, saved_nash_path: str):
  model_path = args.model_dir
  if not model_path.startswith("/"):
    model_path = os.getcwd() + "/" + model_path
  model_path = model_path + f"/step_{args.restore_step}.pkl"
  if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file {model_path} does not exist.")
  
  nash_path= saved_nash_path
  if not nash_path.startswith("/"):
    nash_path = os.getcwd() + "/" + nash_path
  if not os.path.exists(nash_path):
    raise FileNotFoundError(f"Nash policy file {nash_path} does not exist.")
  
  print(f"Evaluating policy of model loaded from {model_path} against nash policy loaded from {saved_nash_path}")

  model = load_model(model_path)
  # game = JaxLeduc()
  # buffer = ReplayBuffer(game, 0, 0, 100)
  # dreamer_model = DreamerMA(DreamerMAConfig(), buffer)
  # model = RNaDDreamerJoint(dreamer_model, RNaDConfig())
  assert isinstance(model, DreamerMA), f"The loaded model should be an instance of DreamerMA. Instead got {model.__class__}"
  
  game = InformedRealGame(model) if (isinstance(model, DreamerMA) and not model.optimizer.model.use_real_infoset) else model.game
  p1_nash_val, p2_nash_val, nash_infoset_map, nash_behaviorals = load_model(nash_path)
  print(f"Loaded nash policies of game with game value {p1_nash_val} (from player 1 perspective)")
  model_map, model_behaviorals = extract_model_policy(model, game)
  found_p1_nash, found_p2_nash = policy_expected_value(game, (nash_infoset_map, nash_behaviorals), eps=1e-5)
  found_p1_nash, found_p2_nash = args.scale_factor * found_p1_nash, args.scale_factor * found_p2_nash
  print(f"Found nash values: {found_p1_nash} {found_p2_nash}")
  assert np.isclose(found_p1_nash, p1_nash_val, atol=1e-5), f"Found nash value {found_p1_nash} and saved nash value {p1_nash_val} for player 1 differ!"
  assert np.isclose(found_p1_nash, p1_nash_val, atol=1e-5), f"Found nash value {found_p2_nash} and saved nash value {p2_nash_val} for player 2 differ!"
  p2_br_val, p1_br_val, p1_br, p2_br = model_best_response(model, game, (nash_infoset_map, nash_behaviorals))
  print(f"Found nash exploitabilities:")
  print(f"P2 best response value against p1: {p2_br_val}")
  print(f"P1 best response value against p2 {p1_br_val}")
  model_p1_val, model_p2_val = policy_expected_value(game, (model_map, model_behaviorals))
  model_p1_val, model_p2_val = args.scale_factor * model_p1_val, args.scale_factor * model_p2_val
  print(f"Model values {model_p1_val}, {model_p2_val}")
  p2_br_val, p1_br_val, p1_br, p2_br = model_best_response(model, game)
  p1_br_val, p2_br_val = args.scale_factor * p1_br_val, args.scale_factor * p2_br_val
  print(f"P2 best response value against p1: {p2_br_val}")
  print(f"P1 best response value against p2 {p1_br_val}")
  #compare_policies(model.world_model.game, (model_map, model_behaviorals), (nash_infoset_map, nash_behaviorals))
        
  

def main():
  args = parser.parse_args()
  if args.experiment_type == "nash":
    test_nash(args, saved_nash_path=args.nash_strategy_path)
  else:
    plot_comparison(args)
  

if __name__ == "__main__":
  main()