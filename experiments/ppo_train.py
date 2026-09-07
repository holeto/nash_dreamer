from train_utils import PPOConfig, OptimizerConfig, parse_sequence, load_model
from experiments.joint_train import train_loop, SimPPO, JaxGame


def train_ppo(args, game: JaxGame):
  """Train a single agent PPO approximate best response against a frozen opponent.

  The opponent is loaded once from --opponent_path and never trained. Its actor is
  extracted at construction, so the checkpoint may be a DreamerMA, a SimRNaD or a
  SimMMD, as long as it was trained on the original game infosets."""
  seeds = parse_sequence(args.seeds)
  config = PPOConfig(
        bin_range = args.bin_range,
        report_gradnorms = args.report_gradnorms,

        player_id = args.player_id,

        #PPO parameters
        num_epochs = args.num_epochs,
        clip_epsilon = args.clip_epsilon,
        entropy_coeff = args.entropy_coeff,
        adv_norm_eps = args.adv_norm_eps,

        #TD(lambda)/GAE parameters
        gamma = args.gamma,
        td_lambda = args.td_lambda,

        sampling_epsilon = args.sampling_epsilon,

        #Return logging. There is no replay buffer, the learner tracks its own returns
        smoothing_window = args.smoothing_window,
        return_log_frequency = args.return_log_frequency,

        # Ordered as (hidden_layer_features, num_hidden_layers)
        actor_network_details = (args.actor_hidden_features, args.actor_hidden_layers),
        critic_network_details = (args.critic_hidden_features, args.critic_hidden_layers),

        target_network_update = args.target_network_update
    )
  opt_config = OptimizerConfig(lr = args.lr,
                                      agc = args.agc,
                                      eps = args.opt_eps,
                                      beta1 = args.beta_1,
                                      beta2 = args.beta_2,
                                      momentum = args.momentum,
                                      nesterov = args.nesterov,
                                      schedule = args.opt_schedule,
                                      warmup = args.warmup,
                                      anneal = args.anneal)

  print(f"Loading the frozen opponent from {args.opponent_path}")
  opponent = load_model(args.opponent_path)
  print(f"  opponent is a {type(opponent).__name__} trained on {opponent.game.to_compact_str()}")
  print(f"  learning a best response as player {args.player_id}")
  if args.entropy_coeff > 0:
    print(f"Warning! entropy_coeff={args.entropy_coeff} > 0 softens the best response, which "
          f"UNDER-estimates the opponent's exploitability. Use 0 when the best-response value "
          f"is the number you want.")

  template_model = SimPPO(game, config, opt_config, opponent, seeds[0], batch_size=args.batch_size)
  for seed in seeds:
    train_loop(args, seed, template_model)
