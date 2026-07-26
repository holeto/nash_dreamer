from train_utils import MMDConfig, OptimizerConfig, BufferConfig, parse_sequence
from experiments.joint_train import train_loop, SimMMD, JaxGame


def train_mmd(args, game:JaxGame):
  seeds = parse_sequence(args.seeds)
  config = MMDConfig(
        bin_range = args.bin_range,
        report_gradnorms = args.report_gradnorms,

        #MMD/PPO parameters
        num_epochs = args.num_epochs,
        clip_epsilon = args.clip_epsilon,
        kl_coeff = args.kl_coeff,
        magnet_coeff = args.magnet_coeff,
        adv_norm_eps = args.adv_norm_eps,

        #TD(lambda)/GAE parameters
        gamma = args.gamma,
        td_lambda = args.td_lambda,

        # Ordered as (hidden_layer_features, num_hidden_layers)
        actor_network_details = (args.actor_hidden_features, args.actor_hidden_layers),
        critic_network_details = (args.critic_hidden_features, args.critic_hidden_layers),

        target_network_update = args.target_network_update
    )
  buffer_config = BufferConfig(buffer_size = args.buffer_size,
                                sampling_epsilon = args.sampling_epsilon,
                                replay_ratio = args.replay_ratio,
                                smoothing_window = args.smoothing_window,
                                log_returns = args.log_returns,
                                return_log_frequency = args.return_log_frequency)
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
  template_model = SimMMD(game, config, opt_config, buffer_config, seeds[0], batch_size=args.batch_size)
  for seed in seeds:
    train_loop(args, seed, template_model)
