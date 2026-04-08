import jax.numpy as jnp
from train_utils import RNaDConfig, OptimizerConfig, BufferConfig, parse_sequence
from experiments.joint_train import train_loop, SimRNaD, JaxGame


def train_rnad(args, game:JaxGame):
  seeds = parse_sequence(args.seeds)
  config = RNaDConfig(
        bin_range = args.bin_range,
        report_gradnorms = args.report_gradnorms,

        eta=args.eta,
        sampling_epsilon=args.sampling_epsilon,

        # Entropy schedule parameters
        entropy_schedule_size = parse_sequence(args.entropy_schedule_size),
        entropy_schedule_repeats = parse_sequence(args.entropy_schedule_repeats),
        
        #V-Trace parameters
        rho_vtrace = args.rho_vtrace if args.rho_vtrace >= 0 else jnp.inf,
        c_vtrace = args.c_vtrace if args.c_vtrace >= 0 else jnp.inf,
        gamma_vtrace = args.gamma_vtrace,
        lambda_vtrace = args.lambda_vtrace,

        # NeuRD parameters
        neurd_clip = args.neurd_clip,
        neurd_threshold = args.neurd_threshold,

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
  template_model = SimRNaD(game, config, opt_config, buffer_config, seeds[0], batch_size=args.batch_size)
  for seed in seeds:
    train_loop(args, seed, template_model)