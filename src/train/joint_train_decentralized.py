import os

from shutil import rmtree

from nash_dreamer.dreamer_ma import DreamerMAConfig, LATEST_STEP_FILENAME
from nash_dreamer.dreamer_ma_decentralized import DecentralizedDreamerMA
from envs.jax_game import JaxGame
from nash_dreamer.train_utils import *


def train_nash_dreamer_decentralized(args, game: JaxGame):
  """Perform the decentralized NashDreamer training on a particular game.

  Identical configuration to train_nash_dreamer, but builds a
  DecentralizedDreamerMA: one RSSM per player, no latent-infoset networks, a
  per-player critic, and independent latent sampling during imagination.

  Args:
      args (_type_): Argument specification. Detailed description of arguments can be found in parsing_utils.py
      game (JaxGame): The game to train on
  """
  assert args.train_mode == "rnad", (
    "Only the RNaD actor-critic is implemented for the decentralized variant, got "
    f"'{args.train_mode}'. Use the 'rnad' positional after nash_dreamer_decentralized.")
  assert args.use_original_infoset, (
    "The decentralized variant is only defined for real infosets, since it has no "
    "latent-infoset networks to learn one. Pass --use_original_infoset.")
  assert game.information_state_tensor_shape() == game.observation_tensor_shape(), (
    f"--use_original_infoset requires the game to provide its infoset in place of the "
    f"observation, but {game.game_name()} declares infoset shape "
    f"{game.information_state_tensor_shape()} and observation shape "
    f"{game.observation_tensor_shape()}.")

  seeds = parse_sequence(args.seeds)
  #Create the initial model. All the other
  # models will only change the model
  # state to prevent retracing
  wm_config = DreamerMAConfig(
      batch_size=args.batch_size,
      report_gradnorms = args.report_gradnorms,

      use_original_infoset = args.use_original_infoset,

      #Weights of the individual loss terms of the world model.
      #beta_infoset is accepted but unused: the decentralized model has no
      #latent-infoset networks and therefore no infoset loss terms.
      beta_prediction = args.beta_prediction,
      beta_dynamics = args.beta_dynamics,
      beta_representation = args.beta_representation,
      beta_infoset = args.beta_infoset,

      free_bits_clip_threshold = args.free_bits_threshold,
      uniform_mix = args.uniform_mix,

      encoded_classes = args.encoded_classes,
      encoded_categories = args.encoded_categories,
      bin_range = args.wm_bin_range,

      jsd=args.jsd,
      max_divergence_scaling = args.max_divergence_scaling,

      # Ordered as (hidden_layer_features, num_hidden_layers)
      sequential_network_details = (args.recurrent_state_size, args.sequential_mlp_features, args.sequential_mlp_layers),
      encoder_network_details = (args.encoder_tokens, args.encoder_hidden_features, args.encoder_hidden_layers),
      observer_network_details = (args.observer_hidden_features, args.observer_hidden_layers),
      decoder_network_details = (args.decoder_hidden_features, args.decoder_hidden_layers),
      dynamics_network_details = (args.dynamics_hidden_features, args.dynamics_hidden_layers),
      reward_predictor_network_details = (args.reward_predictor_hidden_features, args.reward_predictor_hidden_layers),
      done_predictor_network_details = (args.done_predictor_hidden_features, args.done_predictor_hidden_layers),
      legal_actions_network_details = (args.legal_predictor_hidden_features, args.legal_predictor_hidden_layers),
      #Unused by the decentralized model, kept so the config stays the shared dataclass.
      infoset_network_details = (args.latent_infoset_size, args.infoset_network_hidden_features, args.infoset_network_hidden_layers),
      infoset_decoder_details = (args.infoset_decoder_hidden_features, args.infoset_decoder_hidden_layers),
      infoset_predictor_details = (args.infoset_predictor_hidden_features, args.infoset_predictor_hidden_layers),
      obs_loss_bce = not args.obs_loss_l2
    )

  buffer_config = BufferConfig(buffer_size = args.buffer_size,
                                sampling_epsilon = args.real_sampling_epsilon,
                                replay_ratio = args.replay_ratio,
                                smoothing_window = args.smoothing_window,
                                log_returns = args.log_returns,
                                return_log_frequency = args.return_log_frequency,)
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
  ac_config = RNaDConfig(
      bin_range = args.ac_bin_range,
      train_real_policy = args.train_real_policy,
      report_gradnorms = args.report_gradnorms,

      beta_imagination = args.beta_imagination,
      beta_real = args.beta_real,

      eta=args.eta,
      sampling_epsilon=args.img_sampling_epsilon,
      cf_is_clip=args.cf_is_clip,

      num_starts = args.num_starts,

      #World model extraction parameters
      state_sample_threshold=args.state_sample_threshold,
      terminal_threshold = args.terminal_threshold,
      legal_threshold = args.legal_threshold,
      wm_warm_up_period = args.wm_warm_up,

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
  model = DecentralizedDreamerMA(wm_config, buffer_config, ac_config, opt_config, game, seeds[0])
  for seed in seeds:
    train_loop_decentralized(args, seed, model)


@track
def train_loop_decentralized(args, seed: int, template_model: DecentralizedDreamerMA):
  """Run the actual training loop.

  A copy of joint_train.train_loop rather than a call to it: that function
  hardcodes the DreamerMA constructor when building a fresh model, so a subclass
  instance would be silently rebuilt as the centralized base class. It also
  names the checkpoint directory after the centralized variant, which would put
  decentralized runs on top of centralized ones.

  Args:
      args (_type_): Argument specification. Detailed description of arguments can be found in parsing_utils.py
      seed (int): The PRNG seed for this training instance
      template_model (DecentralizedDreamerMA): A precreated template model, that has the same
      configuration as all the models during the training, except seed.
  """
  print(f"Running the decentralized training for seed {seed}")
  game = template_model.game
  model_root_dir = args.model_save_dir
  if not model_root_dir:
    #Distinct from "nash_dreamer_{train_mode}" so decentralized runs never land
    # in the centralized run's checkpoint directory.
    model_root_dir = f"nash_dreamer_decentralized_{args.train_mode}"

  model_save_dir = f"/trained_networks/{model_root_dir}/{game.to_compact_str()}/seed_{seed}/"
  model_save_dir = os.getcwd() + model_save_dir
  saved_model_file = ""
  if args.clean_dir:
    if args.continue_train:
      print(f"Warning! clean_dir and continue_train flags were supplied together. clean_dir is taking precedence.")
    try:
      rmtree(model_save_dir)
    except Exception as e:
      print(f"Removing a directory {model_save_dir} failed with exception {e}.")
  if args.continue_train and not args.clean_dir:
    latest_step_file = model_save_dir + LATEST_STEP_FILENAME
    try:
      with open(latest_step_file, 'r') as f:
        latest_step_suffix = f.readline()
      saved_model_file = model_save_dir + latest_step_suffix
    except FileNotFoundError as e:
      print(f"File {latest_step_file} was not found. Creating a clean model.")

  if saved_model_file:
    print(f"Loading model from path {saved_model_file}")
    model = load_model(saved_model_file)
    assert isinstance(model, DecentralizedDreamerMA), f"The loaded model should be a DecentralizedDreamerMA instance, not {model.__class__}"
    assert seed == model.init_seed, f"The given seed {seed} and the initial seed of the stored model {model.init_seed} do not match."
  else:
    print("Creating clean model")
    model = DecentralizedDreamerMA(template_model.wm_config, template_model.buffer_config,
                                   template_model.ac_config, template_model.opt_config, game, seed)
  #Will still retrace the nnx networks.
  # We have to do this, as the seed affects
  # their initialization as well.
  template_model.__setstate__(model.__getstate__())
  template_model.train_model(model_save_dir, args.num_steps, args.print_each, args.save_each, args.save_first)
