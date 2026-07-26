import chex
import jax
import numpy as np
import jax.numpy as jnp
import os
import pickle
import psutil
import resource
import time

from typing import Sequence, Tuple



LATEST_STEP_FILENAME = "latest.txt"


##################################################################
### MEMORY AND TIME USAGE DEBUGGING, taken from
### https://stackoverflow.com/questions/938733/total-memory-used-by-python-process
#################################################################
def elapsed_since(start):
    return time.strftime("%H:%M:%S", time.gmtime(time.time() - start))


def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss

def get_peak_memory():
    # resource.getrusage returns ru_maxrss in kilobytes on Linux systems. 
    # We multiply by 1024 to convert it to bytes to match psutil.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def track(func):
    def wrapper(*args, **kwargs):
        mem_before = get_process_memory()
        start = time.time()
        result = func(*args, **kwargs)
        elapsed_time = elapsed_since(start)
        mem_after = get_process_memory()
        peak_memory = get_peak_memory()
        print("{}: memory before: {:,}, after: {:,}, consumed: {:,}, peak: {:,}; exec time: {}".format(
            func.__name__,
            mem_before, mem_after, mem_after - mem_before, peak_memory,
            elapsed_time))
        return result
    return wrapper

#################################################################
### END OF CODE FROM https://stackoverflow.com/questions/938733/total-memory-used-by-python-process
#################################################################

  

@chex.dataclass(frozen=True)
class PredictionStepWithLegal():
  recurrent_state: chex.Array
  repr_state: chex.Array
  tokens: chex.Array  # encoder output, shape (..., encoder_tokens_features)
  deter_state: chex.Array
  decoded_obs: chex.Array
  reward_dist_logit: chex.Array
  done_logit: chex.Array
  legal_logit: chex.Array
  dynamics_state: chex.Array
  #Latent infoset predictions and decoding
  joint_latent_infoset: chex.Array
  infoset_decoded_obs: chex.Array
  infoset_decoded_actions: chex.Array
  infoset_predicted_recurrent: chex.Array
  infoset_predicted_deter: chex.Array


@chex.dataclass(frozen=True)
class ActorCriticTimeStep():
  
  obs: chex.Array = () # [..., Player, infoset_dim] for IIGs 
                      #or [..., Player, num_classes * num_categoricals + hidden_state_size otherwise + num_players]
                      # For each player, a one hot encoding of the player index is also appended
                      # to the model state. Otherwise it would be impossible to return distinct policies 
                      # for multiple players playing at the given state.
  legal: chex.Array = () # [..., Player, A] Legal actions in the given state
  
  action: chex.Array = () # [..., Player, A] action sampled at the given state
  policy: chex.Array = () # [..., Player, A] =policy at the given state
  
  reward: chex.Array = () # [...] Reward after playing an action
  valid: chex.Array = () # [...] Flag determining, whether we should train in this state

@chex.dataclass(frozen=True)
class TimeStep():
  
  obs: chex.Array = () # [..., Player, obs_dim]
  negative_obs: chex.Array = () # [..., num_negatives, num_players * obs_features]

  legal: chex.Array = () # [..., Player, A] for multi agent or [..., A] for single_agent
  
  action: chex.Array = () # [..., Player, A] for multi agent or [..., A] for single agent
  policy: chex.Array = () # [..., Player, A] for multi agent or [..., A] for single agent
  
  reward: chex.Array = () # [...] Reward for reaching a state
  valid: chex.Array = () # [...] Flag determining, whether we should train in this state
  terminal: chex.Array = () #[...] Flag determining whether the state is terminal




@chex.dataclass(frozen=True)
class BufferConfig:
  buffer_size: int
  sampling_epsilon: float #Uniform policy mixture in the sampling policy
  replay_ratio: int = -1 #How many steps should be collected from the replay buffer per
                          # online collected env step

  #For logging returns
  return_log_frequency: int = 10
  smoothing_window: int = 50
  log_returns: bool = False
  
  number_of_negatives: int = 5 #Number of negative samples to use for the contrastive loss. If 0, no contrastive loss is used.

@chex.dataclass(frozen=True)
class RNaDConfig:


  train_real_policy: bool = False # Whether to train policy also on real trajectories
  report_gradnorms: bool = False #Whether to report gradient norms.

  beta_imagination: float = 1.0
  beta_real: float = 0.3 # Coeficients for the loss parts. Beta imagination is used for Dreamer
                          # unrolled trajectories and beta real for trajectories from the real environment
                          # Used only for joint training.
  
  #Ordered as hidden layer size, num hidden layers
  actor_network_details: Tuple[int, int] = (256, 1)
  critic_network_details: Tuple[int, int] = (256, 1)
  
  entropy_schedule_repeats: Sequence[int] = (1,)
  entropy_schedule_size: Sequence[int] = (1000,)

  eta: float = 0.2 #Regularization strenght

  #Clipping parameters for the counterfactual importance sampling correction.
  cf_is_clip: float = 100.0


  #V-trace parameters
  rho_vtrace: float = 1.0 # Clipping parameter. Affects to which policy estimate V-trace converges. Inf means convergence to the estimate for the learned policy
  c_vtrace: float = 1.0 # Clipping parameter
  gamma_vtrace: float = 1.0 # Discount factor
  lambda_vtrace: float = 1.0 #Same as TD-learning lambda
  
  num_starts: int = -1 #How many starting points to take from each trajectory for the imagination unroll. IF -1 take the same amount as the length of the trajectory.

  #NeuRD parameters
  neurd_clip: float = 10000
  neurd_threshold: float = 2.0


  sampling_epsilon: float = 0.0
  state_sample_threshold: float = 0.05 #A threshold when sampling states. The outcomes for
                                        #each categorical below this threshold are ignored (or, specificaly a minimum
                                        # of this threshold and the lowest of max probability outcomes of the categoricals). 
  terminal_threshold:float =  0.5 #Thresholds when to consider the state terminal, or the actions
  legal_threshold: float = 0.5    # Legal, when we take the sigmoid over the Dreamer produced logits.
  bin_range: int = 20 #Number of the exponentially spaced bins for the value categorical distribution prediction
  wm_warm_up_period: int = 1000 #How many steps to let the world model "warm-up" and only train on real trajectories, before starting to imagine.
  
  target_network_update: float = 1e-3


@chex.dataclass(frozen=True)
class MMDConfig:
  """Configuration of the standalone Magnetic Mirror Descent learner.
  The magnet is fixed at the uniform policy, which makes the magnet term
  KL(pi || magnet) an entropy exploration bonus (up to a per state constant).
  The resulting algorithm is effectively PPO with an entropy bonus and
  the mirror descent proximal term KL(pi || pi_old) made explicit."""

  report_gradnorms: bool = False #Whether to report gradient norms.

  #MMD/PPO parameters
  num_epochs: int = 4 #Number of inner gradient steps taken on each collected on-policy batch
  clip_epsilon: float = 0.2 #The PPO clipping parameter for the policy ratio
  kl_coeff: float = 0.1 #Weight of the explicit proximal term KL(pi || pi_old)
  #Weight of the magnet term KL(pi || uniform), i.e. the entropy exploration bonus.
  # THIS IS THE PARAMETER TO SWEEP PER GAME. It sets the temperature of the quantal
  # response equilibrium that MMD converges to, and the best value depends on how
  # mixed the equilibrium of the game is.
  #  - A pure equilibrium wants a small magnet, because a uniform magnet pulls away
  #    from every pure strategy. Measured on goofspiel_3 over 1000 steps, nash_conv
  #    reached 0.0000 for every value in [0, 0.2], but 0.9534 at 1.0.
  #  - A mixed equilibrium wants a large magnet, because the regularization is what
  #    stops the policy gradient from cycling away from it. Measured on RPS, whose
  #    equilibrium is uniform, over 200 steps, nash_conv reached 0.105 at 1.0 but
  #    1.784 at 0.2 and 1.998 at 0.
  # Note that this lives on the scale of the advantage, which the standard PPO
  # normalization rescales to unit standard deviation. That keeps the policy gradient
  # force at O(1) even at an equilibrium, where the whole advantage is sampling noise,
  # while the magnet force is exactly zero at the uniform policy and only grows as the
  # policy drifts away from it. So do not carry the much smaller RNaD eta over here,
  # RNaD does not normalize its advantages.
  magnet_coeff: float = 0.05
  adv_norm_eps: float = 1e-8 #Numerical stability term for the advantage normalization

  #TD(lambda)/GAE parameters
  gamma: float = 1.0 # Discount factor
  td_lambda: float = 0.95 #Same as TD-learning lambda

  #Ordered as hidden layer size, num hidden layers
  actor_network_details: Tuple[int, int] = (256, 1)
  critic_network_details: Tuple[int, int] = (256, 1)

  bin_range: int = 20 #Number of the exponentially spaced bins for the value categorical distribution prediction

  target_network_update: float = 1e-3


@chex.dataclass(frozen=True)
class DreamerMAConfig():
  batch_size: int


  encoded_classes: int # Number of classes for each categorical distribution in state
  encoded_categories: int # Number of categorical distributions in state


  use_original_infoset: bool = False
  report_gradnorms: bool = False # Whether to report world model gradient norms

  #Weights of the individual loss terms of the world model
  beta_prediction: float = 1
  beta_dynamics: float = 1
  beta_representation: float = 0.1
  beta_infoset: float = 1.0
  beta_contrastive: float = 1.0

  #Distributional loss parameters
  jsd: bool =False #Whether to use JSD or KL for the prior/posterior distance
  max_divergence_scaling: bool = False #Whether to scale the prior/posterior distance
  
  #Contrastive loss parameters
  contrastive_temperature: float = 0.1 #Temperature parameter for the contrastive loss.

  free_bits_clip_threshold: float = 1 #Threshold for loss clip in free bits.
  uniform_mix: float = 0.01 # Amount of uniform mixture added to the 
                            # network produced categoricals. 
  
  bin_range: int = 20 #Number of the exponentially spaced bins for certain predictions such as reward in one direction, bins will be spaced out as symexp([-bin_range, ..., bin_range])
  
  sequential_network_details: tuple[int, int, int] = (256, 256, 1) # Ordered as size of hidden state, number of features for the MLP processing, number of layers in the MLP processing
  encoder_network_details: tuple[int, int, int] = (256, 256, 1) #Ordered as observation tokens size, hidden_layer_features, num_hidden_layers
  infoset_network_details: tuple[int, int, int] = (256, 256, 1) #Ordered as size of latent infoset, hidden_layer_features, num_hidden_layers
  # Ordered as (hidden_layer_features, num_hidden_layers)
  infoset_decoder_details: tuple[int, int] = (256, 1)
  infoset_predictor_details: tuple[int, int] = (256, 1)
  decoder_network_details: tuple[int, int] = (256, 1)
  observer_network_details: tuple[int, int] = (256, 1)
  dynamics_network_details: tuple[int, int] = (256, 1)
  reward_predictor_network_details: tuple[int, int] = (256, 1)
  done_predictor_network_details: tuple[int, int] = (256, 1)
  legal_actions_network_details: tuple[int, int] = (256, 1)

  obs_loss_bce: bool = True  # If True, use BCE loss for observation reconstruction; if False, use L2 (symlog targets)

@chex.dataclass(frozen=True)
class ActorCriticConfig():

  # The EMA coefficient for update
  # of the target network parameters
  # is 1 - this value
  target_network_update: float = 1e-3

  
  train_real_policy: bool = False # Whether to train policy also on real trajectories
  report_gradnorms: bool = False #Whether to report gradient norms.

  beta_imagination: float = 1.0
  beta_real: float = 0.3 # Coeficients for the loss parts. Beta imagination is used for Dreamer
                          # unrolled trajectories and beta real for trajectories from the real environment
  eta:float = 3e-4  #Coefficient for the entropy exploration bonus
  gamma: float = 0.997 # Gamma for TD-learning
  td_lambda: float = 0.95 # Lambda for TD-learning

  upper_percentile: float = 95
  lower_percentile: float = 5 #Percentiles for the return normalization range
  range_ema_coeff: float = 0.99 # Coeeficient for the EMA update of retun normalization range
  num_starts: int = -1 #How many starting points to take from each trajectory for the imagination unroll. IF -1 take the same amount as the length of the trajectory.

  #Ordered as hidden layer size, num hidden layers
  actor_network_details: Tuple[int, int] = (256, 1)
  critic_network_details: Tuple[int, int] = (256, 1)
  
  sampling_epsilon: float = 0.0
  state_sample_threshold: float = 0.05 #A threshold when sampling states. The outcomes for
                                        #each categorical below this threshold are ignored (or, specificaly a minimum
                                        # of this threshold and the lowest of max probability outcomes of the categoricals). 
  terminal_threshold:float =  0.5 #Thresholds when to consider the state terminal, or the actions
  legal_threshold: float = 0.5    # Legal, when we take the sigmoid over the Dreamer produced logits.
  bin_range: int = 20 #Number of the exponentially spaced bins for the value categorical distribution prediction
  wm_warm_up_period: int = 1000 #How many steps to let the world model "warm-up" and only train on real trajectories, before starting to imagine.


@chex.dataclass(frozen=True)
class OptimizerConfig():
  lr: float = 3e-4,
  agc: float = 0.3,
  eps: float = 1e-20,
  beta1: float = 0.9,
  beta2: float = 0.999,
  momentum: bool = True,
  nesterov: bool = False,
  schedule: str = 'const',
  warmup: int = 1000,
  anneal: int = 0,



def symlog(x: chex.Array):
  return jnp.sign(x) * jnp.log(jnp.abs(x) + 1)

def symexp(x: chex.Array):
  return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1)

def legal_policy(logit: chex.Array, legal: chex.Array):
  """Get a softmaxed policy out of logit, with
  zeros at illegal actions. Assumes that these have actions in the last
  dimension and the same shape."""
  chex.assert_equal_shape([logit, legal])
  shifted_logit = logit - logit.max(axis=-1, keepdims=True)
  exp_logit = jnp.exp(shifted_logit)
  #The only way this can potentially break is 
  # if +- inf or NaN appears already in the exp_logit
  # at which point it is an error in the network
  masked_exp_logit = exp_logit * legal
  normalization = jnp.sum(masked_exp_logit, axis=-1, keepdims=True)
  policy = masked_exp_logit / (normalization + (normalization == 0))
  return policy

def legal_log_policy(logit: chex.Array, legal: chex.Array):
  """Uses a legal_policy to get the masked policy
  and then return a log of it, with the exception
  of illegal actions which have 0 instead of -inf. 
  Assumes that these have actions in the last
  dimension and the same shape."""
  chex.assert_equal_shape([logit, legal])
  shifted_logit = logit - logit.max(axis=-1, keepdims=True)
  exp_logit = jnp.exp(shifted_logit)
  masked_exp_logit = exp_logit * legal
  
  normalization = jnp.sum(masked_exp_logit, axis=-1, keepdims=True)
  log_policy = shifted_logit - jnp.log(normalization + (normalization == 0))
  legal_log_policy = jnp.where(legal > 0, log_policy, 0.0)
  return legal_log_policy

def policy_ratio(pi: chex.Array, mu: chex.Array, actions_oh: chex.Array, valid: chex.Array) -> chex.Array: 
  pi_actions_prob = jnp.sum(pi * actions_oh, axis=-1, keepdims=True) * valid + (1 - valid)
  mu_actions_prob = jnp.sum(mu * actions_oh, axis=-1, keepdims=True) * valid + (1 - valid)
  
  return pi_actions_prob / mu_actions_prob
  
def tree_where(pred: chex.Array, x: chex.ArrayTree, y: chex.ArrayTree) -> chex.ArrayTree:
  
  def _where(x, y):
    return jnp.where(pred, x, y).astype(x.dtype)
  
  return jax.tree.map(_where, x, y)

def tree_index(x: chex.ArrayTree, indices: chex.Array, axis=-1) -> chex.ArrayTree:
  """Selects indices specified by a one dimensional index array
  in all elements of the tree along the specified axis. If indices
  are not one-dimensional they are flattened first."""
  indices = indices.ravel()
  def _select(x):
    expanded_indices_shape = [1, ] * x.ndim
    expanded_indices_shape[axis] = indices.shape[0]
    item_indices = jnp.reshape(indices, expanded_indices_shape)
    return jnp.take_along_axis(x, item_indices, axis=axis)
  return jax.tree.map(_select, x)
  
def apply_force_with_threshold(decision_outputs: chex.Array, force: chex.Array,
                               threshold: float,
                               threshold_center: chex.Array) -> chex.Array:
  """Apply the force with below a given threshold."""
  chex.assert_equal_shape((decision_outputs, force, threshold_center))
  can_decrease = decision_outputs - threshold_center > -threshold
  can_increase = decision_outputs - threshold_center < threshold
  force_negative = jnp.minimum(force, 0.0)
  force_positive = jnp.maximum(force, 0.0)
  clipped_force = can_decrease * force_negative + can_increase * force_positive
  return decision_outputs * jax.lax.stop_gradient(clipped_force)


def get_loss_mean_with_mask(loss: chex.Array, mask: chex.Array, normalization_mult=1) -> chex.Array:
  """Mask a loss using mask and compute its mean, 
  such that elements with 0 in the mask are correctly ignored.
    Ensure loss and mask are of broadcastable shape. """
  normalization_factor = jnp.sum(mask) * normalization_mult
  broadcasted_mask = jnp.zeros(loss.shape, mask.dtype) + mask
  masked_loss = loss * broadcasted_mask
  summed_loss = jnp.sum(masked_loss)
  return summed_loss / (normalization_factor + (normalization_factor == 0))


def get_percentiles_with_mask(data: chex.Array, mask:chex.Array, percentile: chex.Array):
  """Get the given percentile of the data,
  while ignoring the data given by mask. The percentile
  should be a number (or array like of numbers) in range (0, 100).
  The logic is that the percentile should only ensure that the given
  percent of the unmasked data are under the returned value.
  Ensure that data and mask are of broadcastable shape.
     """
  broadcasted_mask = jnp.zeros(data.shape, mask.dtype) + mask
  max_val = jnp.max(data)
  #mask out the invalid values with the maximum value
  #that way we ensure they will be at the end after sorting
  masked_data = broadcasted_mask * data + (1 - broadcasted_mask) * max_val

  num_valid = jnp.sum(broadcasted_mask)
  #Rescale the percentile. To accurately
  # reflect we do not care about the invalid data at the end
  scale_factor = (num_valid - 1) / (data.size - 1)
  new_percentile = scale_factor * percentile
  return jnp.percentile(masked_data, new_percentile)


def get_value_from_bins(dist_logits: chex.Array, bin_range: int, use_symexp=True):
  """Reads out the prediction from the predicted
  logits of the categorical distribution, by multiplying it with the bins."""
  #Implementing the summation order suggestion
  # from https://arxiv.org/pdf/2301.04104 page 18
  bins = jnp.arange((2 * bin_range) + 1) - bin_range
  bins = bins.reshape((1,) * (dist_logits.ndim - 1) + bins.shape)
  v_probs = jax.nn.softmax(dist_logits)
  pos_bins = bins * (bins >= 0)
  # flip the probs and bins for the negative
  # to ensure summation from small to large in magnitude 
  neg_bins = bins * (bins < 0)
  v_pos_part = jnp.sum(v_probs * pos_bins, axis=-1, keepdims=True)
  v_neg_part = jnp.sum(jnp.flip(v_probs * neg_bins), axis=-1, keepdims=True)
  v = v_pos_part + v_neg_part
  if use_symexp:
    v = symexp(v)
  return v

def wm_timestep_to_timestep(wm_timestep: TimeStep, wm_prediction_step: PredictionStepWithLegal, use_infoset: bool) ->ActorCriticTimeStep:
    #Do not forget that the world model timestep rewards and terminal
    # are w.r.t. the current state. We want
    # reward for playing an action in the current state, not for getting to it
    # so, they are shifted by 1 forward in time
    # compared to our desired actor-critic timesteps.
    # We also do not want the last step, since that is always a terminal state
    legal = wm_timestep.legal[:-1]
    action = wm_timestep.action[:-1]
    policy = wm_timestep.policy[:-1]
    reward = wm_timestep.reward[1:]
    valid = jnp.logical_and(~wm_timestep.terminal[:-1], wm_timestep.valid[:-1])
    
    #if use_infoset is specified, we assume that
    # obs is our infoset and that we want to use it
    # Otherwise, use the latent_infoset
    if use_infoset:
      obs = wm_timestep.obs[:-1]
    else:
      obs = wm_prediction_step.joint_latent_infoset[:-1]
    ac_timestep = ActorCriticTimeStep(obs = obs,
                                      legal=legal,
                                      action=action,
                                      policy = policy,
                                      reward = reward,
                                      valid=valid)
    return ac_timestep
  
def get_reference_policy(obs: chex.Array, legal_actions: chex.Array):
  """Returns the reference sampling policy. For now returns just a uniform policy.
  TODO: This is just for the basic testing, change this function"""
  return legal_actions / legal_actions.sum(axis=-1, keepdims=True)

def uniform_policy(infoset: chex.Array, legal_actions: chex.Array):
  return legal_actions / legal_actions.sum(axis=-1, keepdims=True)


def save_model(model, path): 
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "wb") as f:
    pickle.dump(model, f)
    
def load_model(path:str):
  path = path.strip("\n")
  with open(path, "rb") as f:
    return pickle.load(f)
  
  
def parse_sequence(str_spec: str) -> list[int]:
    """A helper utility to parse a list of integers from the string specification

    Args:
        str_spec (str): A string specification of the sequence to be used 
                        of form (num_1, num_2, num_N) or a single number "num"

    Returns:
        list[int]: A list of the integer parsed sequence.
    """
    str_spec = str_spec.strip()
    
    # Relax the assertion to allow either the tuple format OR a standalone number
    is_tuple = str_spec.startswith('(') and str_spec.endswith(')')
    is_single_num = str_spec.replace('-', '', 1).isdecimal()
    
    assert is_tuple or is_single_num, f"Invalid string specification: {str_spec}"
    
    # strip('()') handles both "(42, 99)" and "42" flawlessly
    str_tokens = str_spec.strip('()').split(',')
    
    seq = []
    for s in str_tokens:
        s_clean = s.strip()
        
        # The replace check ensures negative numbers bypass the isdecimal() 
        # rejection so they can trigger your random seed logic below
        if not s_clean.replace('-', '', 1).isdecimal():
            continue
            
        num = int(s_clean)
        if num < 0:
            num = np.random.randint(0, 2**32 - 1)
        seq.append(num)
        
    assert len(seq) > 0, f"No valid integer was found in the provided specification: {str_spec}"
    return seq
  