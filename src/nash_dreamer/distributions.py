"""Define the distribution utilities used by the various models.
Heavily inspired by https://github.com/symoon11/dreamerv3-flax"""

import jax
import jax.numpy as jnp
from nash_dreamer.train_utils import symlog
import chex

def add_uniform_mix(logits: chex.Array, uniform_mix: float = 0.01):
  """Creates a mixture between the actual logits induced distribution
  and uniform distribution, to prevent KL losses spike early
  as described in https://arxiv.org/pdf/2301.04104 page 5. """
  
  probs = jax.nn.softmax(logits, axis=-1)
  uniform = jnp.ones_like(probs) / probs.shape[-1]
  # Mix the probability with the uniform distribution.
  probs = (1.0 - uniform_mix) * probs + uniform_mix * uniform
  logits_with_uniform = jnp.log(probs)
  return logits_with_uniform

def sample_categorical(logits: chex.Array, key, sample_threshold: float = 0.0)-> chex.Array:
  """Given a PRNG key produced by split, sample from each
  of the categorical distributions logits and return the
  one-hot encoded outcome for each of the distributions.
  This function does NOT split internally,
  make sure the key passed to it is not reused.
  Sample threshold ensures that outcomes with probability lower than 
   this threshold are ignored (with the exception of if that would cause
   an categorical to have no valid outcomes). """
  # Calculate the logits-induced probability.
  starting_probs = jax.nn.softmax(logits, axis=-1)
  max_probs = jnp.max(starting_probs, axis=-1)
  #Make sure the thresholding does not make 
  # any categorical have no valid outcomes
  threshold = jnp.minimum(sample_threshold, jnp.min(max_probs))
  #perform the thresholding
  probs = starting_probs * (starting_probs >= threshold)
  normalization = jnp.sum(probs, axis=-1, keepdims=True)
  #TODO: The normalization == 0 is probably not necessary
  probs = probs / (normalization + (normalization == 0))
  #renormalize
  # Recalculate the logits
  thresholded_logits = jnp.log(probs)
  num_classes = logits.shape[-1]
  sampled_classes = jax.random.categorical(key, thresholded_logits, axis=-1)
  oh_sampled_classes = jax.nn.one_hot(sampled_classes, num_classes, axis=-1)
  #Perform the STE
  output = jax.lax.stop_gradient(oh_sampled_classes) + (probs - jax.lax.stop_gradient(probs))
  return output

"""Joint latent codes.

The prior is normally `encoded_classes` INDEPENDENT categoricals, which cannot represent a
dependency between them -- and the dynamics loss cannot see one either, since kl_divergence
sums over the class axis. With --joint_prior the prior instead emits a single distribution over
all `encoded_categories ** encoded_classes` joint codes. These four helpers move between the
two representations. They are only tractable while that product is small; a large factorization
needs an autoregressive prior instead.

The index bijection throughout is the base-`encoded_categories` number whose digits are the per
class outcomes, CLASS 0 MOST SIGNIFICANT: (0,0), (0,1), ... (0,K-1), (1,0), ... It matches
world_model_experiments.chance_code_usage.combo_index.
"""

def factored_to_joint(probs: chex.Array, encoded_classes: int) -> chex.Array:
  """[..., classes, categories] independent categoricals -> [..., categories ** classes] joint.

  Used to express the factored POSTERIOR as a joint so it can be compared against a joint prior
  in one KL. encoded_classes is the static config value, so this loop unrolls at trace time."""
  joint = probs[..., 0, :]
  for c in range(1, encoded_classes):
    #[..., categories ** c, categories] -> [..., categories ** (c + 1)]
    expanded = joint[..., :, None] * probs[..., c, None, :]
    joint = expanded.reshape(*expanded.shape[:-2], -1)
  return joint

def joint_to_grid(joint_index: chex.Array, encoded_classes: int, encoded_categories: int) -> chex.Array:
  """Flat joint code -> [..., classes, categories] one-hot grid. Inverts factored_to_joint's
  index convention, so the grid keeps the shape every latent consumer already expects."""
  digits = jnp.stack(
      [(joint_index // (encoded_categories ** (encoded_classes - 1 - c))) % encoded_categories
       for c in range(encoded_classes)], axis=-1)
  return jax.nn.one_hot(digits, encoded_categories, axis=-1)

def joint_marginals(joint_probs: chex.Array, encoded_classes: int, encoded_categories: int) -> chex.Array:
  """[..., categories ** classes] joint -> [..., classes, categories] per class marginals.

  DIAGNOSTICS AND GRADIENTS ONLY. Do not sample from these: drawing each class independently
  from its own marginal rebuilds the outer product, which is exactly the independence a joint
  prior exists to remove. Use sample_joint_categorical."""
  grid = joint_probs.reshape(*joint_probs.shape[:-1], *((encoded_categories,) * encoded_classes))
  leading = grid.ndim - encoded_classes
  return jnp.stack(
      [jnp.sum(grid, axis=tuple(leading + i for i in range(encoded_classes) if i != c))
       for c in range(encoded_classes)], axis=-2)

def sample_joint_categorical(logits: chex.Array, key, encoded_classes: int,
                             encoded_categories: int, sample_threshold: float = 0.0) -> chex.Array:
  """Sample ONE joint latent code and return it as a [..., classes, categories] one-hot grid.

  The joint counterpart of sample_categorical. Sampling jointly is the entire point: the classes
  are drawn together, so a code combination the prior has learned is impossible is never
  produced.

  sample_threshold keeps the meaning it has in the factored case -- "ignore outcomes this far
  below a typical one" -- by rescaling with the ratio of the two uniform masses,
  (1/K**C) / (1/K). Passing a caller's per class threshold straight through would compare it
  against 1/K**C instead of 1/K, i.e. be wrong by a factor of K**(C-1): with the default 0.05
  and a (2, 6) latent that is the difference between pruning nothing and pruning all but the
  single most likely code."""
  joint_threshold = sample_threshold / (encoded_categories ** (encoded_classes - 1))
  sampled_joint = sample_categorical(logits, key, joint_threshold)
  grid = joint_to_grid(jnp.argmax(sampled_joint, axis=-1), encoded_classes, encoded_categories)
  #Straight through in the same shape as sample_categorical's: forward is the sampled grid, the
  # gradient is the joint's per class marginal. No caller differentiates through the prior today
  # (every imagination rollout is stop_gradient'd by its caller), but a silent zero here would be
  # a trap for anything that later does.
  marginals = joint_marginals(jax.nn.softmax(logits, axis=-1), encoded_classes, encoded_categories)
  return jax.lax.stop_gradient(grid) + (marginals - jax.lax.stop_gradient(marginals))

def get_normal_log_prob(mean_logits: chex.Array, value: chex.Array, use_symlog=False) ->chex.Array:
  """Get log prob of the normal distributions represented by the predictor outputs.
  Since the predictors output logits for mean and variance is assumed to be one, 
  the log prob reduces to -MSE. Can use the symlog transformation from https://arxiv.org/pdf/2301.04104
  page 7 to improve robustness. If using it, make sure to then convert the network prediction
  with the inverse symexp for inference."""
  chex.assert_equal_shape([mean_logits, value]) 
  value = jnp.where(use_symlog, symlog(value), value)
  log_prob = -(value - mean_logits) **2
  return log_prob

def get_bin_log_prob(dist_logits: chex.Array, bins: chex.Array,  value: chex.Array, use_symlog = True)->chex.Array:
  """Get log prob of the discrete distribution corresponding to the exponentially spaced bins.
  From https://arxiv.org/pdf/2301.04104  page 7. First two hot encodes value, and the 
  final log prob is twohot(value) * logsoftmax(dist_logits). 
  Expects dist_logits to be of shape [Trajectory, batch, ..., 2*bin_range + 1],
  value to be of shape [Trajectory, batch, ..., 1] or [Trajectory, batch, ....] and
  bins of shape [2 * bin_range + 1]"""
  value = value.reshape(value.shape + (1, ) * (dist_logits.ndim - value.ndim))
  chex.assert_equal_shape_suffix([dist_logits, bins], 1) # The last dimension of bins and dist_logits should match 
  chex.assert_equal_shape_prefix([dist_logits, value], -1) # All dimension except the last should match
  #[Trajectory, Batch, 2 * bin_range + 1]
  val_two_hot = two_hot_encode(bins, value, use_symlog=use_symlog)
  return jnp.sum(val_two_hot * jax.nn.log_softmax(dist_logits), axis=-1, keepdims=True)

def get_categorical_log_prob(dist_logits: chex.Array, oh_target: chex.Array):
  """Gets the probability of the one-hot encoded target under the categorical
  distribution parametrized by dist_logits as oh_target * logsoftmax(dist_logits)"""
  chex.assert_equal_shape((dist_logits, oh_target))
  return jnp.sum(oh_target * jax.nn.log_softmax(dist_logits), axis=-1, keepdims=True)
   

def two_hot_encode(bins: chex.Array, value:chex.Array, use_symlog= True) -> chex.Array:
  """Perform the two hot encoding of value (by default transformed by symlog)
  in the range of bins. There will be two nonzero values of the two closest bins, 
  with values proportional to the bin closeness."""
  value = jnp.where(use_symlog, symlog(value), value)
  promoted_bins = bins.reshape((1, ) * (value.ndim - 1) + bins.shape)
  below = value >= promoted_bins
  above = value <= promoted_bins
  #Making use of argmax/argmin returning the first occurence as a tie breaking strategy
  int_start_idx = jnp.where(jnp.sum(below, axis=-1) == bins.shape[0] - 1, bins.shape[0] - 1, jnp.maximum(jnp.argmin(below, axis=-1).astype(jnp.int32) - 1, 0))
  int_end_idx = jnp.argmax(above, axis= -1).astype(jnp.int32)

  equal = int_start_idx == int_end_idx
  equal_oh = jax.nn.one_hot(int_start_idx, bins.shape[0], axis=-1)
  
  #[Trajectory, Batch, 1]
  start_bins = jnp.take_along_axis(promoted_bins, int_start_idx[..., None], axis=-1)
  end_bins = jnp.take_along_axis(promoted_bins, int_end_idx[..., None], axis=-1)
  start_dist = jnp.abs(value - start_bins)
  end_dist = jnp.abs(end_bins - value) 
  total = start_dist + end_dist
  weight_start = start_dist / total
  weight_end = end_dist / total


  #Watch out!!! The end weight needs to go to the start and vice-versa.
  # The reason for that is because rather than the distance, we want the probability
  # that target belongs to a certain bin. Eg. if the target is 0.8, it is between
  # bins 0 and 1, with distance 0.8 from the start and 0.2 from end. But
  # this means that it belongs to bin 0 with pbt 0.2 and bin 1 with pbt 0.8
  start_oh = jax.nn.one_hot(int_start_idx, bins.shape[0], axis=-1) * weight_end
  end_oh = jax.nn.one_hot(int_end_idx, bins.shape[0], axis=-1) * weight_start
  
  #[Trajectory, Batch, 2 * bin_range + 1]
  two_hot = jnp.where(equal[..., None], equal_oh, start_oh + end_oh)

  return two_hot

def kl_divergence(orig: chex.Array, other: chex.Array) ->chex.Array:
  """Computes KL divergence. Expects both orig and other to already be softmaxed
  into probability distributions. Returning kl_divergence is summed over the last two dimensions
  [categoricals, classes]."""
  div_term = jnp.log(orig) - jnp.log(other)
  divergence_unmasked = jnp.sum(orig * div_term, axis=(-1, -2))
  return divergence_unmasked

def jsd(orig: chex.Array, other: chex.Array) -> chex.Array:
  """Computes the Jensen-Shannon divergence between two distributions. Expects both orig and other to already be softmaxed
  into probability distributions. Returning jsd is summed over the last two dimensions
  [categoricals, classes]."""
  mixture = (orig + other) * 0.5
  return (kl_divergence(orig, mixture) + kl_divergence(other, mixture))  * 0.5
   


   
   
