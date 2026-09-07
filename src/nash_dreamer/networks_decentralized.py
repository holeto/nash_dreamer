"""Per-player (decentralized) variants of the centralized world-model networks.

Each class here is its centralized counterpart in `networks.py` with the
`num_players` factor removed from exactly one layer, so that the module sees and
produces one player's slice only.  The intended use is to instantiate a single
module and `nnx.vmap` it over a leading player axis: parameters are shared, the
inputs are not.  Player specialization is still possible because every game's
infoset tensor is prefixed with a player-id one-hot.

Networks that are already player-agnostic are NOT duplicated here -- import
`LinNormRelu`, `HiddenMLP`, `ObservedPredictor`, `DynamicsPredictor`,
`RewardPredictor`, `DonePredictor`, `ActorNetwork` and `CriticNetwork` straight
from `networks.py`.
"""

import chex
from flax import nnx
import jax.numpy as jnp

from nash_dreamer.networks import LinNormRelu, HiddenMLP


class DecSequenceModel(nnx.Module):
  '''
    Decentralized counterpart of networks.SequenceModel.
    Produces the next recurrent state of a SINGLE player from that player's own
    recurrent state, stochastic state and OWN action.  Because the opponent's
    action is not observed, the transition it has to model is not identifiable
    -- that is the intended source of instability, not an oversight.
  '''
  def __init__(self, encoded_classes, encoded_categories, action_features,
               hidden_features: int, linear_hidden_layers: int,
               recurrent_state_size, rngs: nnx.Rngs):
    self.hidden_init = LinNormRelu(recurrent_state_size, hidden_features, rngs=rngs)
    self.action_init = LinNormRelu(action_features, hidden_features, rngs=rngs)
    self.deter_init = LinNormRelu(encoded_classes * encoded_categories, hidden_features, rngs=rngs)
    # Concatenation of the actual recurrent state, with
    # the embeddings of the recurrent state, action and stochastic state
    core_input_size = recurrent_state_size + 3 * hidden_features
    self.core_mlp = HiddenMLP(core_input_size, num_layers=linear_hidden_layers, rngs=rngs)
    #This is the projection to the reset, cand and update gates
    self.gate_head = nnx.Linear(core_input_size, 3 * recurrent_state_size, rngs=rngs)

  def __call__(self, recurrent_state: chex.Array, deter_state: chex.Array, action: chex.Array):
    """Ensure that actions are already one hot encoded.
    Deter state is the already sampled state out of stochastic state.
    Action has shape (..., action_features) -- one player's own action."""
    flat_deter = jnp.reshape(deter_state, (*deter_state.shape[:-2], -1))
    x0 = self.hidden_init(recurrent_state)
    x1 = self.deter_init(flat_deter)
    x2 = self.action_init(action)
    x = jnp.concatenate([recurrent_state, x0, x1, x2], axis=-1)
    x = self.core_mlp(x)
    gates = self.gate_head(x)
    reset, cand, update = jnp.split(gates, 3, axis=-1)
    reset = nnx.sigmoid(reset)
    cand = nnx.tanh(reset * cand)
    #The -1 makes the update naturally smaller
    # making the network more biased towards
    # keeping the old recurrent_state
    update = nnx.sigmoid(update - 1)
    new_recurrent_state = update * cand + (1 - update) * recurrent_state

    return new_recurrent_state


class DecEncoder(nnx.Module):
  """Decentralized counterpart of networks.Encoder.
  Receives one player's own observation together with that player's recurrent
  state, and returns the latent feature vector used to produce its posterior
  stochastic state logits."""
  def __init__(self, observation_features, recurrent_state_size, tokens_features,
               hidden_features, num_layers, rngs: nnx.Rngs) -> None:
    self.tokens_features = tokens_features
    self.init_layer = LinNormRelu(recurrent_state_size + observation_features, hidden_features, rngs)
    self.core_mlp = HiddenMLP(hidden_features, num_layers, rngs)
    self.last_layer = nnx.Linear(hidden_features, tokens_features, rngs=rngs)

  def __call__(self, recurrent_state: chex.Array, observation: chex.Array):
    x = jnp.concatenate([recurrent_state, observation], axis=-1)
    x = self.init_layer(x)
    x = self.core_mlp(x)
    tokens = self.last_layer(x)
    return tokens


class DecDecoder(nnx.Module):
  """Decentralized counterpart of networks.Decoder.
  Reconstructs only the OWN observation of the player whose latent state is
  passed in, as a tensor of shape (..., observation_features)."""
  def __init__(self, recurrent_state_size, observation_features, encoded_classes,
               encoded_categories, hidden_features, num_layers, rngs: nnx.Rngs) -> None:
    self.observation_features = observation_features
    self.init_layer = LinNormRelu(recurrent_state_size + encoded_classes * encoded_categories, hidden_features, rngs)
    self.core_mlp = HiddenMLP(hidden_features, num_layers, rngs)
    self.last_layer = nnx.Linear(hidden_features, observation_features, rngs=rngs)

  def __call__(self, recurrent_state: chex.Array, encoded_state: chex.Array):
    x = jnp.concatenate([recurrent_state, encoded_state.reshape(*encoded_state.shape[:-2], -1)], axis=-1)
    x = self.init_layer(x)
    x = self.core_mlp(x)
    return self.last_layer(x)


class DecLegalActionsNetwork(nnx.Module):
  """Decentralized counterpart of networks.LegalActionsNetwork.
  Predicts only the acting player's own legal action logits, which sidesteps the
  soundness caveat the centralized version carries: nothing about the opponent's
  legal actions is inferred from a shared latent."""
  def __init__(self, action_dimension, encoded_classes, encoded_categories,
               recurrent_state_size, hidden_features, num_layers, rngs: nnx.Rngs) -> None:
    self.init_layer = LinNormRelu((recurrent_state_size + (encoded_classes * encoded_categories)), hidden_features, rngs)
    self.core_mlp = HiddenMLP(hidden_features, num_layers, rngs)
    self.legal_layer = nnx.Linear(hidden_features, action_dimension, rngs=rngs)

    self.action_dimension = action_dimension

  def __call__(self, recurrent_state: chex.Array, encoded_state: chex.Array):
    # Flatten [K, C]
    flat_encoded_state = encoded_state.reshape(*encoded_state.shape[:-2], -1)
    x = jnp.concatenate([recurrent_state, flat_encoded_state], axis=-1)
    x = self.init_layer(x)
    x = self.core_mlp(x)
    return self.legal_layer(x)


class DecEmbeddingCritic(nnx.Module):
  """Decentralized counterpart of networks.EmbeddingCritic.
  Scores one player's own encoder tokens against that player's own observation,
  for the contrastive (InfoNCE) term of the world-model loss."""
  def __init__(self, num_tokens: int, observation_features: int, rngs: nnx.Rngs):
    self.W = nnx.Linear(observation_features, num_tokens, rngs=rngs)

  def __call__(self, embedding: chex.Array, observation: chex.Array):
    #Project the observation into the embedding space
    # We want the embedding to contain positively
    # correlated information with the observation
    projected_obs = self.W(observation)
    return jnp.sum(projected_obs * embedding, axis=-1)
