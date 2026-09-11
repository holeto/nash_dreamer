"""Top-level model for the decentralized NashDreamer variant.

Subclasses `DreamerMA` so that `train_step`, `train_model`, `generate_key`, the
pickle hooks and every `isinstance(model, DreamerMA)` check in the evaluation
scripts keep working unchanged.  Only two methods are overridden: `init`, which
swaps in the decentralized world model / buffer / learner, and
`update_world_model`, which adds the player axis and drops the four
latent-infoset loss terms along with the networks that produced them.

All config dataclasses are reused verbatim from `train_utils`; the
`infoset_*_details` and `beta_infoset` fields are simply ignored here, the same
arrangement `MMDConfig` already documents for the fields only MMDDreamer reads.
"""

import chex
import jax
import jax.numpy as jnp
from flax import nnx
import optax
from functools import partial

from nash_dreamer.train_utils import *
from nash_dreamer.distributions import *
from nash_dreamer.ma_rssm import MARSSM
from nash_dreamer.ma_rssm_decentralized import (DecentralizedMARSSM, DecentralizedPredictionStep,
                                   create_decentralized_dreamer_optimizer)
from nash_dreamer.replay_buffer_decentralized import DecentralizedWMReplayBuffer
from nash_dreamer.rnad_dreamer_decentralized import DecentralizedRNaDDreamer
from nash_dreamer.dreamer_ma import DreamerMA
from nash_dreamer.networks import CriticNetwork
from envs.jax_game import JaxGame


class DecentralizedDreamerMA(DreamerMA):
  """NashDreamer with one RSSM per player and no latent-infoset networks."""

  def init(self):
    self.jax_rngs = jax.random.key(self.init_seed)

    assert self.wm_config.use_original_infoset, (
      "The decentralized variant is only defined for real infosets. "
      "Pass --use_original_infoset.")
    assert isinstance(self.ac_config, RNaDConfig), (
      "Only the RNaD actor-critic is implemented for the decentralized variant, "
      f"got {self.ac_config.__class__.__name__}.")

    self.buffer = DecentralizedWMReplayBuffer(self.game, self.buffer_config, self.wm_config,
                                              self.init_seed, self.ac_config.state_sample_threshold)

    rngs = nnx.Rngs(jax.random.key(self.init_seed))
    self.optimizer = create_decentralized_dreamer_optimizer(self.game, self.wm_config, self.ac_config,
                                                            self.opt_config, rngs, self.init_seed)
    ma_rssm = self.optimizer.model
    self.learner_steps = 0

    #Assuming that this contains a terminal state as well
    self.trajectory_max = self.game.max_trajectory_length()
    self.non_chance_trajectory_max = self.game.max_trajectory_lenght_no_chance()

    self.action_dimension = self.game.num_distinct_actions()
    self.infoset_size = ma_rssm.infoset_size
    self.latent_infoset_size = ma_rssm.latent_infoset_size
    self.use_real_infoset = ma_rssm.use_real_infoset
    assert self.game.num_players() > 1, f"This implementation of Dreamer assumes a game with at least 2 players not {self.game.num_players()}"
    self.recurrent_state_size = ma_rssm.rec_state_size
    #Vanilla SGD coupled with the gradients
    # we compute manually in actor critic will handle
    # the EMA updates for us.
    target_tx = optax.sgd(self.ac_config.target_network_update)
    #Decentralized: the target critic mirrors the per-player critic, so its
    # input is ONE player's infoset, not the joint one.
    target_optimizer = nnx.Optimizer(model=CriticNetwork(self.infoset_size,
                                                         self.ac_config.bin_range,
                                                         self.ac_config.critic_network_details[0],
                                                         self.ac_config.critic_network_details[1],
                                                         rngs=rngs),
                                                         tx = target_tx)
    self.network_keys = ma_rssm.network_names[:-2]
    self.actor_critic = DecentralizedRNaDDreamer(self.game, self.ac_config, self.optimizer, target_optimizer)
    #Four networks, not five: there is no infoset network to sample with.
    self.buffer.cache_sampling(ma_rssm.seq, ma_rssm.enc, ma_rssm.observer, ma_rssm.actor)
    self.grad_norms = {k: 0 for k in self.network_keys}
    #Key order matters: it is zipped positionally against the `losses` list in
    # update_world_model. The four infoset terms are gone.
    self.metrics = {'dec': 0, 'con': 0, 'leg': 0, 'rew': 0, 'dyn': 0, 'rep': 0}
    assert self.game.information_state_tensor_shape() == self.game.observation_tensor_shape(), "Specification of use_real_infoset is only sound when the environment provides infoset in place of observation!"
    print(f"Decentralized world model. Using original game infosets of shape {self.infoset_size}, "
          f"per-player latent state of shape {self.latent_infoset_size}")
    if not self.wm_config.max_divergence_scaling:
      print(f"Scaling turned off")
      self.maximum_divergence = 1
    #For each of the scaling losses,
    # we multiply by 2, since the metric is computed in both directions
    elif self.wm_config.jsd:
      self.maximum_divergence = 2 * jnp.log(self.wm_config.encoded_categories ** self.wm_config.encoded_classes)
    else:
      self.maximum_divergence = 2 * (jnp.log(self.wm_config.encoded_categories ** self.wm_config.encoded_classes) - jnp.log((self.wm_config.uniform_mix)))

    print(f"Using {'JSD' if self.wm_config.jsd else 'KL'} for prior/posterior distance. Maximum value is {self.maximum_divergence}")
    #This class overrides init() wholesale rather than extending DreamerMA's, so anything
    # DreamerMA.init does has to be repeated here. That includes rejecting --joint_prior:
    # DecentralizedMARSSM builds its own per player DynamicsPredictor, so the joint head would
    # have to be ported there separately. Failing loudly beats silently ignoring the flag and
    # reporting a factored prior's numbers as if they came from a joint one.
    assert not self.wm_config.joint_prior, (
      "--joint_prior is not implemented for the decentralized world model, only the centralized "
      "one. Its prior is applied per player through its own DynamicsPredictor and needs its own "
      "port, which is deliberately out of scope until the centralized version is validated.")
    self.init_stage_one()
    self.check_two_stage()

  @partial(nnx.jit, static_argnums=(0, 4))
  def update_world_model(self, optimizer: nnx.Optimizer, timestep: TimeStep, rng_key,
                         two_stage_warm_up: bool = False):
    """Compound loss for the entire decentralized world model.

    Every network is applied per player over a leading player axis, and each
    player's chain is driven by its OWN observation and OWN action. Six loss
    terms remain; the four latent-infoset terms have no counterpart here.

    two_stage_warm_up is the static flag the inherited `train_step` passes while a
    stage one runs, with the same meaning as in the centralized model: no dynamics loss,
    and no representation loss either unless the VQ-VAE commitment term replaces it."""
    sample_keys = jax.random.split(rng_key, self.non_chance_trajectory_max * self.wm_config.batch_size)
    sample_keys = sample_keys.reshape((self.non_chance_trajectory_max, self.wm_config.batch_size))
    #Same posterior freeze as the centralized model: after stage one the Encoder/Observer output
    # is detached, because _sample_deter is a straight-through estimator and every downstream loss
    # would otherwise keep training them. See dreamer_ma.py for the full reasoning.
    two_stage = self.uses_two_stage()
    freeze_posterior = two_stage and not two_stage_warm_up
    #--complete_two_stage additionally freezes every network but the prior in stage two.
    freeze_world_model = self.wm_config.complete_two_stage and not two_stage_warm_up

    def world_model_loss(ma_rssm: DecentralizedMARSSM):
      l_pred, l_dyn, l_rep = 0, 0, 0

      #Per-player application of the shared modules.
      per_player = lambda net, *args: nnx.vmap(
        MARSSM.call_net, in_axes=(None, *(0 for _ in args)), out_axes=0)(net, *args)

      #[Trajectory, Batch, ...]
      @nnx.scan(in_axes=(nnx.Carry, 0, None), out_axes=(nnx.Carry, 0))
      def _predict_over_timestep(carry, xs, model: DecentralizedMARSSM):

        recurrent_state, timestep = carry
        action, obs, cur_key = xs
        #Each player encodes only its own observation into its own posterior,
        # which its own recurrent state then contextualizes.
        stochastic_state = model.get_encoder_no_jit(recurrent_state, obs)
        if freeze_posterior:
          stochastic_state = jax.lax.stop_gradient(stochastic_state)
        stochastic_state = add_uniform_mix(stochastic_state, self.wm_config.uniform_mix)
        #Independent sample per player -- see DecentralizedMARSSM._sample_deter
        # for why this must not be a single stacked sample_categorical call.
        deterministic_state = model._sample_deter(stochastic_state, cur_key)
        #Cut the dynamics network's input too, or the one loss still running would keep
        # training the sequential network that produced it. See dreamer_ma.py.
        dynamics_input = jax.lax.stop_gradient(recurrent_state) if freeze_world_model else recurrent_state
        prior_stochastic_state = per_player(model.dyn, dynamics_input)
        prior_stochastic_state = add_uniform_mix(prior_stochastic_state, self.wm_config.uniform_mix)
        #Each player reconstructs only its own observation.
        decoded_obs = per_player(model.dec, recurrent_state, deterministic_state)
        reward = per_player(model.rew, recurrent_state, deterministic_state)
        done = per_player(model.term, recurrent_state, deterministic_state)
        legal = per_player(model.leg, recurrent_state, deterministic_state)
        #Dont use symexp here during training. Otherwise we would be training
        # the symexp outputs to match the symlog inputs.
        #Player i advances on action[i] only: it never observes the opponent's
        # action, which is what makes its transition model unidentifiable.
        new_recurrent = per_player(model.seq, recurrent_state, deterministic_state, action)
        preds = DecentralizedPredictionStep(
                                recurrent_state = recurrent_state,
                                repr_state = stochastic_state,
                                deter_state = deterministic_state,
                                decoded_obs = decoded_obs,
                                reward_dist_logit = reward,
                                done_logit = done,
                                legal_logit = legal,
                                dynamics_state = prior_stochastic_state,
                                #Packed from the state already sampled above.
                                joint_latent_infoset = model._pack_latent(recurrent_state, deterministic_state))

        return (new_recurrent, timestep + 1), preds

      xs = (timestep.action, timestep.obs, sample_keys)
      init_recurrent = ma_rssm.get_init_recurrent(self.wm_config.batch_size)
      vectorized_predict = nnx.vmap(_predict_over_timestep, in_axes=((0, None), 1, None), out_axes=(0, 1))
      _, predictions = vectorized_predict((init_recurrent, 0), xs, ma_rssm)

      #[Trajectory, Batch, num_players, obs_size]
      if self.wm_config.obs_loss_bce:
        reconstruction_loss = optax.sigmoid_binary_cross_entropy(predictions.decoded_obs, timestep.obs)
      else:
        reconstruction_loss = -get_normal_log_prob(predictions.decoded_obs, timestep.obs, use_symlog=True)
      dec = get_loss_mean_with_mask(reconstruction_loss, timestep.valid[..., None, None], normalization_mult=2)
      l_pred += dec
      #[Trajectory, Batch, num_players, 1]
      continuation_loss = optax.sigmoid_binary_cross_entropy(predictions.done_logit, timestep.terminal[..., None, None])
      con = get_loss_mean_with_mask(continuation_loss, timestep.valid[..., None, None], normalization_mult=2)
      l_pred += con
      #[Trajectory, Batch, players, action_dim]
      legal_loss = optax.sigmoid_binary_cross_entropy(predictions.legal_logit, timestep.legal)
      #Legal actions should not be trained in terminal states, as there are no legal actions there
      leg = get_loss_mean_with_mask(legal_loss, timestep.valid[..., None, None] & ~timestep.terminal[..., None, None], normalization_mult=2)
      l_pred += leg
      #[Trajectory, Batch, num_players, 2 * bin_range + 1]
      bins = jnp.arange((2 * self.wm_config.bin_range) + 1) - self.wm_config.bin_range
      #Each player predicts the reward from ITS OWN perspective: r for player 1,
      # -r for player 2. get_predictor_no_jit inverts this back to the
      # p1-perspective scalar the rest of the code expects.
      per_player_reward = jnp.stack((timestep.reward, -timestep.reward), axis=-1)
      reward_loss = -get_bin_log_prob(predictions.reward_dist_logit, bins, per_player_reward, use_symlog=True)
      rew = get_loss_mean_with_mask(reward_loss, timestep.valid[..., None, None], normalization_mult=2)
      l_pred += rew

      #Using free bits to clip dynamics and representation losses
      # thus disabling their gradient when they are below free_bits_clip_threshold
      #[Trajectory, Batch, num_players, encoded_classes, encoded_categories]
      posterior = nnx.softmax(predictions.repr_state, axis=-1)
      prior = nnx.softmax(predictions.dynamics_state, axis=-1)
      #[Trajectory, Batch, num_players]
      metric = jsd if self.wm_config.jsd else kl_divergence
      #Same gating as the centralized model, including the posterior freeze after stage one:
      # the VQ-VAE commitment term is stage one only, and the KL balancing representation loss
      # never runs under two stage training. See dreamer_ma.py for the full reasoning.
      compute_dynamics = not two_stage_warm_up
      compute_representation = ((self.wm_config.vq_vae_posterior and two_stage_warm_up)
                                if two_stage else True)
      if compute_dynamics:
        dynamics_loss = metric(jax.lax.stop_gradient(posterior), prior)
        l_dyn += jnp.maximum(self.wm_config.free_bits_clip_threshold,
                             get_loss_mean_with_mask(dynamics_loss, timestep.valid[..., None], normalization_mult=2))
      if compute_representation:
        if self.wm_config.vq_vae_posterior:
          # VQ-VAE style commitment loss: instead of pulling the posterior toward the prior,
          # sharpen it toward its own per-variable argmax (a stop-gradient one-hot target) --
          # cross entropy between the posterior and that target, independent of the prior.
          # The two trailing axes summed over are the per-player [classes, categories], the
          # player axis stays and is averaged by the mask below, as every other term here.
          argmax_target = jax.lax.stop_gradient(
              jax.nn.one_hot(jnp.argmax(posterior, axis=-1), posterior.shape[-1]))
          repr_loss = -jnp.sum(argmax_target * jnp.log(posterior), axis=(-1, -2))
        else:
          repr_loss = metric(posterior, jax.lax.stop_gradient(prior))
        l_rep += jnp.maximum(self.wm_config.free_bits_clip_threshold,
                             get_loss_mean_with_mask(repr_loss, timestep.valid[..., None], normalization_mult=2))

      #Multiply the prediction losses with
      # the maximum information metric value, to prevent collapse
      # being the optimal solution.
      mults = [*(self.wm_config.beta_prediction, ) * 4,
               self.wm_config.beta_dynamics / self.maximum_divergence,
               self.wm_config.beta_representation / self.maximum_divergence]

      losses = [dec, con, leg, rew, l_dyn, l_rep]

      wm_keys = self.metrics.keys()
      metrics = {k: v * m for k, v, m in zip(wm_keys, losses, mults)}

      if freeze_world_model:
        #Reported but detached, exactly as in the centralized model.
        grad_losses = [l if k == 'dyn' else jax.lax.stop_gradient(l)
                       for k, l in zip(wm_keys, losses)]
      else:
        grad_losses = losses

      compound_loss = sum(l * m for l, m in zip(grad_losses, mults))

      return compound_loss, (predictions, metrics)

    grad_norms = self.grad_norms.copy()
    func_data, grad = nnx.value_and_grad(world_model_loss, has_aux=True, argnums=(0))(
                    optimizer.model)
    if self.wm_config.report_gradnorms:
      for k in self.network_keys:
        grad_norms[k] = optax.tree.norm(grad[k], ord=2)

    loss, (pred_step, metrics) = func_data
    optimizer.update(grad)

    return loss, pred_step, metrics, grad_norms
