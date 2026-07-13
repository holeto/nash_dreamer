"""Evaluates a learned Leduc poker world model starting from a specific endgame state
(after both private cards and the public card have been dealt), rather than the root.
This avoids the large branching factor at the root (30 private-card outcomes).
"""

from argparse import ArgumentParser
import os
import numpy as np
import jax
import flax.nnx as nnx
import jax.numpy as jnp
import time

from numpy.testing import verbose

from dreamer_ma import DreamerMA
from ma_rssm import symlog
from train_utils import load_model

from experiments.tree_view_utils import *
from experiments.eval_utils import *


parser = ArgumentParser()
parser.add_argument("--model_dir", type=str,
                    default="trained_networks/metacentrum_nets/leduc/seed_42",
                    help="Path to the directory of saved models")
parser.add_argument("--restore_step", type=int, default=30000,
                    help="Saved step of the model to restore.")
parser.add_argument("--verbose", action="store_true",
                    help="Print information about states being checked")
parser.add_argument("--render_tree", action="store_true",
                    help="Create the model EFG-style tree and render it")
parser.add_argument("--p1_card", type=int, default=0,
                    help="Player 1 private card, 0-indexed (0–5)")
parser.add_argument("--p2_card", type=int, default=2,
                    help="Player 2 private card, 0-indexed (0–5), must differ from p1_card")
parser.add_argument("--public_card", type=int, default=None,
                    help="Public card, 0-indexed (0–5), must differ from both private cards. "
                         "If omitted, the walk starts at the public card chance node and "
                         "expands all valid public card outcomes.")
parser.add_argument("--round1_actions", type=str, default="2,2",
                    help="Comma-separated action IDs for round 1 (2=call/check, 3=raise). "
                         "Default '2,2' is check-check, the shortest path to the public card.")
parser.add_argument("--max_deters", type=int, default=2,
                    help="Keep only the top-K most probable deter states at each expansion step")
parser.add_argument("--all_private_cards", action="store_true",
                    help="Iterate over all 30 ordered private card pairs and render a tree for each. "
                         "Implies --render_tree. --p1_card and --p2_card are ignored when set.")
parser.add_argument("--pre_public", action="store_true",
                    help="Start the walk right after the private cards are dealt (round 1 betting), "
                         "before the public card is revealed. The tree walker will naturally expand "
                         "round 1 actions and the public card chance node. Mutually exclusive with "
                         "--public_card.")


@chex.dataclass
class WalkCarry:
    legals: chex.Array
    game_state: GameState
    obs: chex.Array
    recurrent_state: chex.Array
    stoch_state: chex.Array
    deter_state: chex.Array
    joint_latent_infoset: chex.Array
    reward: chex.Array
    terminal: chex.Array
    after_chance: chex.Array


def check_state_one_outcome(model: DreamerMA, carry: WalkCarry, eps: float, verbose=False):
    """Check whether the best fitting deterministic state for the state
    produces valid results."""
    mistake_probs = np.zeros(5)
    differences = np.zeros(5)
    ma_rssm = model.optimizer.model
    decoded_obs = ma_rssm.get_decoder_no_jit(carry.recurrent_state, carry.deter_state)
    p1_decoded_obs, p2_decoded_obs = decoded_obs[0], decoded_obs[1]
    pred_reward, pred_terminal, pred_legal = ma_rssm.get_predictor(carry.recurrent_state, carry.deter_state)
    p1_obs_max_difference = jnp.max(jnp.abs(carry.obs[0] - p1_decoded_obs))
    p2_obs_max_difference = jnp.max(jnp.abs(carry.obs[1] - p2_decoded_obs))
    reward_difference = jnp.abs(carry.reward - pred_reward)
    legal_diference = not carry.terminal and jnp.any(pred_legal != carry.legals)
    det_prob = jnp.prod(carry.stoch_state[carry.deter_state.astype(jnp.bool)])

    differences[0] = p1_obs_max_difference
    if p1_obs_max_difference >= eps:
        mistake_probs[0] = det_prob
        if verbose:
            print(f"Real obs and decoded obs for player 1 differ by more than {eps}.")
            print(f"Max difference {p1_obs_max_difference}")
            print(f"Real obs: {carry.obs[0]}")
            print(f"Decoded obs: {p1_decoded_obs}")
    differences[1] = p2_obs_max_difference
    if p2_obs_max_difference >= eps:
        mistake_probs[1] = det_prob
        if verbose:
            print(f"Real obs and decoded obs for player 2 differ by more than {eps}.")
            print(f"Max difference {p2_obs_max_difference}")
            print(f"Real obs: {carry.obs[1]}")
            print(f"Decoded obs: {p2_decoded_obs}")
    differences[2] = int(pred_terminal != carry.terminal)
    if pred_terminal != carry.terminal:
        mistake_probs[2] = det_prob
        if verbose:
            print(f"Predicted terminal {pred_terminal} does not match real terminal {carry.terminal}.")
    differences[3] = reward_difference
    if reward_difference >= eps:
        mistake_probs[3] = det_prob
        if verbose:
            print(f"Predicted reward {pred_reward} differs from real reward {carry.reward} by more than {eps}.")
    differences[4] = int(legal_diference)
    if legal_diference:
        mistake_probs[4] = det_prob
        if verbose:
            print(f"Predicted legal actions {pred_legal} do not match real legal actions {carry.legals}.")
    return mistake_probs, differences


def get_topk_outcomes(model, stoch_state, recurrent_state, obs, probability_eps, max_deters):
    """Like get_next_outcomes but keeps only the max_deters most probable deter states
    per outcome, to bound the tree expansion."""
    deters, probs = get_next_outcomes(model, stoch_state, recurrent_state, obs, probability_eps)
    result_deters, result_probs = [], []
    for d_list, p_list in zip(deters, probs):
        if len(d_list) <= max_deters:
            result_deters.append(d_list)
            result_probs.append(p_list)
        else:
            top_idx = np.argsort(p_list)[::-1][:max_deters]
            result_deters.append([d_list[i] for i in top_idx])
            result_probs.append([p_list[i] for i in top_idx])
    return result_deters, result_probs

def decode_outcomes(model, stoch_state, recurrent_state, probabilty_eps):
    deter_states = (stoch_state >= probabilty_eps).astype(int)
    class_indices, category_indices = np.nonzero(deter_states)
    per_class_valids = []
    num_classes = stoch_state.shape[0]
    for i in range(num_classes):
        single_class_indices = category_indices[class_indices == i]
        per_class_valids.append(single_class_indices)

    combinations = cartesian_product(*per_class_valids)
    probs = []
    deters = []
    public_cards = []
    private_cards = []
    public_card_mask = np.arange(7)[None, ...]
    card_mask = np.arange(6)[None, ...]
    for comb in combinations:
        prob = np.prod(stoch_state[np.arange(num_classes), comb])
        sampled_deter = jax.nn.one_hot(comb, stoch_state.shape[-1])
        decoded_obs = model.optimizer.model.get_decoder(recurrent_state, sampled_deter)
        #Public card from the view of each player
        public_card_oh = decoded_obs[:, 8:15] >= 0.4
        public_card = np.sum(public_card_oh * public_card_mask, axis=-1)
        if np.sum(public_card_oh) != 2:
            print(f"Decoded public card is not unique: {decoded_obs[:, 8:15]}")
            print(f"Decoded public card one-hot: {public_card_oh}")
            print(f"Decoded public card: {public_card}")
            print(f"Outcome prob: {prob}")
            jax.debug.breakpoint()
        #Private cardsfrom the view of each players
        private_card_oh = decoded_obs[:, 2:8] >= 0.4
        if np.sum(private_card_oh) != 2:
            print(f"Decoded private card is not unique: {decoded_obs[:, 2:8]}")
            print(f"Decoded private card one-hot: {private_card_oh}")
            print(f"Decoded private card: {private_card}")
            print(f"Outcome prob: {prob}")
            jax.debug.breakpoint()
        private_card = np.sum(private_card_oh * card_mask, axis=-1)
        probs.append(prob)
        deters.append(sampled_deter)
        public_cards.append(public_card)
        private_cards.append(private_card)
    jax.debug.breakpoint()


def _filter_stoch(logits, probability_eps: float):
    """Softmax → zero out below eps → renormalize."""
    s = np.asarray(jax.nn.softmax(logits, axis=-1))
    s = s * (s >= probability_eps)
    total = np.sum(s, axis=-1, keepdims=True)
    return s / np.where(total > 0, total, 1.0)


def _replay_private_cards(model: DreamerMA, p1_card: int, p2_card: int,
                           probability_eps: float, get_both_obs):
    """Apply the private-card chance action and initialize the world model.

    Returns (recurrent, deter, stoch, joint_infoset, joint_dummy_action, state, obs, legals).
    """
    assert 0 <= p1_card <= 5 and 0 <= p2_card <= 5, "Card indices must be in 0–5."
    assert p1_card != p2_card, "p1_card and p2_card must be distinct."

    game = model.game
    ma_rssm = model.optimizer.model
    num_players = game.num_players()

    # generate_all_private_card_nodes orders outcomes as:
    #   for p1 in 0..5: for p2 in (all other cards in order)
    # So outcome_idx = p1_card * (total_cards - 1) + position_of_p2_among_remaining
    init_state, _ = game.initialize_structures()
    remaining = sorted(c for c in range(game.total_cards) if c != p1_card)
    p2_pos = remaining.index(p2_card)
    private_outcome_idx = p1_card * (game.total_cards - 1) + p2_pos
    private_chance_action = jnp.array([0, private_outcome_idx])
    state, terminal, reward, legals = game.apply_action(init_state, private_chance_action)

    # Init world model at post-private state via encoder (posterior).
    recurrent = ma_rssm.get_init_recurrent()
    obs = get_both_obs(state)
    stoch = _filter_stoch(ma_rssm.get_encoder(recurrent, obs), probability_eps)
    dyn_stoch = _filter_stoch(ma_rssm.get_dynamics(recurrent), probability_eps)
    decode_outcomes(model, dyn_stoch, recurrent, probability_eps)
    deters_list, probs_list = get_next_outcomes(model, stoch, recurrent, obs[None], probability_eps)
    deters_list, probs_list = deters_list[0], probs_list[0]
    if len(deters_list) == 0:
        raise RuntimeError("No deter states above threshold for post-private state. "
                           "Try lowering --probability_eps.")
    deter = deters_list[int(np.argmax(probs_list))]

    joint_dummy_infoset = jnp.zeros((num_players, ma_rssm.latent_infoset_size))
    joint_dummy_action = jnp.zeros((num_players, model.action_dimension))
    joint_infoset = ma_rssm.get_next_infoset_all(joint_dummy_infoset, obs, joint_dummy_action)
    
    # mistake_probs = np.zeros(5)
    # differences = np.zeros(5)
    # ma_rssm = model.optimizer.model
    # decoded_obs = ma_rssm.get_decoder_no_jit(recurrent, deter)
    # p1_decoded_obs, p2_decoded_obs = decoded_obs[0], decoded_obs[1]
    # pred_reward, pred_terminal, pred_legal = ma_rssm.get_predictor(recurrent, deter)
    # p1_obs_max_difference = jnp.max(jnp.abs(obs[0] - p1_decoded_obs))
    # p2_obs_max_difference = jnp.max(jnp.abs(obs[1] - p2_decoded_obs))
    # reward_difference = jnp.abs(reward - pred_reward)
    # legal_diference = not terminal and jnp.any(pred_legal != legals)
    # det_prob = jnp.prod(stoch[deter.astype(jnp.bool)])

    # differences[0] = p1_obs_max_difference
    # if p1_obs_max_difference >= 0.2:
    #     mistake_probs[0] = det_prob
    #     print(f"Real obs and decoded obs for player 1 differ by more than 0.2.")
    #     print(f"Max difference {p1_obs_max_difference}")
    #     print(f"Real obs: {obs[0]}")
    #     print(f"Decoded obs: {p1_decoded_obs}")
    # differences[1] = p2_obs_max_difference
    # if p2_obs_max_difference >= 0.2:
    #     mistake_probs[1] = det_prob
    #     print(f"Real obs and decoded obs for player 2 differ by more than 0.2.")
    #     print(f"Max difference {p2_obs_max_difference}")
    #     print(f"Real obs: {obs[1]}")
    #     print(f"Decoded obs: {p2_decoded_obs}")
    # differences[2] = int(pred_terminal != terminal)
    # if pred_terminal != terminal:
    #     mistake_probs[2] = det_prob
    #     print(f"Predicted terminal {pred_terminal} does not match real terminal {terminal}.")
    # differences[3] = reward_difference
    # if reward_difference >= 0.2:
    #     mistake_probs[3] = det_prob
    #     print(f"Predicted reward {pred_reward} differs from real reward {reward} by more than 0.2.")
    # differences[4] = int(legal_diference)
    # if legal_diference:
    #     mistake_probs[4] = det_prob
    #     print(f"Predicted legal actions {pred_legal} do not match real legal actions {legals}.")
    # jax.debug.breakpoint()
    #jax.debug.breakpoint()
    return recurrent, deter, stoch, joint_infoset, joint_dummy_action, state, obs, legals


def _replay_to_public_chance(model: DreamerMA, p1_card: int, p2_card: int,
                              round1_action_ids: list[int], probability_eps: float,
                              get_both_obs):
    """Replay game and world model up to (but not past) the public card chance node.

    Returns (recurrent, deter, joint_infoset, last_action_oh, chance_state).
    - recurrent/deter: world model state at the chance node (pre-public-card)
    - joint_infoset: latent infoset at the chance node
    - last_action_oh: one-hot action that triggered the chance node (the last round-1 action)
    - chance_state: the game state at the public card chance node
    """
    game = model.game
    ma_rssm = model.optimizer.model
    num_players = game.num_players()
    num_actions = game.num_distinct_actions()

    recurrent, deter, _, joint_infoset, last_action_oh, state, obs, _ = _replay_private_cards(
        model, p1_card, p2_card, probability_eps, get_both_obs)

    # Replay round-1 actions through both game and world model.
    for action_id in round1_action_ids:
        active_player = int(np.asarray(state.turns_this_round[0])) % 2
        joint_action = np.zeros(num_players, dtype=np.int32)
        joint_action[active_player] = action_id
        joint_action_jnp = jnp.array(joint_action)

        next_state, _, _, _ = game.apply_action(state, joint_action_jnp)
        action_oh = jax.nn.one_hot(joint_action_jnp, num_actions)

        next_recurrent = ma_rssm.get_next_recurrent(recurrent, deter, action_oh)
        next_stoch = _filter_stoch(ma_rssm.get_dynamics(next_recurrent), probability_eps)
        next_obs = get_both_obs(next_state)

        next_deters, next_probs = get_next_outcomes(
            model, next_stoch, next_recurrent, next_obs[None], probability_eps)
        next_deters, next_probs = next_deters[0], next_probs[0]
        if len(next_deters) > 0:
            deter = next_deters[int(np.argmax(next_probs))]

        joint_infoset = ma_rssm.get_next_infoset_all(joint_infoset, next_obs, action_oh)

        state = next_state
        recurrent = next_recurrent
        last_action_oh = action_oh
    

    assert game.is_chance(state), (
        "After replaying round1_actions the game state should be a chance node "
        "(public card). Check that the action sequence produces equal bets in round 1.")

    return recurrent, deter, joint_infoset, last_action_oh, state


def replay_to_endgame(model: DreamerMA, p1_card: int, p2_card: int, public_card: int,
                      round1_action_ids: list[int], probability_eps: float,
                      get_both_obs):
    """Replay game and world model along a specific path to reach the endgame state
    (after both private cards and the specific public card are dealt).

    Returns (recurrent_state, stoch_state, joint_infoset, endgame_state, endgame_legals).
    """
    assert 0 <= public_card <= 5, "public_card must be in 0–5."
    assert public_card not in (p1_card, p2_card), \
        "public_card must differ from both private cards."

    recurrent, _, joint_infoset, last_action_oh, chance_state = _replay_to_public_chance(
        model, p1_card, p2_card, round1_action_ids, probability_eps, get_both_obs)

    ma_rssm = model.optimizer.model

    # Apply public-card chance action.
    # generate_all_public_card_nodes: public_cards = arange(6) + 1 → values 1–6.
    # Index public_card (0-based) selects that card.
    pub_chance_action = jnp.array([0, public_card])
    endgame_state, _, _, endgame_legals = model.game.apply_action(chance_state, pub_chance_action)

    # Re-encode with the encoder (posterior) at the post-public state.
    # Update infoset using the LAST round-1 action + post-public observation.
    endgame_obs = get_both_obs(endgame_state)
    endgame_stoch = _filter_stoch(ma_rssm.get_encoder(recurrent, endgame_obs), probability_eps)
    joint_infoset = ma_rssm.get_next_infoset_all(joint_infoset, endgame_obs, last_action_oh)

    return recurrent, endgame_stoch, joint_infoset, endgame_state, endgame_legals


def endgame_walk_test(model: DreamerMA, p1_card: int, p2_card: int,
                      public_card: int | None,
                      round1_action_ids: list[int], max_deters: int,
                      pre_public: bool = False,
                      difference_eps=0.2, probability_eps=0.05, probability_threshold=0.005,
                      verbose=False, visualise_tree=False, tree_suffix: str = ""):
    """Walk the learned world model tree starting from a specific Leduc position.

    Mode 0 (pre_public=True): walk starts right after the private cards are dealt.
      The walker naturally expands all round-1 actions and the public card chance node.

    Mode 1 (public_card given): walk starts after both private cards and that specific
      public card have been dealt (round 2 betting subtree only).

    Mode 2 (public_card is None, pre_public=False): walk starts at the public card
      chance node and expands all valid public card outcomes via the dynamics prior.
    """
    def get_both_obs(state):
        _, p1_obs, p2_obs, _ = model.game.get_info(state)
        return jnp.stack([p1_obs, p2_obs], axis=0)

    use_real_infoset = model.use_real_infoset
    num_players = model.game.num_players()
    mistake_probs = 0
    visited = {}
    vectorized_get_obs = jax.vmap(get_both_obs, in_axes=(0), out_axes=(0))
    model_tree_root = Node("", data={"type": PAST_ACTION, "action": -1}) if visualise_tree else None
    ma_rssm = model.optimizer.model

    def _tree_walk(carry: WalkCarry, depth=0, reach_probability: float = 1.0,
                   action_outcome_history="", subtree_parent: Node = None,
                   outcome: int = -1, outcome_prob: float = 0, create_model_node: bool = False):
        nonlocal mistake_probs

        visited[action_outcome_history] = True
        state_mistake_probs, state_differences = check_state_one_outcome(
            model, carry, difference_eps, verbose)
        parent = subtree_parent
        if visualise_tree and create_model_node:
            deter_path = parent.name + f"d{outcome}"
            deter_node = Node(deter_path,
                              parent=parent,
                              data={"differences": state_differences,
                                    "prob": outcome_prob,
                                    "type": MODEL_NODE})
            parent = deter_node
        mistake_probs = mistake_probs + (state_mistake_probs * reach_probability)
        if carry.terminal:
            return
        policy_obs = (carry.obs if use_real_infoset
                      else ma_rssm.get_infoset(carry.recurrent_state, carry.deter_state,
                                               carry.joint_latent_infoset))
        pi = np.asarray(ma_rssm.get_policy_both(policy_obs, carry.legals))
        if verbose:
            print(f"Checking state {carry.game_state}")
            print(f"Reach probs {reach_probability}")
            print(f"Policy: {pi}")
        pi_mask = pi >= probability_eps
        actions = np.tile(np.arange(pi.shape[-1]), (num_players, 1)).reshape(pi.shape)
        valid_actions = [actions[i][pi_mask[i]] for i in range(num_players)]
        actions = cartesian_product(*valid_actions)
        for a in actions:
            action_prob = np.prod(pi[np.arange(a.shape[0]), a])
            action_parent = parent
            if visualise_tree:
                action_path = parent.name + f"a{a}"
                action_node = Node(action_path,
                                   parent=action_parent,
                                   data={"type": PAST_ACTION, "action": a, "prob": action_prob})
                action_parent = action_node
            next_state, next_terminal, next_reward, next_legals = model.game.apply_action(
                carry.game_state, a)
            ai_oh = jax.nn.one_hot(a, carry.legals.shape[-1])
            next_recurrent = ma_rssm.get_next_recurrent(carry.recurrent_state, carry.deter_state, ai_oh)
            dyn = ma_rssm.get_dynamics(next_recurrent)
            next_stoch_state = _filter_stoch(dyn, probability_threshold)

            is_chance = model.game.is_chance(next_state)
            if is_chance and pre_public:
                continue  # In round-1 walk mode, stop at the public card chance node.
            chance_outcomes = model.game.depth_chance_valid_outcomes(depth + 1)
            if is_chance:
                next_states, next_terminals, next_rewards, next_legals, next_probs = \
                    unroll_chance_node(model.game, next_state, chance_outcomes)
                next_terminals = np.asarray(next_terminals)
                next_rewards = np.asarray(next_rewards)
                next_legals = np.asarray(next_legals)
            else:
                next_states = jax.tree.map(lambda x: x[None, ...], next_state)
                next_terminals = np.asarray(next_terminal)[None, ...]
                next_rewards = np.asarray(next_reward)[None, ...]
                next_legals = np.asarray(next_legals)[None, ...]
            next_obs = vectorized_get_obs(next_states)
            next_obs = np.asarray(next_obs)
            next_deters, next_probs = get_topk_outcomes(
                model, next_stoch_state, next_recurrent, next_obs, probability_eps, max_deters)
            for i in range(next_terminals.shape[0]):
                outcome_parent = action_parent
                next_terminal = next_terminals[i]
                next_joint_infoset = ma_rssm.get_next_infoset_all(
                    carry.joint_latent_infoset, next_obs[i], ai_oh)
                next_reward = next_rewards[i]
                next_legal = next_legals[i]
                next_state = jax.tree.map(lambda x: x[i], next_states)
                single_outcome_deters = next_deters[i]
                single_outcome_probs = next_probs[i]
                outcome_prob = np.sum(single_outcome_probs)
                if next_terminals.shape[0] > 1 and visualise_tree:
                    outcome_path = parent.name + f"o{i}"
                    outcome_node = Node(outcome_path,
                                        parent=outcome_parent,
                                        data={"prob": outcome_prob, "type": PAST_CHANCE})
                    outcome_parent = outcome_node
                for j, deter in enumerate(single_outcome_deters):
                    new_carry = WalkCarry(
                        legals=next_legal,
                        obs=next_obs[i],
                        game_state=next_state,
                        recurrent_state=next_recurrent,
                        stoch_state=next_stoch_state,
                        deter_state=deter,
                        joint_latent_infoset=next_joint_infoset,
                        reward=next_reward,
                        terminal=next_terminal,
                        after_chance=is_chance)
                    prob = jnp.prod(next_stoch_state[deter.astype(jnp.bool)])
                    _tree_walk(new_carry, depth=depth + 1 + int(is_chance),
                               subtree_parent=outcome_parent,
                               action_outcome_history=action_outcome_history + f"a{a}o{i}",
                               reach_probability=reach_probability * prob,
                               outcome=j, outcome_prob=single_outcome_probs[j] / outcome_prob,
                               create_model_node=True)

    # -------------------------------------------------------------------------
    # Initialization: three modes.
    # -------------------------------------------------------------------------
    if pre_public:
        # --- Mode 0: walk starts right after private cards (full round-1 + public card tree) ---
        recurrent, deter, stoch, joint_infoset, _, state, obs, legals = _replay_private_cards(
            model, p1_card, p2_card, probability_threshold, get_both_obs)
        starting_depth = 1
        end_deters, end_probs = get_topk_outcomes(
            model, stoch, recurrent, np.asarray(obs)[None], probability_eps, max_deters)
        end_deters = end_deters[0]
        end_probs = end_probs[0]
        outcome_prob = np.sum(end_probs) if end_probs else 1.0
        if not end_deters:
            raise RuntimeError("No deter states found for the post-private starting point. "
                               "Try lowering --probability_eps or --probability_threshold.")
        for j, deter_j in enumerate(end_deters):
            init_carry = WalkCarry(
                legals=legals,
                obs=obs,
                game_state=state,
                recurrent_state=recurrent,
                stoch_state=stoch,
                deter_state=deter_j,
                joint_latent_infoset=joint_infoset,
                reward=jnp.array(0),
                terminal=jnp.array(False),
                after_chance=jnp.array(True))
            prob = jnp.prod(stoch[deter_j.astype(jnp.bool)])
            _tree_walk(init_carry, depth=starting_depth,
                       subtree_parent=model_tree_root,
                       action_outcome_history=f"priv_d{j}",
                       reach_probability=float(prob),
                       outcome=j,
                       outcome_prob=end_probs[j] / outcome_prob,
                       create_model_node=True)
    elif public_card is not None:
        # --- Mode 1: specific endgame (single public card outcome) ---
        recurrent, stoch, joint_infoset, endgame_state, endgame_legals = replay_to_endgame(
            model, p1_card, p2_card, public_card, round1_action_ids, probability_threshold,
            get_both_obs)
        starting_depth = 2 + len(round1_action_ids)
        endgame_obs = get_both_obs(endgame_state)
        end_deters, end_probs = get_topk_outcomes(
            model, stoch, recurrent, np.asarray(endgame_obs)[None], probability_eps, max_deters)
        end_deters = end_deters[0]
        end_probs = end_probs[0]
        outcome_prob = np.sum(end_probs) if end_probs else 1.0
        if not end_deters:
            raise RuntimeError("No deter states found for the endgame starting point. "
                               "Try lowering --probability_eps or --probability_threshold.")
        for j, deter in enumerate(end_deters):
            init_carry = WalkCarry(
                legals=endgame_legals,
                obs=endgame_obs,
                game_state=endgame_state,
                recurrent_state=recurrent,
                stoch_state=stoch,
                deter_state=deter,
                joint_latent_infoset=joint_infoset,
                reward=jnp.array(0),
                terminal=jnp.array(False),
                after_chance=jnp.array(True))
            prob = jnp.prod(stoch[deter.astype(jnp.bool)])
            _tree_walk(init_carry, depth=starting_depth,
                       subtree_parent=model_tree_root,
                       action_outcome_history=f"endgame_d{j}",
                       reach_probability=float(prob),
                       outcome=j,
                       outcome_prob=end_probs[j] / outcome_prob,
                       create_model_node=True)
    else:
        # --- Mode 2: all public card outcomes (dynamics-based, shared stoch) ---
        recurrent, _, joint_infoset, last_action_oh, chance_state = _replay_to_public_chance(
            model, p1_card, p2_card, round1_action_ids, probability_threshold, get_both_obs)
        starting_depth = 2 + len(round1_action_ids)

        # Single shared stochastic state from dynamics — consistent with how _tree_walk
        # handles chance nodes encountered during the walk (prior, not posterior).
        pub_stoch = _filter_stoch(ma_rssm.get_dynamics(recurrent), probability_threshold)

        chance_depth = 1 + len(round1_action_ids)
        num_pub_outcomes = model.game.depth_chance_valid_outcomes(chance_depth)
        next_states, next_terminals, next_rewards, next_legals, next_probs = unroll_chance_node(
            model.game, chance_state, num_pub_outcomes)

        next_terminals = np.asarray(next_terminals)
        next_rewards = np.asarray(next_rewards)
        next_legals = np.asarray(next_legals)

        # Cluster deters from the shared stoch across all outcomes at once.
        all_next_obs = np.asarray(vectorized_get_obs(next_states))
        outcome_deters, outcome_probs_all = get_topk_outcomes(
            model, pub_stoch, recurrent, all_next_obs, probability_eps, max_deters)

        for i in range(next_terminals.shape[0]):
            outcome_parent = model_tree_root
            single_deters = outcome_deters[i]
            single_probs = outcome_probs_all[i]
            total_prob = np.sum(single_probs) if single_probs else 1.0

            # Update infoset with last round-1 action + post-public observation.
            outcome_infoset = ma_rssm.get_next_infoset_all(
                joint_infoset, all_next_obs[i], last_action_oh)

            if visualise_tree:
                outcome_node = Node(f"o{i}",
                                    parent=outcome_parent,
                                    data={"prob": float(next_probs[i]), "type": PAST_CHANCE})
                outcome_parent = outcome_node

            for j, deter in enumerate(single_deters):
                init_carry = WalkCarry(
                    legals=next_legals[i],
                    obs=all_next_obs[i],
                    game_state=jax.tree.map(lambda x: x[i], next_states),
                    recurrent_state=recurrent,
                    stoch_state=pub_stoch,
                    deter_state=deter,
                    joint_latent_infoset=outcome_infoset,
                    reward=next_rewards[i],
                    terminal=next_terminals[i],
                    after_chance=jnp.array(True))
                prob = jnp.prod(pub_stoch[deter.astype(jnp.bool)])
                _tree_walk(init_carry, depth=starting_depth,
                           subtree_parent=outcome_parent,
                           action_outcome_history=f"o{i}d{j}",
                           reach_probability=float(next_probs[i]) * float(prob),
                           outcome=j,
                           outcome_prob=single_probs[j] / total_prob,
                           create_model_node=True)

    num_visited_states = len(visited)
    avg_mistake_probs = mistake_probs / num_visited_states
    avg_mistake_probs = np.minimum(avg_mistake_probs, 1.0)
    if visualise_tree:
        render_tree(model_tree_root, model, suffix=tree_suffix)
    return avg_mistake_probs


def main():
    args = parser.parse_args()
    model_dir = args.model_dir
    if not model_dir.startswith("/"):
        model_dir = os.getcwd() + "/" + model_dir
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory {model_dir} does not exist.")

    round1_action_ids = [int(x) for x in args.round1_actions.split(",")]
    start_time = time.time()

    # Load model once.
    model = None
    found = False
    for filename in os.listdir(model_dir):
        if not os.path.isfile(os.path.join(model_dir, filename)):
            continue
        parts = filename.split(".")
        if len(parts) != 2 or parts[1] != "pkl":
            continue
        step = int(parts[0].split("_")[-1])
        if step != args.restore_step:
            continue

        model_path = os.path.join(model_dir, filename)
        if model is None:
            model = load_model(model_path)
            assert isinstance(model, DreamerMA), \
                f"The saved model should be DreamerMA, got {model.__class__}"
        else:
            temp_model = load_model(model_path)
            nnx.update(model.optimizer, nnx.state(temp_model.optimizer))
        found = True
        print(f"Restored model from {model_path}")
        break

    if not found:
        raise FileNotFoundError(
            f"No file matching step_{args.restore_step}.pkl found in {model_dir}.")

    if args.pre_public and args.public_card is not None:
        raise ValueError("--pre_public and --public_card are mutually exclusive.")

    # Build list of (p1_card, p2_card) pairs to evaluate.
    if args.all_private_cards:
        total_cards = model.game.total_cards  # 6 for standard Leduc
        private_pairs = [(p1, p2)
                         for p1 in range(total_cards)
                         for p2 in range(total_cards)
                         if p1 != p2]
        public_card = None
        visualise_tree = True
        mode_desc = "Mode 0 (round 1 walk)" if args.pre_public else "Mode 2 (public card chance)"
        print(f"Iterating all {len(private_pairs)} private card pairs "
              f"({mode_desc}, round1_actions={round1_action_ids}, max_deters={args.max_deters})")
    else:
        private_pairs = [(args.p1_card, args.p2_card)]
        public_card = args.public_card
        visualise_tree = args.render_tree
        if args.pre_public:
            mode_desc = "round 1 walk (pre-public)"
        elif public_card is not None:
            mode_desc = f"public_card={public_card}"
        else:
            mode_desc = "all public card outcomes (chance node)"
        print(f"Starting evaluation: p1_card={args.p1_card}, p2_card={args.p2_card}, "
              f"{mode_desc}, round1_actions={round1_action_ids}, max_deters={args.max_deters}")

    category_names = ["Player 1 obs", "Player 2 obs", "Terminal", "Reward", "Legal actions"]
    for p1_card, p2_card in private_pairs:
        suffix = f"_p1c{p1_card}_p2c{p2_card}" if args.all_private_cards else ""
        if args.pre_public:
            suffix += "_round1"
        if args.all_private_cards:
            print(f"\n--- p1_card={p1_card}, p2_card={p2_card} ---")
        mistake_probs = endgame_walk_test(
            model,
            p1_card=p1_card,
            p2_card=p2_card,
            public_card=public_card,
            round1_action_ids=round1_action_ids,
            max_deters=args.max_deters,
            pre_public=args.pre_public,
            verbose=args.verbose,
            visualise_tree=visualise_tree,
            tree_suffix=suffix)

        print(f"Mistake probabilities (averaged over {len(mistake_probs)} categories):")
        for name, prob in zip(category_names, mistake_probs):
            print(f"  {name}: {prob:.4f}")

    print(f"\nTotal evaluation time: {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
