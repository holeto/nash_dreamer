import jax
import jax.numpy as jnp
import numpy as np
import chex
import functools
from games.jax_game import JaxGame, GameState, InformationType

INVALID_ID = 0
FORFEIT_ID = 1
DARE_ID = 2


@chex.dataclass(frozen=True)
class DuelGameState(GameState):
    action_history: chex.Array    # [max_turns, num_actions-1] = [4, 2]
    strengths: chex.Array         # [2] int16 (0=weak, 1=strong, 2=very_strong)
    strength_boost: chex.Array    # [] int16 scalar: 0=none, 1=P1 boosted, 2=P2 boosted (PUBLIC)
    current_chips: chex.Array     # [2] int16
    turns_this_round: chex.Array  # [1] int16
    current_round: chex.Array     # [] int16 scalar: 0 or 1
    terminal: chex.Array          # bool scalar
    turn: int
    is_chance: chex.Array         # bool scalar


class JaxStrengthDuel(JaxGame):
    def __init__(self):
        # INVALID, FORFEIT, DARE
        self.num_actions = 3
        # Two rounds, max 2 actions per round
        self.max_turns = 4
        # weak=0, strong=1, very_strong=2
        self.num_strength_values = 3
        self.players = 2
        # Starting ante 1 each; P1 raises to 2, P2 matches to 2 (round 1),
        # P1 raises to 3, P2 matches to 3 (round 2) → max chips = 3
        self.max_reward = 3
        # 4 initial chance outcomes: (weak,weak), (weak,strong), (strong,weak), (strong,strong)
        self.initial_chance_outcomes = 4
        self.invalid_action_mask = jax.nn.one_hot(INVALID_ID, self.num_actions)

    def game_name(self):
        return "strength_duel"

    def params_dict(self):
        return {}

    def information_type(self):
        return InformationType.IIG

    def num_players(self):
        return 2

    def num_distinct_actions(self):
        return self.num_actions

    def max_chance_outcomes(self):
        return self.initial_chance_outcomes

    def max_trajectory_length(self):
        # 4 action turns + 2 chance nodes + 1 terminal
        return self.max_turns + 3

    def max_trajectory_lenght_no_chance(self):
        return self.max_turns + 1

    def depth_chance_outcomes(self, depth: int):
        if depth == 0:
            return self.initial_chance_outcomes
        elif depth == 3:
            return self.initial_chance_outcomes  # padded to 4
        return 1

    def depth_chance_valid_outcomes(self, depth: int):
        if depth == 0:
            return self.initial_chance_outcomes
        elif depth == 3:
            return 2  # only 2 valid boost outcomes
        return 1

    def public_state_tensor_shape(self):
        # action history flat (4*2=8) + strength_boost one-hot (3 values)
        return self.max_turns * (self.num_actions - 1) + self.num_strength_values

    def information_state_tensor_shape(self):
        # player id one-hot (2) + own strength one-hot (3) + public state (11)
        return self.players + self.num_strength_values + self.public_state_tensor_shape()

    def state_tensor_shape(self):
        # both strengths one-hot (2*3=6) + public state (11)
        return 2 * self.num_strength_values + self.public_state_tensor_shape()

    def observation_tensor_shape(self):
        return self.information_state_tensor_shape()

    @functools.partial(jax.jit, static_argnums=(0,))
    def generate_all_initial_nodes(self, game_state: DuelGameState):
        """4 equally-probable initial strength assignments: WW, WS, SW, SS."""
        all_strengths = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=jnp.int16)

        forfeit_oh = jax.nn.one_hot(FORFEIT_ID, self.num_actions)
        # Bets are equal at round start → forfeit illegal for P1
        p1_legal = jnp.ones(self.num_actions) - self.invalid_action_mask - forfeit_oh
        p2_legal = self.invalid_action_mask
        legals = jnp.stack([p1_legal, p2_legal], axis=0)
        legals = jnp.tile(legals[None, ...], (self.initial_chance_outcomes, 1, 1))

        stacked = jax.tree_util.tree_map(
            lambda x: jnp.tile(x[None, ...], (self.initial_chance_outcomes,) + (1,) * len(x.shape)),
            game_state)

        game_states = DuelGameState(
            action_history=stacked.action_history,
            strengths=all_strengths,
            strength_boost=stacked.strength_boost,
            current_chips=stacked.current_chips.astype(jnp.int16),
            turns_this_round=stacked.turns_this_round.astype(jnp.int16),
            current_round=stacked.current_round.astype(jnp.int16),
            terminal=stacked.terminal.astype(bool),
            turn=stacked.turn,
            is_chance=jnp.zeros(self.initial_chance_outcomes, dtype=bool))

        probs = jnp.ones(self.initial_chance_outcomes) / self.initial_chance_outcomes
        return game_states, legals, probs

    @functools.partial(jax.jit, static_argnums=(0,))
    def generate_all_boost_nodes(self, game_state: DuelGameState):
        """2 valid strength boosts padded to 4: outcome 0 boosts P1, outcome 1 boosts P2."""
        boost_increments = jnp.array([[1, 0], [0, 1], [0, 0], [0, 0]], dtype=jnp.int16)
        # strength_boost field: 1=P1 boosted, 2=P2 boosted, 0=invalid padding
        boost_indicators = jnp.array([1, 2, 0, 0], dtype=jnp.int16)

        forfeit_oh = jax.nn.one_hot(FORFEIT_ID, self.num_actions)
        # Bets equal at round 2 start → forfeit illegal for P1
        p1_legal = jnp.ones(self.num_actions) - self.invalid_action_mask - forfeit_oh
        p2_legal = self.invalid_action_mask
        legals = jnp.stack([p1_legal, p2_legal], axis=0)
        legals = jnp.tile(legals[None, ...], (self.initial_chance_outcomes, 1, 1))

        stacked = jax.tree_util.tree_map(
            lambda x: jnp.tile(x[None, ...], (self.initial_chance_outcomes,) + (1,) * len(x.shape)),
            game_state)

        game_states = DuelGameState(
            action_history=stacked.action_history,
            strengths=(stacked.strengths + boost_increments).astype(jnp.int16),
            strength_boost=boost_indicators,
            current_chips=stacked.current_chips.astype(jnp.int16),
            # Reset turns_this_round for the new round
            turns_this_round=jnp.zeros((self.initial_chance_outcomes, 1), dtype=jnp.int16),
            current_round=jnp.ones(self.initial_chance_outcomes, dtype=jnp.int16),
            terminal=stacked.terminal.astype(bool),
            turn=stacked.turn,
            is_chance=jnp.zeros(self.initial_chance_outcomes, dtype=bool))

        probs = jnp.array([0.5, 0.5, 0.0, 0.0])
        return game_states, legals, probs

    def is_chance(self, game_state: DuelGameState) -> chex.Array:
        return game_state.is_chance

    @functools.partial(jax.jit, static_argnums=(0,))
    def get_outcomes_and_probs(self, game_state: DuelGameState):
        outcomes = jnp.stack(
            [jax.nn.one_hot(0, self.initial_chance_outcomes),
             jnp.arange(self.initial_chance_outcomes)],
            axis=-1)

        def invalid_probs(gs):
            return jnp.zeros(self.initial_chance_outcomes)

        def initial_probs(gs):
            return jnp.ones(self.initial_chance_outcomes) / self.initial_chance_outcomes

        def boost_probs(gs):
            return jnp.array([0.5, 0.5, 0.0, 0.0])

        probs = jax.lax.cond(
            game_state.is_chance,
            lambda s: jax.lax.cond(s.turn == 0, initial_probs, boost_probs, s),
            invalid_probs,
            game_state)

        return outcomes, probs

    @functools.partial(jax.jit, static_argnums=(0,))
    def initialize_structures(self):
        current_chips = jnp.ones(self.players, dtype=jnp.int16)
        action_history = jnp.zeros([self.max_turns, self.num_actions - 1])
        turns_this_round = jnp.zeros(1, dtype=jnp.int16)
        legals = jnp.ones((2, self.num_actions))

        game_state = DuelGameState(
            action_history=action_history,
            strengths=jnp.full(2, -1, dtype=jnp.int16),
            strength_boost=jnp.array(0, dtype=jnp.int16),
            current_chips=current_chips,
            turns_this_round=turns_this_round,
            current_round=jnp.array(0, dtype=jnp.int16),
            terminal=jnp.array(False),
            turn=0,
            is_chance=jnp.array(True))
        return game_state, legals

    @functools.partial(jax.jit, static_argnums=(0,))
    def get_info(self, game_state: DuelGameState):
        strengths_oh = jax.nn.one_hot(game_state.strengths, self.num_strength_values)  # [2, 3]
        boost_oh = jax.nn.one_hot(game_state.strength_boost, self.num_strength_values)  # [3]

        public_state = jnp.concatenate(
            [game_state.action_history.ravel(), boost_oh.ravel()], axis=0)  # [11]
        public_state = jnp.where(game_state.is_chance, jnp.zeros_like(public_state), public_state)

        p1_player = jax.nn.one_hot(0, 2)
        p1_info = jnp.concatenate([p1_player, strengths_oh[0], public_state], axis=0)  # [16]
        p1_info = jnp.where(game_state.is_chance, jnp.zeros_like(p1_info), p1_info)

        p2_info = jnp.concatenate([1 - p1_player, strengths_oh[1], public_state], axis=0)  # [16]
        p2_info = jnp.where(game_state.is_chance, jnp.zeros_like(p2_info), p2_info)

        state_tensor = jnp.concatenate([strengths_oh.ravel(), public_state], axis=0)  # [17]
        state_tensor = jnp.where(game_state.is_chance, jnp.zeros_like(state_tensor), state_tensor)

        return state_tensor, p1_info, p2_info, public_state

    @functools.partial(jax.jit, static_argnums=(0,))
    def apply_action(self, game_state: DuelGameState, actions: chex.Array):
        return jax.lax.cond(
            game_state.is_chance,
            self.apply_action_chance,
            self.apply_action_no_chance,
            game_state, actions)

    @functools.partial(jax.jit, static_argnums=(0,))
    def apply_action_chance(self, game_state: DuelGameState, actions: chex.Array):
        init_chance = game_state.turn == 0

        def apply_init(gs, acts):
            outcomes, legals, probs = self.generate_all_initial_nodes(gs)
            action_oh = jax.nn.one_hot(acts[1], self.initial_chance_outcomes)
            outcome = jax.tree_util.tree_map(
                lambda x: jnp.sum(
                    x * jnp.reshape(action_oh, (action_oh.shape[0],) + (1,) * len(x.shape[1:])),
                    axis=0),
                outcomes)
            legals = jnp.sum(action_oh[..., None, None] * legals, axis=0)
            return outcome, jnp.array(False), jnp.array(0, dtype=jnp.float32), legals

        def apply_boost(gs, acts):
            outcomes, legals, probs = self.generate_all_boost_nodes(gs)
            action_oh = jax.nn.one_hot(acts[1], self.initial_chance_outcomes)
            outcome = jax.tree_util.tree_map(
                lambda x: jnp.sum(
                    x * jnp.reshape(action_oh, (action_oh.shape[0],) + (1,) * len(x.shape[1:])),
                    axis=0),
                outcomes)
            legals = jnp.sum(action_oh[..., None, None] * legals, axis=0)
            return outcome, jnp.array(False), jnp.array(0, dtype=jnp.float32), legals

        outcome, terminal, reward, legals = jax.lax.cond(
            init_chance, apply_init, apply_boost, game_state, actions)

        new_game_state = DuelGameState(
            action_history=outcome.action_history,
            strengths=outcome.strengths.astype(jnp.int16),
            strength_boost=outcome.strength_boost.astype(jnp.int16),
            current_chips=outcome.current_chips.astype(jnp.int16),
            turns_this_round=outcome.turns_this_round.astype(jnp.int16),
            current_round=outcome.current_round.astype(jnp.int16),
            terminal=outcome.terminal.astype(bool),
            turn=outcome.turn.astype(int),
            is_chance=outcome.is_chance.astype(bool))

        return new_game_state, terminal, reward, legals

    @functools.partial(jax.jit, static_argnums=(0,))
    def apply_action_no_chance(self, game_state: DuelGameState, actions: chex.Array):
        oh_actions = jax.nn.one_hot(actions, self.num_actions)  # [2, 3]
        oh_turn = jax.nn.one_hot(game_state.turn, self.max_turns)  # [4]
        forfeit_oh = jax.nn.one_hot(FORFEIT_ID, self.num_actions)  # [3]

        # Scalar current player index
        current_player = game_state.turns_this_round[0] % 2
        player_mask = jax.nn.one_hot(current_player, self.players)  # [2]

        # Determine if current player forfeited
        current_action_oh = jnp.sum(oh_actions * player_mask[:, None], axis=0)  # [3]
        forfeited = jnp.any(current_action_oh * forfeit_oh)

        # Chip update: DARE raises if bets equal, matches if unequal
        max_chips = jnp.max(game_state.current_chips)
        bets_equal_before = jnp.all(jnp.isclose(
            game_state.current_chips[0].astype(jnp.float32),
            game_state.current_chips[1].astype(jnp.float32)))

        raise_chips = (game_state.current_chips + player_mask).astype(jnp.int16)
        match_chips = (game_state.current_chips * (1 - player_mask) +
                       player_mask * max_chips).astype(jnp.int16)
        dare_chips = jnp.where(bets_equal_before, raise_chips, match_chips)
        current_chips = jnp.where(forfeited, game_state.current_chips, dare_chips)

        new_bets_equal = current_chips[0] == current_chips[1]

        # Record action in history (shift action id down by 1 to exclude INVALID)
        oh_valid_action = jax.nn.one_hot(actions[current_player] - 1, self.num_actions - 1)  # [2]
        this_turn_played = oh_valid_action * oh_turn[..., None]  # [4, 2]
        action_history = game_state.action_history + this_turn_played

        # Trigger round-2 chance node when round 1 ends with both players having dared
        play_chance = jnp.logical_and(
            jnp.logical_and(game_state.current_round == 0,
                            game_state.turns_this_round[0] >= 1),
            jnp.logical_and(~forfeited, new_bets_equal))

        # Terminal: forfeit, or both dared in round 2 making bets equal
        terminal = jnp.logical_or(
            forfeited,
            jnp.logical_and(
                jnp.logical_and(game_state.current_round > 0,
                                game_state.turns_this_round[0] >= 1),
                new_bets_equal))
        terminal = jnp.squeeze(terminal)

        # Winner: strength comparison (or forfeiting player loses)
        tie = game_state.strengths[0] == game_state.strengths[1]
        winner_by_strength = jnp.argmax(game_state.strengths)
        winner = jnp.where(forfeited, 1 - current_player, winner_by_strength)

        # Reward normalised to [-1, 1]; loser's chips / max_reward
        reward = jnp.where(
            terminal,
            jnp.where(
                jnp.logical_and(tie, ~forfeited),
                jnp.array(0.0),
                ((1 - 2 * winner.astype(jnp.float32)) *
                 current_chips[1 - winner].astype(jnp.float32)) / self.max_reward),
            jnp.array(0.0))

        # Propagate terminal from already-terminal states
        terminal = jnp.logical_or(game_state.terminal, terminal)

        # Legal actions for next player; forfeit only legal when facing a raise
        new_acting_legals = jnp.ones(self.num_actions) - self.invalid_action_mask
        new_acting_legals = jnp.where(new_bets_equal,
                                       new_acting_legals - forfeit_oh,
                                       new_acting_legals)

        # Reset turns_this_round on round transition (-1 + 1 trick mirrors Leduc)
        turns_this_round = jnp.where(play_chance, -1, game_state.turns_this_round)
        next_player = (turns_this_round + 1) % 2

        new_legals = jnp.where(
            next_player == 0,
            jnp.stack([new_acting_legals, self.invalid_action_mask], axis=0),
            jnp.stack([self.invalid_action_mask, new_acting_legals], axis=0))
        new_legals = jnp.where(play_chance, jnp.ones_like(new_legals), new_legals)

        new_game_state = DuelGameState(
            action_history=action_history,
            strengths=game_state.strengths.astype(jnp.int16),
            strength_boost=game_state.strength_boost.astype(jnp.int16),
            current_chips=current_chips.astype(jnp.int16),
            turns_this_round=(turns_this_round + 1).astype(jnp.int16),
            current_round=game_state.current_round.astype(jnp.int16),
            terminal=terminal.astype(bool),
            turn=game_state.turn + 1,
            is_chance=play_chance)

        return new_game_state, terminal, reward, new_legals


class JaxStrengthDuelNoBoost(JaxStrengthDuel):
    """Variant without a strength-boost chance node between rounds.
    Supports a configurable number of rounds; strengths are fixed after the initial deal.
    Tensor shapes are smaller since strength_boost is not part of the public state."""

    def __init__(self, num_rounds: int = 2):
        super().__init__()
        self.num_rounds = num_rounds
        self.max_turns = num_rounds * 2      # 2 actions (P1 dare, P2 dare/forfeit) per round
        self.max_reward = num_rounds + 1     # chips start at 1, each round adds 1 to the pot

    def game_name(self):
        return "strength_duel_no_boost"

    def params_dict(self):
        return {'num_rounds': self.num_rounds}

    def max_trajectory_length(self):
        # num_rounds*2 action turns + 1 initial chance node + 1 terminal
        return self.max_turns + 2

    def depth_chance_outcomes(self, depth: int):
        if depth == 0:
            return self.initial_chance_outcomes
        return 1

    def depth_chance_valid_outcomes(self, depth: int):
        if depth == 0:
            return self.initial_chance_outcomes
        return 1

    def public_state_tensor_shape(self):
        # action history only — no boost one-hot
        return self.max_turns * (self.num_actions - 1)

    def information_state_tensor_shape(self):
        return self.players + self.num_strength_values + self.public_state_tensor_shape()

    def state_tensor_shape(self):
        return 2 * self.num_strength_values + self.public_state_tensor_shape()

    def observation_tensor_shape(self):
        return self.information_state_tensor_shape()

    @functools.partial(jax.jit, static_argnums=(0,))
    def get_info(self, game_state: DuelGameState):
        strengths_oh = jax.nn.one_hot(game_state.strengths, self.num_strength_values)  # [2, 3]

        public_state = game_state.action_history.ravel()  # [8]
        public_state = jnp.where(game_state.is_chance, jnp.zeros_like(public_state), public_state)

        p1_player = jax.nn.one_hot(0, 2)
        p1_info = jnp.concatenate([p1_player, strengths_oh[0], public_state], axis=0)  # [13]
        p1_info = jnp.where(game_state.is_chance, jnp.zeros_like(p1_info), p1_info)

        p2_info = jnp.concatenate([1 - p1_player, strengths_oh[1], public_state], axis=0)  # [13]
        p2_info = jnp.where(game_state.is_chance, jnp.zeros_like(p2_info), p2_info)

        state_tensor = jnp.concatenate([strengths_oh.ravel(), public_state], axis=0)  # [14]
        state_tensor = jnp.where(game_state.is_chance, jnp.zeros_like(state_tensor), state_tensor)

        return state_tensor, p1_info, p2_info, public_state

    @functools.partial(jax.jit, static_argnums=(0,))
    def apply_action_no_chance(self, game_state: DuelGameState, actions: chex.Array):
        oh_actions = jax.nn.one_hot(actions, self.num_actions)  # [2, 3]
        oh_turn = jax.nn.one_hot(game_state.turn, self.max_turns)  # [4]
        forfeit_oh = jax.nn.one_hot(FORFEIT_ID, self.num_actions)  # [3]

        current_player = game_state.turns_this_round[0] % 2
        player_mask = jax.nn.one_hot(current_player, self.players)  # [2]

        current_action_oh = jnp.sum(oh_actions * player_mask[:, None], axis=0)  # [3]
        forfeited = jnp.any(current_action_oh * forfeit_oh)

        max_chips = jnp.max(game_state.current_chips)
        bets_equal_before = jnp.all(jnp.isclose(
            game_state.current_chips[0].astype(jnp.float32),
            game_state.current_chips[1].astype(jnp.float32)))

        raise_chips = (game_state.current_chips + player_mask).astype(jnp.int16)
        match_chips = (game_state.current_chips * (1 - player_mask) +
                       player_mask * max_chips).astype(jnp.int16)
        dare_chips = jnp.where(bets_equal_before, raise_chips, match_chips)
        current_chips = jnp.where(forfeited, game_state.current_chips, dare_chips)

        new_bets_equal = current_chips[0] == current_chips[1]

        oh_valid_action = jax.nn.one_hot(actions[current_player] - 1, self.num_actions - 1)  # [2]
        this_turn_played = oh_valid_action * oh_turn[..., None]  # [4, 2]
        action_history = game_state.action_history + this_turn_played

        # Transition to next round when bets equalise and it is not the final round
        both_acted = game_state.turns_this_round[0] >= 1
        round_transition = jnp.logical_and(
            jnp.logical_and(game_state.current_round < self.num_rounds - 1, both_acted),
            jnp.logical_and(~forfeited, new_bets_equal))

        # Terminal: forfeit at any point, or bets equalise in the final round
        terminal = jnp.logical_or(
            forfeited,
            jnp.logical_and(
                jnp.logical_and(game_state.current_round >= self.num_rounds - 1, both_acted),
                new_bets_equal))
        terminal = jnp.squeeze(terminal)

        tie = game_state.strengths[0] == game_state.strengths[1]
        winner_by_strength = jnp.argmax(game_state.strengths)
        winner = jnp.where(forfeited, 1 - current_player, winner_by_strength)

        reward = jnp.where(
            terminal,
            jnp.where(
                jnp.logical_and(tie, ~forfeited),
                jnp.array(0.0),
                ((1 - 2 * winner.astype(jnp.float32)) *
                 current_chips[1 - winner].astype(jnp.float32)) / self.max_reward),
            jnp.array(0.0))

        terminal = jnp.logical_or(game_state.terminal, terminal)

        new_acting_legals = jnp.ones(self.num_actions) - self.invalid_action_mask
        new_acting_legals = jnp.where(new_bets_equal,
                                       new_acting_legals - forfeit_oh,
                                       new_acting_legals)

        turns_this_round = jnp.where(round_transition, -1, game_state.turns_this_round)
        next_player = (turns_this_round + 1) % 2

        new_legals = jnp.where(
            next_player == 0,
            jnp.stack([new_acting_legals, self.invalid_action_mask], axis=0),
            jnp.stack([self.invalid_action_mask, new_acting_legals], axis=0))

        new_current_round = jnp.where(
            round_transition,
            (game_state.current_round + 1).astype(jnp.int16),
            game_state.current_round).astype(jnp.int16)

        new_game_state = DuelGameState(
            action_history=action_history,
            strengths=game_state.strengths.astype(jnp.int16),
            strength_boost=game_state.strength_boost.astype(jnp.int16),
            current_chips=current_chips.astype(jnp.int16),
            turns_this_round=(turns_this_round + 1).astype(jnp.int16),
            current_round=new_current_round,
            terminal=terminal.astype(bool),
            turn=game_state.turn + 1,
            is_chance=jnp.array(False))

        return new_game_state, terminal, reward, new_legals


def main():
    game = JaxStrengthDuel()

    def _tree_walk(state: DuelGameState, legals, terminal, depth=0):
        legals = np.asarray(legals)
        print(f"depth={depth} chips={np.asarray(state.current_chips)} "
              f"strengths={np.asarray(state.strengths)} "
              f"boost={int(state.strength_boost)} "
              f"round={int(state.current_round)} "
              f"terminal={bool(terminal)}")
        if terminal:
            return
        if game.is_chance(state):
            outcomes, probs = game.get_outcomes_and_probs(state)
            for outcome, prob in zip(outcomes, probs):
                if prob < 1e-5:
                    continue
                new_state, new_terminal, reward, new_legals = game.apply_action(state, outcome)
                _tree_walk(new_state, new_legals, False, depth=depth + 1)
            return
        for a1i, a1 in enumerate(legals[0]):
            if a1 < 0.5:
                continue
            for a2i, a2 in enumerate(legals[1]):
                if a2 < 0.5:
                    continue
                joint_action = jnp.array([a1i, a2i])
                new_state, new_terminal, reward, new_legals = game.apply_action(state, joint_action)
                _tree_walk(new_state, new_legals, new_terminal, depth=depth + 1)

    init_state, init_legals = game.initialize_structures()
    _tree_walk(init_state, init_legals, False)


if __name__ == "__main__":
    main()
