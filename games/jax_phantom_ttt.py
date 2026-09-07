import jax
import chex
import jax.numpy as jnp
import numpy as np

import functools
from games.jax_game import JaxGame, GameState, InformationType

#Dummy action played by the player who is not acting
INVALID_ID = 0
NUM_CELLS = 9


@chex.dataclass(frozen=True)
class PhantomTTTState(GameState):
    board: chex.Array #[Player, Cell] the actual pieces, not observed by the opponent
    attempt_history: chex.Array #[Player, Own turn, Cell] one hot of the attempted cell
    attempt_results: chex.Array #[Player, Own turn] 1 if the attempt placed the symbol
    own_turns: chex.Array #[Player] how many attempts the player has made so far
    current_player: chex.Array #Which player attempts next
    placements: chex.Array #How many attempts succeeded in total, common knowledge
    terminal: chex.Array #Remember this to make sure that the state is correctly marked as terminal,
    #when playing additional actions in terminal state
    turn: chex.Array #Global attempt index, kept for bookkeeping only


class JaxPhantomTTT(JaxGame):
  """Phantom Tic-Tac-Toe, the classical variant where a player who attempts a cell
  occupied by the opponent is denied and moves again.

  A player observes their own pieces and the cells they were denied, but never the
  rest of the opponent's board. Attempting one's own cell is illegal, as the player
  knows about it. Unlike in the classical rules, re-attempting a cell that was already
  denied to you is illegal as well, which bounds the depth of the game.

  This is a turn based game embedded in the simultaneous move interface the same way
  Leduc is, the player who is not acting has only the dummy INVALID action legal."""

  def __init__(self) -> None:
    #Invalid action and an attempt of each of the cells
    self.num_actions = NUM_CELLS + 1
    self.players = 2
    self.board_size = 3
    #Player 1 places 5 symbols and can be denied at most 4 times
    self.max_own_turns = 9
    #The longest sequence of attempts from the root, brute forced over the whole game
    self.max_turns = 17
    self.invalid_action_mask = jax.nn.one_hot(INVALID_ID, self.num_actions)
    self.lines = jnp.asarray(self._win_line_masks(), dtype=jnp.float32)

  def _win_line_masks(self) -> np.ndarray:
    """Cell masks of all the winning lines, rows, columns and both diagonals."""
    cells = np.arange(NUM_CELLS).reshape(self.board_size, self.board_size)
    lines = [cells[i, :] for i in range(self.board_size)]
    lines += [cells[:, i] for i in range(self.board_size)]
    lines += [np.diag(cells), np.diag(np.fliplr(cells))]
    masks = np.zeros((len(lines), NUM_CELLS), dtype=np.float32)
    for i, line in enumerate(lines):
      masks[i, line] = 1.0
    return masks

  def game_name(self):
    return "phantom_ttt"

  def params_dict(self):
    return {}

  def information_type(self):
    return InformationType.IIG

  def num_players(self):
    return self.players

  def num_distinct_actions(self):
    return self.num_actions

  def max_trajectory_length(self):
    # The longest sequence of attempts plus the terminal state the last of them reaches
    return self.max_turns + 1

  def public_state_tensor_shape(self):
    # One hot encoded amount of the successful placements
    # One hot encoded player to move
    return (NUM_CELLS + 1) + self.players

  def information_state_tensor_shape(self):
    # One hot encoded receiving player
    # One hot encoded cell attempted at each of the player's own turns
    # A bit per own turn, whether the attempt placed the symbol or was denied
    return self.players + self.max_own_turns * NUM_CELLS + self.max_own_turns + self.public_state_tensor_shape()

  def observation_tensor_shape(self):
    return self.information_state_tensor_shape()

  def state_tensor_shape(self):
    # Pieces of both of the players
    # Private attempt history and its results of both of the players
    return (self.players * NUM_CELLS + self.players * self.max_own_turns * NUM_CELLS
            + self.players * self.max_own_turns + self.public_state_tensor_shape())

  @functools.partial(jax.jit, static_argnums=(0))
  def legal_actions(self, game_state: PhantomTTTState):
    """Legal actions of both of the players. Every attempt makes the cell illegal for
    the acting player from then on, either because they now own it, or because they
    were told that the opponent does. So the legal cells are exactly the ones the
    acting player has never attempted."""
    acting = game_state.current_player
    legal_cells = 1 - jnp.sum(game_state.attempt_history[acting], axis=0)
    acting_legals = jnp.concatenate([jnp.zeros(1), legal_cells], axis=0)
    legals = jnp.where(acting == 0,
                       jnp.stack([acting_legals, self.invalid_action_mask], axis=0),
                       jnp.stack([self.invalid_action_mask, acting_legals], axis=0))
    #In the absorbing terminal state a player can have all of the cells illegal, so
    # we let both of them play only the dummy action. The mask must never be all zeros,
    # the trajectory sampling samples an action from it in every step.
    terminal_legals = jnp.tile(self.invalid_action_mask[None, ...], (self.players, 1))
    return jnp.where(game_state.terminal, terminal_legals, legals)

  @functools.partial(jax.jit, static_argnums=(0))
  def initialize_structures(self):
    game_state = PhantomTTTState(board=jnp.zeros((self.players, NUM_CELLS)),
                            attempt_history=jnp.zeros((self.players, self.max_own_turns, NUM_CELLS)),
                            attempt_results=jnp.zeros((self.players, self.max_own_turns)),
                            own_turns=jnp.zeros(self.players, dtype=jnp.int32),
                            current_player=jnp.array(0, dtype=jnp.int32),
                            placements=jnp.array(0, dtype=jnp.int32),
                            terminal=jnp.array(False),
                            turn=jnp.array(0, dtype=jnp.int32))
    return game_state, self.legal_actions(game_state)

  # State Tensor  -> Pieces [Player, Cell], Attempts [Player, Own turn, Cell], Placed [Player, Own turn], Public
  # Iset tensor   -> Observing Player, Attempts [Own turn, Cell], Placed [Own turn], Public
  # Public tensor -> Placements [Placement], Player to move [Player]
  @functools.partial(jax.jit, static_argnums=(0))
  def get_info(self, game_state: PhantomTTTState):
    #The amount of the successful placements is common knowledge, because the turn passes
    # only on a success and so the successes strictly alternate. The amount of the attempts
    # is not, a player never learns how many times the opponent was denied, hence it is
    # deliberately encoded in none of the tensors.
    placements_oh = jax.nn.one_hot(game_state.placements, NUM_CELLS + 1)
    to_move_oh = jax.nn.one_hot(game_state.current_player, self.players)
    public_state_tensor = jnp.concatenate([placements_oh, to_move_oh], axis=0)

    p1_player = jax.nn.one_hot(0, self.players)

    p1_infoset_tensor = jnp.concatenate([p1_player,
                                         game_state.attempt_history[0].ravel(),
                                         game_state.attempt_results[0],
                                         public_state_tensor], axis=0)
    p2_infoset_tensor = jnp.concatenate([1 - p1_player,
                                         game_state.attempt_history[1].ravel(),
                                         game_state.attempt_results[1],
                                         public_state_tensor], axis=0)

    state_tensor = jnp.concatenate([game_state.board.ravel(),
                                    game_state.attempt_history.ravel(),
                                    game_state.attempt_results.ravel(),
                                    public_state_tensor], axis=0)

    return state_tensor, p1_infoset_tensor, p2_infoset_tensor, public_state_tensor

  @functools.partial(jax.jit, static_argnums=(0))
  def apply_action(self, game_state: PhantomTTTState, actions: chex.Array):
    acting = game_state.current_player
    action = actions[acting]
    #The dummy action gives an all zero one hot, so it can never touch the board
    cell_oh = jax.nn.one_hot(action - 1, NUM_CELLS)
    #Steps taken in the absorbing terminal state and the dummy joint action, which the
    # chance branch of the replay buffer traces even for deterministic games, must not
    # change anything.
    active = jnp.logical_and(~game_state.terminal, action != INVALID_ID)

    acting_oh = jax.nn.one_hot(acting, self.players)
    own_turn_oh = jax.nn.one_hot(game_state.own_turns[acting], self.max_own_turns)
    #The attempt is denied exactly when the opponent occupies the cell. The acting player
    # can never attempt their own cell, that one is masked out of the legal actions.
    placed = jnp.sum(cell_oh * game_state.board[1 - acting]) == 0

    board = game_state.board + acting_oh[..., None] * cell_oh[None, ...] * placed
    attempt_history = game_state.attempt_history + acting_oh[..., None, None] * own_turn_oh[None, ..., None] * cell_oh[None, None, ...]
    attempt_results = game_state.attempt_results + acting_oh[..., None] * own_turn_oh[None, ...] * placed
    own_turns = game_state.own_turns + acting_oh.astype(game_state.own_turns.dtype)
    placements = game_state.placements + placed.astype(game_state.placements.dtype)
    #This is the whole turn based rule, a denied attempt keeps the turn with the same player
    current_player = jnp.where(placed, 1 - acting, acting)

    line_completed = jnp.any(jnp.sum(self.lines * board[acting], axis=-1) >= self.board_size)
    won = jnp.logical_and(placed, line_completed)
    terminal = jnp.logical_or(won, placements >= NUM_CELLS)
    #Reward from the perspective of player 1. A full board without a line is a draw.
    reward = jnp.where(won, 1 - 2 * acting, 0).astype(jnp.float32)

    new_game_state = PhantomTTTState(board=board,
                            attempt_history=attempt_history,
                            attempt_results=attempt_results,
                            own_turns=own_turns,
                            current_player=current_player.astype(jnp.int32),
                            placements=placements,
                            terminal=terminal,
                            turn=game_state.turn + 1)
    new_game_state = jax.tree_util.tree_map(lambda new, old: jnp.where(active, new, old), new_game_state, game_state)
    reward = jnp.where(active, reward, 0.0)

    return new_game_state, new_game_state.terminal, reward, self.legal_actions(new_game_state)
