import jax
import jax.numpy as jnp
from envs.jax_game import JaxGame, GameState, InformationType
import chex

import functools
from typing import Union
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

@chex.dataclass(frozen=True)
class BattleshipsState(GameState):
    ships: chex.Array # [Pl, Y, X] -> It is not binary! First ship has value 1, second 2, etc.
    shots: chex.Array # [Pl, Y, X] -> It is not binary! No shot has value 0, miss has value 1, hit has value 2, sunken has value 3

    action_history: chex.Array # [Pl, Turn, YX one-hot]
    ship_hits: chex.Array # [Pl, Ships] -> Player 0 has values whether it hit the ship of player 1.
    shots_order: chex.Array # [Pl, Shots, 2] -> row/col of each shot in order (indexed by shot number)

    phase: int  # 0: placement phase, 1: shooting phase
    turn: int

class JaxBattleships(JaxGame):
  def __init__(self, board_shape:Union[tuple[int, int], int] = (10, 10)  , ship_sizes: list[int] = [5, 4, 3, 3, 2]):
    if isinstance(board_shape, int):
      board_shape = (board_shape, board_shape)
    self.height = board_shape[0]
    self.width = board_shape[1]
    assert self.height > 0, "Height must be positive"
    assert self.width > 0, "Width must be positive"
    self.board_shape = board_shape
    self.board_size = self.height * self.width
    for ship_size in ship_sizes:
      assert ship_size > 0, "Ship size must be positive"
      assert ship_size <= self.height or ship_size <= self.width, "Ship size must be less than or equal to the height or width of the board"
    self.ship_sizes = ship_sizes
    self.ship_sizes_jax = jnp.array(ship_sizes)
    self.max_ship = jnp.max(self.ship_sizes_jax)

  def game_name(self):
    return "battleships"

  def params_dict(self):
    return {"board_h": self.height, "board_w": self.width, "num_ships": len(self.ship_sizes)}

  def information_type(self):
    return InformationType.IIG

  def num_players(self):
    return 2

  def max_trajectory_length(self):
    return len(self.ship_sizes) + self.board_size

  def num_distinct_actions(self):
    # Actions are:
    # 0 to board_size - 1: Shoot
    # board_size to 2*board_size - 1: Place ship horizontally
    # 2*board_size to 3*board_size - 1: Place ship vertically
    return 3 * self.height * self.width

  def state_tensor_shape(self):
    num_ships = len(self.ship_sizes)
    return (
      2 +  # Phase
      2 * self.board_size +  # Ships (both players, clipped to binary)
      2 * 3 * self.board_size +  # Shots one-hot (miss/hit/sunken per player)
      2 * (num_ships + self.board_size) * (self.height + self.width) +  # Action history
      num_ships +  # Placed ships
      2 * num_ships  # Sunken ships per player
    )

  def information_state_tensor_shape(self):
    return (2 + # Player
      2 + # Phase
      self.board_size + # Ships
      2 * 3 * self.board_size + # Shots (Miss, Hit, Sunken) [Player * Shot_result * Board Size]
      (len(self.ship_sizes) + self.board_size) * (self.board_shape[0] + self.board_shape[1]) + # Player actions
      self.board_size * (self.board_shape[0] + self.board_shape[1]) + # Opponent actions
      len(self.ship_sizes) + # Placed ships
      2 * len(self.ship_sizes) # Sunken ships per player
    )

  def observation_tensor_shape(self):
    return self.information_state_tensor_shape()

  def public_state_tensor_shape(self):

    return (
      2 + # Phase
      2 * 3 * self.board_size + # Shots (Miss, Hit, Sunken) [Player * Shot_result * Board Size]
      2 * self.board_size * (self.board_shape[0] + self.board_shape[1]) + # Shot actions
      len(self.ship_sizes) + # Placed ships
      2 * len(self.ship_sizes) # Sunken ships per player
    )

  @functools.partial(jax.jit, static_argnums=(0,))
  def initialize_structures(self):
    # Initialize empty game state
    state = BattleshipsState(
      ships=jnp.zeros((2,) + self.board_shape, dtype=jnp.int32), #  ships
      shots=jnp.zeros((2,) + self.board_shape, dtype=jnp.int32), #  shots

      action_history = jnp.zeros((2, len(self.ship_sizes) + self.board_shape[0] * self.board_shape[1], self.board_shape[0] + self.board_shape[1]), dtype=jnp.int32), # action history. For each player, each size and each possible position

      ship_hits=jnp.zeros((2, len(self.ship_sizes)), dtype=jnp.int32), # ship hits
      shots_order=jnp.zeros((2, self.board_size, 2), dtype=jnp.int32), # shot order

      phase=0, # phase
      turn=0,
    )

    # Legal actions are all placement and shooting actions initially
    legal_horizontal_placement = jnp.ones(self.board_shape, dtype=jnp.int32)

    # We only use horizontal placement if the ship has length 1
    if self.ship_sizes[0] == 1:
      legal_vertical_placement = jnp.zeros(self.board_shape, dtype=jnp.int32)

    if self.ship_sizes[0] > 1:
      legal_vertical_placement = jnp.ones(self.board_shape, dtype=jnp.int32)
      legal_horizontal_placement = legal_horizontal_placement.at[:,  -self.ship_sizes[0] + 1:].set(0)
      legal_vertical_placement = legal_vertical_placement.at[ -self.ship_sizes[0] + 1:, :].set(0)


    legal_actions = jnp.concatenate([jnp.zeros((self.board_shape[0] * self.board_shape[1]), dtype=jnp.int32), legal_horizontal_placement.flatten(), legal_vertical_placement.flatten()], axis=0)

    legal_actions = jnp.stack([legal_actions, legal_actions], axis=0)
    return state, legal_actions

  # TODO: Do we want to swap players for player 2, so that the iset structure is symmetric between the players?.
  @functools.partial(jax.jit, static_argnums=(0,))
  def get_info(self, game_state: BattleshipsState):

    # Convert game state to tensors for JAX operations
    # We add a 1 to the beginning so that it is not just zero-tensor in the initial state

    game_phase = jax.nn.one_hot(game_state.phase, 2, axis=0)
    miss_hits_sunken = jax.nn.one_hot(game_state.shots - 1, 3, axis=0).flatten()
    sunken_ships = jnp.where(game_state.ship_hits == self.ship_sizes_jax[None, :], 1, 0).flatten()

    placed_ships = jnp.sum(game_state.action_history[0, :len(self.ship_sizes)], axis=1)
    placed_ships = jnp.clip(placed_ships, 0, 1)
    p1_player = jax.nn.one_hot(0, 2)

    ships = jnp.clip(game_state.ships, 0, 1)


    state_tensor = jnp.concatenate([
      game_phase,
      ships.flatten(),
      miss_hits_sunken,
      game_state.action_history.flatten(),
      placed_ships,
      sunken_ships
    ])

    # Information sets for both players
    p1_iset = jnp.concatenate([
      p1_player,
      game_phase,
      ships[0].flatten(),
      miss_hits_sunken,
      game_state.action_history[0].flatten(),
      game_state.action_history[1, len(self.ship_sizes):].flatten(),
      placed_ships,
      sunken_ships
    ])

    p2_iset = jnp.concatenate([
      1 - p1_player,
      game_phase,
      ships[1].flatten(),
      miss_hits_sunken,
      game_state.action_history[1].flatten(),
      game_state.action_history[0, len(self.ship_sizes):].flatten(),
      placed_ships,
      sunken_ships
    ])

    # Public state (shots and sunk ships)
    public_state = jnp.concatenate([
      game_phase,
      miss_hits_sunken,
      game_state.action_history[:, len(self.ship_sizes):].flatten(),
      placed_ships,
      sunken_ships
    ])

    return state_tensor, p1_iset, p2_iset, public_state

  @functools.partial(jax.jit, static_argnums=(0,))
  def apply_action(self, game_state: BattleshipsState, actions):
    turn = game_state.turn

    def find_legal_actions(shots, ships):
      shooting_legals = jnp.where(shots.flatten() == 0, 1, 0)
      # Get the size of the next ship to be placed

      next_ship_size = jnp.take(self.ship_sizes_jax, turn + 1, fill_value=1)

      # For each forbidden position (occupied or adjacent), mark it and positions
      # to the left (up to next_ship_size) as illegal.
      def mark_illegal_horizontal_placement(legal_mask, occupied_mask):
        for i in range(self.max_ship):
          # Shift occupied mask to the right by i positions and mark those positions as illegal
          shifted = jnp.pad(occupied_mask, ((0, 0), (0, i)), mode='constant')[:, i:]
          new_legal_mask = jnp.where(shifted, 0, legal_mask)
          legal_mask = jnp.where(i < next_ship_size, new_legal_mask, legal_mask)
        return legal_mask


      def mark_illegal_vertical_placement(legal_mask, occupied_mask):
        for i in range(self.max_ship):
          # Shift occupied mask to the right by i positions and mark those positions as illegal
          shifted = jnp.pad(occupied_mask, ((0, i), (0, 0)), mode='constant')[i:, :]
          new_legal_mask = jnp.where(shifted, 0, legal_mask)
          legal_mask = jnp.where(i < next_ship_size, new_legal_mask, legal_mask)
        return legal_mask

      # Start with all positions being legal
      legal_horizontal_placement = jnp.ones(self.board_shape, dtype=jnp.int32) * jnp.arange(self.board_shape[1])[None, :]

      legal_horizontal_placement = jnp.where(legal_horizontal_placement < self.board_shape[1] - next_ship_size + 1, 1, 0)

      legal_vertical_placement = jnp.ones(self.board_shape, dtype=jnp.int32) * jnp.arange(self.board_shape[0])[:, None]

      legal_vertical_placement = jnp.where(legal_vertical_placement < self.board_shape[0] - next_ship_size + 1, 1, 0)

      # Create a forbidden mask: occupied cells and their 8-directional neighbors.
      # Ships cannot be placed adjacent to each other (including diagonals).
      def get_forbidden_mask(ships):
        occupied = ships > 0
        h, w = self.board_shape
        padded = jnp.pad(occupied, 1, mode="constant", constant_values=False)
        result = jnp.zeros((h, w), dtype=jnp.bool_)
        for di in range(3):
          for dj in range(3):
            result = result | padded[di:di+h, dj:dj+w]
        return result

      forbidden = get_forbidden_mask(ships)
      horizontal_legals = mark_illegal_horizontal_placement(legal_horizontal_placement, forbidden)
      vertical_legals = mark_illegal_vertical_placement(legal_vertical_placement, forbidden)

      placing_legals = jnp.concatenate([horizontal_legals.flatten(), vertical_legals.flatten()], axis=0)

      legal_actions = jnp.zeros(self.num_distinct_actions(), dtype=jnp.int32)


      legal_actions = jnp.where(turn >= len(self.ship_sizes) - 1, legal_actions.at[:self.board_size].set(shooting_legals), legal_actions.at[self.board_size:].set(placing_legals))

      return legal_actions

    # Define a helper function to process a single player's action
    def apply_action_player(
      action_history: chex.Array,
      pl_ships: chex.Array,
      opp_ships: chex.Array,
      pl_shots: chex.Array,
      pl_ship_hits: chex.Array,
      pl_shots_order: chex.Array,
      action: chex.Array):

      ship_size = jnp.take(self.ship_sizes_jax, turn, fill_value=0)

      action_type = action // self.board_size
      action_idx = action % self.board_size
      y = action_idx // self.height
      x = action_idx % self.width

      y_one_hot = jax.nn.one_hot(y, self.height, dtype=jnp.int32)
      x_one_hot = jax.nn.one_hot(x, self.width, dtype=jnp.int32)

      one_hot_y_x = jnp.concatenate([y_one_hot, x_one_hot], axis=-1)
      new_action_history = action_history.at[turn, :].set(one_hot_y_x)
      # TODO: Could we do this better?
      # Place horizontal ship using dynamic_update_slice
      def place_horizontal_ship(ships, y, x, size, value):
        # Create a ship segment of the right size
        ship_segment = jnp.arange(ships.shape[1])
        new_cells = jnp.where(jnp.logical_and(ship_segment >= x, ship_segment < x + size), value, 0)
        # Merge with existing row to preserve previously placed ships
        merged_row = jnp.where(new_cells > 0, new_cells, ships[y])
        return ships.at[y].set(merged_row)

      def place_vertical_ship(ships, y, x, size, value):
        # Create a ship segment of the right size
        ship_segment = jnp.arange(ships.shape[0])
        new_cells = jnp.where(jnp.logical_and(ship_segment >= y, ship_segment < y + size), value, 0)
        # Merge with existing column to preserve previously placed ships
        merged_col = jnp.where(new_cells > 0, new_cells, ships[:, x])
        return ships.at[:, x].set(merged_col)


      new_ships = jnp.where(action_type == 1,
        place_horizontal_ship(pl_ships, y, x, ship_size, turn+1),
        pl_ships
      )

      # Place vertical ship
      new_ships = jnp.where(action_type == 2,
        place_vertical_ship(pl_ships, y, x, ship_size, turn+1),
        new_ships
      )
      ship_type = opp_ships[y, x] - 1 # -1 if no ship


      # TODO: This can be done better
      new_shots = jnp.where(ship_type >= 0, pl_shots.at[y, x].set(2), pl_shots.at[y, x].set(1))

      new_ship_hits = jax.nn.one_hot(ship_type, len(self.ship_sizes), dtype=jnp.int32) + pl_ship_hits

      ship_sunken = jnp.where(ship_type >= 0, new_ship_hits[ship_type] == self.ship_sizes_jax[ship_type], False)

      new_shots = jnp.where(ship_sunken, new_shots.at[y, x].set(3), new_shots)

      new_shots = jnp.where(action_type == 0, new_shots, pl_shots)
      new_ship_hits = jnp.where(action_type == 0, new_ship_hits, pl_ship_hits)

      # Record shot order: index into shots_order by (turn - num_ships), clamped to valid range
      shot_turn = jnp.maximum(turn - len(self.ship_sizes), 0)
      new_shots_order = pl_shots_order.at[shot_turn, :].set(jnp.array([y, x]))
      new_shots_order = jnp.where(action_type == 0, new_shots_order, pl_shots_order)

      return new_ships, new_shots, new_action_history, new_ship_hits, new_shots_order, find_legal_actions(new_shots, new_ships)


    apply_action_player_vmap = jax.vmap(apply_action_player, in_axes=0, out_axes=0)

    new_ships, new_shots, new_action_history, new_ship_hits, new_shots_order, new_legal_actions = apply_action_player_vmap(game_state.action_history, game_state.ships, jnp.flipud(game_state.ships), game_state.shots, game_state.ship_hits, game_state.shots_order, actions)

    sunked_all_ships = jnp.all(new_ship_hits >= self.ship_sizes_jax, axis=-1)

    reward = jnp.where(sunked_all_ships, 1.0, 0.0)
    reward = reward[0] - reward[1]


    return BattleshipsState(
      ships=new_ships,
      shots=new_shots,
      action_history=new_action_history,
      ship_hits=new_ship_hits,
      shots_order=new_shots_order,
      phase=jnp.where(turn >= len(self.ship_sizes) - 1, 1, 0),
      turn=turn + 1,
    ), jnp.any(sunked_all_ships), reward, new_legal_actions

  def make_canvas(self):
    """Create a canvas with two boards side by side."""
    fig, axes = plt.subplots(
      1, 2, figsize=(2 * self.width + 2, self.height + 1), constrained_layout=True
    )

    patches = [[], []]
    for player, ax in enumerate(axes):
      ax.set_xlim(-0.5, self.width - 0.5)
      ax.set_ylim(-0.5, self.height - 0.5)
      ax.set_aspect("equal")
      ax.invert_yaxis()
      ax.set_title(f"Player {player}")
      ax.set_xticks(range(self.width))
      ax.set_yticks(range(self.height))

      for r in range(self.height):
        for c in range(self.width):
          patch = Rectangle(
            (c - 0.5, r - 0.5),
            1,
            1,
            facecolor="blue",
            edgecolor="white",
            linewidth=1,
          )
          ax.add_patch(patch)
          patches[player].append(patch)

    return fig, axes, patches

  def draw_canvas(self, canvas, state: BattleshipsState):
    """
    Update the canvas with current state.

    Colors:
    - Blue: water (no ship, not shot)
    - Black: ship (not yet hit)
    - Red: ship cell that was hit or sunk
    - Dark blue: water cell that was shot (miss)
    """
    fig, axes, patches = canvas

    for player in range(2):
      opponent = 1 - player
      ships = state.ships[player]       # integer grid; > 0 means ship
      opp_shots = state.shots[opponent] # integer grid; 1=miss, 2=hit, 3=sunken

      for idx, patch in enumerate(patches[player]):
        r, c = idx // self.width, idx % self.width
        is_ship = ships[r, c] > 0
        is_hit = opp_shots[r, c] >= 2  # hit or sunken

        if is_ship and is_hit:
          color = "red"
        elif is_ship:
          color = "black"
        elif opp_shots[r, c] == 1:
          color = "darkblue"
        else:
          color = "blue"

        patch.set_facecolor(color)

    num_ships = len(self.ship_sizes)
    in_placement = state.turn < num_ships
    if in_placement:
      phase_str = f"Placement: Ship {state.turn + 1}/{num_ships}"
    else:
      shooting_turn = state.turn - num_ships
      phase_str = f"Shooting: Turn {shooting_turn}"

    fig.suptitle(phase_str)

  def close_canvas(self, canvas):
    fig, axes, patches = canvas
    plt.close(fig)
