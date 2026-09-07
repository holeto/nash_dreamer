import jax
import chex
import jax.numpy as jnp

import functools
from games.jax_rps import JaxRPS, RPSSTate

i8 = jnp.int8
f32 = jnp.float32


class JaxPerturbedRPS(JaxRPS):
  """Standard RPS with a single perturbed cell, to break the uniform equilibrium.

  The (Scissors, Paper) payoff is 2.0 instead of 1.0 (and its antisymmetric
  partner (Paper, Scissors) is -2.0), giving the payoff matrix

        R    P    S
    R   0   -1   +1
    P  +1    0   -2
    S  -1   +2    0

  The matrix stays antisymmetric, so the game value is still 0, but the unique
  equilibrium moves to (R, P, S) = (1/2, 1/4, 1/4). That is the point: in plain
  JaxRPS the zero-initialized policy head already plays the equilibrium, so
  exploitability cannot distinguish a converged run from an untrained one.

  Because p1 points now span [-2, 2] rather than [-1, 1], the points one-hot
  widens from 4 to 5 entries and the pre-game sentinel moves from -2 (a real
  payoff here) to -3, which one_hot renders as an all-zero block -- the same
  trick JaxStochasticRPS uses.
  """

  def game_name(self):
    return "perturbed_rps"

  def params_dict(self):
    return {}

  def information_state_tensor_shape(self):
    return 9  # 2 for player encoding, 2 for the terminal flag, 5 for player 1 points

  @functools.partial(jax.jit, static_argnums=(0))
  def initialize_structures(self):
    game_state = RPSSTate(terminal = jnp.array(False, dtype=bool),
                          p1_points = jnp.array(-3, dtype=i8))
    return game_state, jnp.ones((2, self.actions))

  @functools.partial(jax.jit, static_argnums=(0,))
  def get_info(self, game_state:RPSSTate):
    terminal_oh = jax.nn.one_hot(game_state.terminal.astype(int), 2)
    #The p1 points span range [-2, ..., 2], so we need to represent 5 values.
    #The pre-game sentinel -3 lands on index -1 and one_hot returns all zeros for it.
    p1_points_oh = jax.nn.one_hot(game_state.p1_points + 2, 5)
    state_tensor = jnp.concatenate([terminal_oh.ravel(), p1_points_oh.ravel()], axis=0)
    p1_infoset_tensor = jnp.concatenate([jax.nn.one_hot(0, 2), state_tensor], axis=0)
    p2_infoset_tensor = jnp.concatenate([jax.nn.one_hot(1, 2), state_tensor], axis=0)
    #We use just p1 points for state_tensor, public state tensor and p2_infoset_tensor as well, since they uniquely define p2 points as well
    return state_tensor, p1_infoset_tensor, p2_infoset_tensor, state_tensor

  @functools.partial(jax.jit, static_argnums=(0,))
  def apply_action(self, game_state:RPSSTate, actions):
    #A single action. The state after applying will always be terminal
    terminal = jnp.array(True, dtype=bool)
    action_difference = actions[0] - actions[1]
    # R: 0, P: 1, S: 2, with these, p1 wins at values -2, 1, loses at -1, 2 and its a tie at 0
    # sign(x) * (3 - 2 * abs(x)) produces these values, assuming sign(0) = 0, which it is in jnp
    p1_points = jnp.sign(action_difference) * (3 - (2 * jnp.abs(action_difference)))

    #Override the single perturbed cell and its antisymmetric partner
    scissors_paper = jnp.logical_and(actions[0] == self.moves["s"], actions[1] == self.moves["p"])
    paper_scissors = jnp.logical_and(actions[0] == self.moves["p"], actions[1] == self.moves["s"])
    p1_points = jnp.where(scissors_paper, 2, jnp.where(paper_scissors, -2, p1_points))

    legal_actions = jnp.ones((2, self.actions))
    new_game_state = RPSSTate(terminal = terminal,
                              p1_points = p1_points.astype(i8))

    return new_game_state, terminal, p1_points.astype(f32), legal_actions
