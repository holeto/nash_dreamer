from games.jax_point_card_matching import PointCardMatching, PointCardMatchingStochastic
from experiments.joint_train import train_nash_dreamer
from experiments.rnad_train import train_rnad
from experiments.mmd_train import train_mmd
from experiments.ppo_train import train_ppo
from experiments.parsing_utils import prepare_experiment_parser

parser = prepare_experiment_parser()
# Game parameters
parser.add_argument("--num_cards", type=int, default=3, help="Number of cards of the game. Should be at least 3")
parser.add_argument("--stochastic", action="store_true", help="Whether to use the point card matching with chance node, or the one without.")
parser.add_argument("--chance_turn_before_terminal", type=int, default=1, help="How many turns before terminal does the chance turn happen. Only for stochastic point card matching.")



def main():
  args = parser.parse_args()
  if args.stochastic:
    game = PointCardMatchingStochastic(num_cards=args.num_cards, chance_turn_before_terminal=args.chance_turn_before_terminal)
  else:
    game = PointCardMatching(num_cards=args.num_cards)
  if args.experiment_type == 'nash_dreamer':
    train_nash_dreamer(args, game)
  elif args.experiment_type == 'mmd':
    train_mmd(args, game)
  elif args.experiment_type == 'ppo':
    train_ppo(args, game)
  else:
    train_rnad(args, game)

if __name__ == "__main__":
  main()