from games.jax_goofspiel import JaxGoofspiel, JaxRandomGoofspiel
from experiments.joint_train import train_nash_dreamer
from experiments.rnad_train import train_rnad
from experiments.mmd_train import train_mmd
from experiments.ppo_train import train_ppo
from experiments.parsing_utils import prepare_experiment_parser



parser = prepare_experiment_parser()
parser.add_argument("--random", action="store_true", help="Whether to train a random variant of Goofspiel rather than descending.")
parser.add_argument("--num_cards", type=int, default=3, help="Number of cards of the Goofspiel game")
parser.add_argument("--observation_only", action="store_true", help="Use per-turn observations instead of full information-set tensors.")


def main():
  args = parser.parse_args()
  if args.random:
    game = JaxRandomGoofspiel(cards=args.num_cards)
  else:
    game = JaxGoofspiel(cards=args.num_cards, observation_only=args.observation_only)
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