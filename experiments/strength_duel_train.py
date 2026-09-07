
from games.jax_strength_duel import JaxStrengthDuel, JaxStrengthDuelNoBoost
from experiments.joint_train import train_nash_dreamer
from experiments.rnad_train import train_rnad
from experiments.mmd_train import train_mmd
from experiments.ppo_train import train_ppo
from experiments.parsing_utils import prepare_experiment_parser


parser = prepare_experiment_parser()
parser.add_argument('--no_boost', action='store_true', help='Use the no-boost version of the game')
parser.add_argument('--num_rounds', type=int, default=2,
                    help='Number of betting rounds (only used with --no_boost)')


def main():
  args = parser.parse_args()
  if args.no_boost:
    game = JaxStrengthDuelNoBoost(num_rounds=args.num_rounds)
  else:
    game = JaxStrengthDuel()
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
