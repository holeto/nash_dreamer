
from games.jax_leduc import JaxLeduc, JaxLeducRound1
from experiments.joint_train import train_nash_dreamer
from experiments.rnad_train import train_rnad
from experiments.parsing_utils import prepare_experiment_parser



parser = prepare_experiment_parser()
parser.add_argument("--one_round", action="store_true",
                    help="Use one-round Leduc (no public card / round 2)")
parser.add_argument("--max_raises", type=int, default=2,
                    help="Number of maximum raises per round in Leduc. Only implemented for the one round version.")


def main():
  args = parser.parse_args()
  game = JaxLeducRound1(max_raises=args.max_raises) if args.one_round else JaxLeduc()
  if args.experiment_type == 'nash_dreamer':
    train_nash_dreamer(args, game)
  else:
    train_rnad(args, game)
    
if __name__ == "__main__":
  main()