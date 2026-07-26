from games.jax_battleships import JaxBattleships
from experiments.joint_train import train_nash_dreamer
from experiments.rnad_train import train_rnad
from experiments.mmd_train import train_mmd
from experiments.parsing_utils import prepare_experiment_parser



parser = prepare_experiment_parser()
parser.add_argument("--board_height", type=int, default=4, help="Height of the board")
parser.add_argument("--board_width", type=int, default=4, help="Width of the board")
parser.add_argument("--ship_sizes", type=str, default="2,1", help="Comma-separated list of ship sizes")


def main():
  args = parser.parse_args()
  ship_sizes = [int(s) for s in args.ship_sizes.split(",")]
  game = JaxBattleships(board_shape=(args.board_height, args.board_width), ship_sizes=ship_sizes)
  if args.experiment_type == 'nash_dreamer':
    train_nash_dreamer(args, game)
  elif args.experiment_type == 'mmd':
    train_mmd(args, game)
  else:
    train_rnad(args, game)

if __name__ == "__main__":
  main()
