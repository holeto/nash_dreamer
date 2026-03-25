
from games.jax_leduc import JaxLeduc
from experiments.joint_train import train_nash_dreamer
from experiments.rnad_train import train_rnad
from experiments.parsing_utils import prepare_experiment_parser



parser = prepare_experiment_parser()


def main():
  args = parser.parse_args()
  game = JaxLeduc()
  if args.experiment_type == 'nash_dreamer':
    train_nash_dreamer(args, game)
  else:
    train_rnad(args, game)
    
if __name__ == "__main__":
  main()