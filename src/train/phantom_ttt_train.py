from envs.jax_phantom_ttt import JaxPhantomTTT
from train.joint_train import train_nash_dreamer
from train.rnad_train import train_rnad
from train.mmd_train import train_mmd
from train.parsing_utils import prepare_experiment_parser



parser = prepare_experiment_parser()


def main():
  args = parser.parse_args()
  game = JaxPhantomTTT()
  if args.experiment_type == 'nash_dreamer':
    train_nash_dreamer(args, game)
  elif args.experiment_type == 'mmd':
    train_mmd(args, game)
  else:
    train_rnad(args, game)

if __name__ == "__main__":
  main()
