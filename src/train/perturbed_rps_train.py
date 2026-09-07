
from envs.jax_rps_perturbed import JaxPerturbedRPS
from train.joint_train import train_nash_dreamer
from train.joint_train_decentralized import train_nash_dreamer_decentralized
from train.rnad_train import train_rnad
from train.mmd_train import train_mmd
from train.parsing_utils_decentralized import prepare_decentralized_experiment_parser


parser = prepare_decentralized_experiment_parser()


def main():
  args = parser.parse_args()
  game = JaxPerturbedRPS()
  if args.experiment_type == 'nash_dreamer':
    train_nash_dreamer(args, game)
  elif args.experiment_type == 'nash_dreamer_decentralized':
    train_nash_dreamer_decentralized(args, game)
  elif args.experiment_type == 'mmd':
    train_mmd(args, game)
  else:
    train_rnad(args, game)


if __name__ == "__main__":
  main()
