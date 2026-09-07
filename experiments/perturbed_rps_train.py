
from games.jax_rps_perturbed import JaxPerturbedRPS
from experiments.joint_train import train_nash_dreamer
from experiments.joint_train_decentralized import train_nash_dreamer_decentralized
from experiments.rnad_train import train_rnad
from experiments.mmd_train import train_mmd
from experiments.parsing_utils_decentralized import prepare_decentralized_experiment_parser


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
