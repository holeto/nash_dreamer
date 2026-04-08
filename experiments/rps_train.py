
from games.jax_rps import JaxRPS, JaxStochasticRPS
from experiments.joint_train import train_nash_dreamer
from experiments.rnad_train import train_rnad
from experiments.parsing_utils import prepare_experiment_parser



parser = prepare_experiment_parser()
parser.add_argument("--stochastic", action="store_true", help="A flag whether to use the stochastic or standard RPS.")


def main():
  args = parser.parse_args()
  game = JaxStochasticRPS() if args.stochastic else JaxRPS()
  if args.experiment_type == 'nash_dreamer':
    train_nash_dreamer(args, game)
  else:
    train_rnad(args, game)
    
if __name__ == "__main__":
  main()