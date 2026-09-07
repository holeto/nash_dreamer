"""Argument parser that additionally exposes the decentralized NashDreamer.

Built as a separate entry point rather than an edit to parsing_utils.py, so the
existing per-game training scripts are untouched.  All the flag groups are
reused from parsing_utils, so `nash_dreamer` and `nash_dreamer_decentralized`
take an identical flag set, including the nested reinforce|rnad|mmd positional.
"""

from argparse import ArgumentParser

from train.parsing_utils import (add_nash_dreamer_arguments, add_sim_rnad_arguments,
                                       add_sim_mmd_arguments)


def prepare_decentralized_experiment_parser():
  parser = ArgumentParser()

  subparsers = parser.add_subparsers(dest='experiment_type', required=True,
                                     help="Which algorithm to train.")

  nd_parser = subparsers.add_parser('nash_dreamer',
      help="Run the full NashDreamer training loop, training both the actor-critic and the centralized world model.")
  add_nash_dreamer_arguments(nd_parser)

  nd_dec_parser = subparsers.add_parser('nash_dreamer_decentralized',
      help="Run NashDreamer with a DECENTRALIZED world model: one RSSM per player, "
           "each seeing only its own observation and its own action, no latent-infoset "
           "networks, and a per-player critic. Requires --use_original_infoset and the "
           "rnad train mode. Expected to be unstable -- that is what it is for.")
  add_nash_dreamer_arguments(nd_dec_parser)

  rnad_parser = subparsers.add_parser('rnad',
      help='Run RNaD training on the real environment without the world model')
  add_sim_rnad_arguments(rnad_parser)

  mmd_parser = subparsers.add_parser('mmd',
      help='Run MMD training on the real environment without the world model')
  add_sim_mmd_arguments(mmd_parser)

  return parser
