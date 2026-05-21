# NashDreamer

NashDreamer is a multi-agent extension of the DreamerV3 world model framework for two-player zero-sum imperfect information games. We train a centralized world model with decentralized execution via latent infosets.

## Repository Structure

```
.
├── dreamer_ma.py             # Top-level model combining world model and actor-critic
├── ma_rssm.py                # Multi-agent Recurrent State Space Model (MA-RSSM)
├── rnad_dreamer.py           # RNaD actor-critic adapted for imagined Dreamer trajectories
├── dreamer_actor_critic.py   # REINFORCE + TD(λ) actor-critic for Dreamer
├── networks.py               # Neural network building blocks
├── distributions.py          # Categorical / symlog distribution utilities
├── optimizer.py              # Optimizer and learning-rate schedule construction
├── replay_buffer.py          # Experience replay buffer
├── sim_rnad.py               # Standalone RNaD baseline (no world model)
├── train_utils.py            # Checkpointing, logging, configuration dataclasses
│
├── experiments/              # Entry-point scripts for training and evaluation
│   ├── goofspiel_train.py    #   Goofspiel game entry point
│   ├── leduc_train.py        #   Leduc Poker entry point
│   ├── battleships_train.py  #   Battleships entry point
│   ├── pcm_train.py          #   Point Card Matching entry point
│   ├── rps_train.py          #   Rock-Paper-Scissors entry point
│   ├── frozen_lake_train.py  #   Frozen Lake entry point
│   ├── joint_train.py        #   Shared NashDreamer training loop
│   ├── rnad_train.py         #   Shared RNaD training loop
│   ├── actor_critic_evaluate.py   # NashConv / return evaluation
│   ├── head_to_head_evaluate.py   # Head-to-head win-rate evaluation
│   ├── plot_metrics.py            # Plot NashConv/return curves
│   ├── plot_head_to_head.py       # Plot head-to-head win-rate comparisons
│   └── parsing_utils.py           # Shared argument definitions
│
└── games/                    # JAX-accelerated game implementations
    ├── jax_game.py               # Abstract JaxGame base class
    ├── jax_goofspiel.py          # Goofspiel (configurable number of cards)
    ├── jax_leduc.py              # Leduc Poker (full and one-round variants)
    ├── jax_battleships.py        # Battleships
    ├── jax_point_card_matching.py # A toy single agent game environment
    ├── jax_rps.py                # Rock-Paper-Scissors
    └── model_game.py           # A game that simultaneously walks through the real game and latent world model. Used for computation of best responses/expected return in the latent space
```

## Requirements

Python 3.11+, with the primary dependencies being JAX and FLAX NNX. Easiest way to install is directly from requirements.txt:

```bash
pip install -U -r requirements.txt
```

## Training

Each game has a dedicated entry-point module under `experiments/`. All training scripts accept a positional argument selecting the algorithm:

- `nash_dreamer` — world model + actor-critic (NashDreamer)
- `rnad` — standalone RNaD baseline (no world model)

Trained models are saved under `trained_networks/<algo>/<game>/seed_<N>/` by default.

### Using the shell scripts

Run from the project root with an active virtual environment:

```bash
# NashDreamer on Goofspiel with RNaD actor-critic (default settings)
./nash_dreamer_train.sh

# NashDreamer on Leduc Poker with REINFORCE actor-critic
GAME=leduc ALGO=reinforce ./nash_dreamer_train.sh

# Standalone RNaD baseline on Goofspiel
./rnad_train.sh

# RNaD baseline on Leduc Poker
GAME=leduc ./rnad_train.sh
```

Override any parameter by setting the corresponding environment variable before the script:

```bash
# Custom learning rate, single seed
GAME=goofspiel ALGO=reinforce OPT_FLAGS="--lr 0.001" SEEDS="(42,)" ./nash_dreamer_train.sh

# Continue training from a previous checkpoint (do not wipe the directory)
GAME=goofspiel EXPERIMENT_ADD_FLAGS="--continue_train" ./nash_dreamer_train.sh
```

### Calling training scripts directly

```bash
# NashDreamer on Goofspiel-3, single seed
python -m experiments.goofspiel_train nash_dreamer \
  --num_steps 1000 --save_each 100 --print_each 100 \
  --seeds "(42,)" --clean_dir --save_first \
  --encoded_classes 1 --encoded_categories 3 \
  --free_bits_threshold 1.0 --beta_representation 0.1 --batch_size 64 \
  --beta_imagination 1.0 --beta_real 0.3 \
  --buffer_size 64 --replay_ratio -1 --smoothing_window 64 --log_returns \
  rnad --eta 0.2

# RNaD baseline on Goofspiel-3, single seed
python -m experiments.goofspiel_train rnad \
  --num_steps 1000 --save_each 100 --print_each 100 \
  --seeds "(42,)" --clean_dir --save_first \
  --eta 0.2 --sampling_epsilon 0.2 --rho_vtrace -1 \
  --buffer_size 64 --replay_ratio -1 --smoothing_window 64 --log_returns

# NashDreamer on Leduc Poker (larger latent space, longer training)
python -m experiments.leduc_train nash_dreamer \
  --num_steps 10000 --save_each 1000 --print_each 1000 \
  --seeds "(42,)" --clean_dir --save_first \
  --encoded_classes 1 --encoded_categories 30 \
  --free_bits_threshold 1.0 --beta_representation 0.1 --batch_size 64 \
  --beta_imagination 1.0 --beta_real 0.3 \
  --buffer_size 64 --replay_ratio -1 --log_returns \
  rnad --eta 0.2
```

**Game-specific flags:**

| Game | Flag | Description |
|------|------|-------------|
| Goofspiel | `--num_cards N` | Number of cards (default: 3) |
| Goofspiel | `--random` | Use random-order Goofspiel variant |
| Goofspiel | `--obs_only` | Use only partial observation instead of infoset representation |
| Leduc | `--one_round` | Single-round Leduc (no public card) |
| Leduc | `--max_raises N` | Max raises per round (one-round only) |

**Key training flags** (full list in `experiments/parsing_utils.py`):

| Flag | Default | Description |
|------|---------|-------------|
| `--num_steps` | — | Total training steps |
| `--seeds` | `"(42, ...)"` | Tuple of RNG seeds to run |
| `--batch_size` | 64 | Minibatch size |
| `--encoded_classes` | 32 | Categorical distributions in the latent state |
| `--encoded_categories` | 32 | Options per categorical distribution |
| `--buffer_size` | 64 | Replay buffer capacity |
| `--replay_ratio` | -1 | Off-policy reuse ratio (−1 = fully online) |
| `--eta` (RNaD) | 0.2 | reward regularization strength |
| `--beta_imagination` | 1.0 | Weight of imagined trajectory actor-critic loss |
| `--beta_real` | 0.3 | Weight of real trajectory actor-critic loss |

## Evaluation

### Nash convergence and policy quality

Evaluates Nash convergence (`nash_conv`), expected utility (`expected_util`), or smoothed training returns (`env_return`) for one or more algorithms across seeds:

```bash
# Evaluate NashDreamer and RNaD on Goofspiel-3 at step 10 000 (default)
./nash_dreamer_evaluate.sh

# Evaluate on Leduc Poker, Nash convergence metric
GAME_NAME="leduc" METRIC="nash_conv" SCALE_FACTOR=13 ./nash_dreamer_evaluate.sh

# Or call directly:
python -m experiments.actor_critic_evaluate \
  --base_path trained_networks \
  --game_name goofspiel_3 \
  --seeds "(42, 99, 160)" \
  --restore_step 10000 \
  loaded --metric nash_conv \
  --algo_dirs "NashDreamer=nash_dreamer_rnad RNaD=rnad"
```

Pass `--restore_step -1` to evaluate all saved checkpoints in the directory.
2
For Leduc Poker, set `--scale_factor 13` to match the reward scaling from -13 to 13.

### Head-to-head evaluation

Plays two trained policies against each other and records win rates:

```bash
# NashDreamer (P1) vs RNaD (P2) on Goofspiel-4 at step 10 000
GAME_NAME="goofspiel_4" RESTORE_STEP=10000 ./head_to_head.sh

# Custom algorithm pairing:
GAME_NAME="goofspiel_3" \
  ALGO_DIRS="NashDreamer=nash_dreamer_rnad NashDreamerREINFORCE=nash_dreamer_reinforce" \
  ./head_to_head.sh

# Or call directly:
python -m experiments.head_to_head_evaluate \
  --base_path trained_networks \
  --game_name goofspiel_4 \
  --seeds "(42, 99, 160)" \
  --restore_step_a 10000 \
  --restore_step_b 10000 \
  --algo_dirs "NashDreamer=nash_dreamer_rnad RNaD=rnad" \
  --num_games 1024 \
  --metric_store_dir metrics/
```

### Plotting

```bash
python -m experiments.plot_metrics        # NashConv / return curves
python -m experiments.plot_head_to_head   # Head-to-head win-rate comparisons
```

## Pre-computed Metrics

The `precomputed_metrics/` directory contains pre-computed evaluation results (NashConv, head-to-head win rates) across all games and seeds reported.
