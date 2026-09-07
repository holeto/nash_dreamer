# NashDreamer

NashDreamer is a multi-agent extension of the DreamerV3 world model framework for two-player zero-sum imperfect information games. We train a centralized world model with decentralized execution via latent infosets.

## Repository Structure

Python source lives under `src/`, as a set of flat, sibling packages (not nested under one top-level
package) — `import train.goofspiel_train`, `import envs.jax_leduc`, etc. all resolve directly once
`src/` is on `PYTHONPATH`, no `src.` prefix needed.

```
.
├── src/
│   ├── nash_dreamer/          # The learners: world model, actor-critics, replay buffers
│   │   ├── dreamer_ma.py          #   Top-level model combining world model and actor-critic
│   │   ├── ma_rssm.py             #   Multi-agent Recurrent State Space Model (MA-RSSM)
│   │   ├── rnad_dreamer.py        #   RNaD actor-critic adapted for imagined Dreamer trajectories
│   │   ├── mmd_dreamer.py         #   MMD actor-critic adapted for imagined Dreamer trajectories
│   │   ├── dreamer_actor_critic.py #  REINFORCE + TD(λ) actor-critic for Dreamer
│   │   ├── networks.py            #   Neural network building blocks
│   │   ├── distributions.py       #   Categorical / symlog distribution utilities
│   │   ├── optimizer.py           #   Optimizer and learning-rate schedule construction
│   │   ├── replay_buffer.py       #   Experience replay buffer
│   │   ├── sim_rnad.py            #   Standalone RNaD baseline (no world model)
│   │   ├── sim_mmd.py             #   Standalone MMD baseline (no world model)
│   │   ├── sim_ppo.py             #   Standalone PPO best-response learner
│   │   ├── train_utils.py         #   Checkpointing, logging, configuration dataclasses
│   │   └── *_decentralized.py     #   One-RSSM-per-player ablation variant (RNaD only)
│   │
│   ├── envs/                  # JAX-accelerated game implementations
│   │   ├── jax_game.py            #   Abstract JaxGame base class
│   │   ├── jax_goofspiel.py       #   Goofspiel (configurable number of cards)
│   │   ├── jax_leduc.py           #   Leduc Poker (full and one-round variants)
│   │   ├── jax_battleships.py     #   Battleships
│   │   ├── jax_phantom_ttt.py     #   Phantom Tic-Tac-Toe
│   │   ├── jax_rps.py             #   Rock-Paper-Scissors (and jax_rps_perturbed.py)
│   │   ├── jax_strength_duel.py   #   Strength Duel
│   │   ├── jax_point_card_matching.py # A toy single agent game environment
│   │   └── model_game.py          #   Walks the real game and the latent world model together;
│   │                               #   used to compute best responses/expected return in latent space
│   │
│   ├── train/                 # Per-game entry points and the shared training loop
│   │   ├── goofspiel_train.py, leduc_train.py, battleships_train.py, pcm_train.py,
│   │   │   rps_train.py, strength_duel_train.py, phantom_ttt_train.py, perturbed_rps_train.py
│   │   ├── joint_train.py         #   Shared NashDreamer/RNaD/MMD/PPO training loop
│   │   ├── joint_train_decentralized.py
│   │   └── parsing_utils.py       #   Shared argument definitions
│   │
│   ├── eval/                  # Evaluation
│   │   ├── actor_critic_evaluate.py   # NashConv / return evaluation
│   │   ├── head_to_head_evaluate.py   # Head-to-head win-rate evaluation
│   │   ├── policy_eval_utils.py       # Exact tree-walk NashConv/best-response (small games only)
│   │   ├── ppo_exploitability.py      # Budgeted approximate exploitability via SimPPO
│   │   └── pttt_exploitability/       # Exact Phantom TTT exploitability via the external `eas` solver
│   │
│   ├── plotting/              # plot_metrics.py and the world-model plot scripts
│   ├── world_model_experiments/   # World-model quality checks against a trained checkpoint
│   ├── tests/                 # pytest suite (gitignored, machine-local)
│   └── local_plotting/, debug/, tabular_experiments/  # Untracked scratch
│
├── *_train.sh, *_evaluate.sh, ...  # Root-level launcher scripts (see below)
├── pyproject.toml, uv.lock         # Dependencies (see Requirements)
└── trained_networks/, metrics/, ... # Generated output
```

## Requirements

Python 3.12.1+, with the primary dependencies being JAX and Flax NNX (built for CUDA 12). Dependencies
are managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

This creates `.venv/` from the pinned `uv.lock`. Prefer running commands with `uv run` rather than
activating `.venv` and calling `python` directly — `uv run` always resolves *this* project's own
locked environment regardless of what else is active in your shell:

```bash
uv run python -m train.goofspiel_train --num_cards 3 rnad --num_steps 1000 --seeds "(42,)"
```

Every command below assumes `PYTHONPATH=src` is set (needed so the `train.`/`eval.`/`envs.`/etc.
packages under `src/` resolve) and that you're running from the repository root, since checkpoint and
metric paths are written relative to it. All of the shell scripts under "Using the shell scripts"
already set this up internally — export it yourself only when calling a module directly:

```bash
export PYTHONPATH="$(pwd)/src"
```

## Training

Each game has a dedicated entry-point module under `src/train/`. All training scripts accept a positional argument selecting the algorithm:

- `nash_dreamer` — world model + actor-critic (NashDreamer)
- `rnad` — standalone RNaD baseline (no world model)
- `mmd` — standalone MMD baseline (no world model)
- `ppo` — single-agent PPO best response against a frozen, already-trained opponent

Trained models are saved under `trained_networks/<algo>/<game>/seed_<N>/` by default.

### Using the shell scripts

Run from the project root:

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
export PYTHONPATH="$(pwd)/src"

# NashDreamer on Goofspiel-3, single seed
uv run python -m train.goofspiel_train nash_dreamer \
  --num_steps 1000 --save_each 100 --print_each 100 \
  --seeds "(42,)" --clean_dir --save_first \
  --encoded_classes 1 --encoded_categories 3 \
  --free_bits_threshold 1.0 --beta_representation 0.1 --batch_size 64 \
  --beta_imagination 1.0 --beta_real 0.3 \
  --buffer_size 64 --replay_ratio -1 --smoothing_window 64 --log_returns \
  rnad --eta 0.2

# RNaD baseline on Goofspiel-3, single seed
uv run python -m train.goofspiel_train rnad \
  --num_steps 1000 --save_each 100 --print_each 100 \
  --seeds "(42,)" --clean_dir --save_first \
  --eta 0.2 --sampling_epsilon 0.2 --rho_vtrace -1 \
  --buffer_size 64 --replay_ratio -1 --smoothing_window 64 --log_returns

# NashDreamer on Leduc Poker (larger latent space, longer training)
uv run python -m train.leduc_train nash_dreamer \
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
| PCM | `--num_cards N` | Number of cards (default: 3) |
| PCM | `--stochastic` | Deal 1 card randomly, instead of descending |
| PCM | `----chance_turn_before_terminal N` | How many turns before terminal node should the card be dealt randomly. Only for stochastic variant|
| RPS | `--stochastic` | Use special stochastic RPS instead of standard. At the start a chance node will decide from one out of 3 perturbed variants uniformly. |
| Goofspiel | `--num_cards N` | Number of cards (default: 3) |
| Goofspiel | `--random` | Use random-order Goofspiel variant |
| Goofspiel | `--obs_only` | Use only partial observation instead of infoset representation |
| Leduc | `--one_round` | Single-round Leduc (no public card) |
| Leduc | `--max_raises N` | Max raises per round (one-round only) |
| Battleship | `--board_height R` | Number of rows on the board of each player |
| Battleship | `--board_width C` | Number of columns on the board of each player |
| Battleship | `--ship_sizes S1,S2,...,SN` | A comma separated string defining the tile sizes for N ships (minimum 1) |


**Key training flags** (full list in `src/train/parsing_utils.py`):

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

### NashConv and policy quality

Evaluates NashConv (`nash_conv`), expected utility (`expected_util`), or smoothed training returns (`env_return`) for one or more algorithms across seeds:

```bash
# Evaluate NashDreamer and RNaD on Goofspiel-3 at step 10 000 (default)
./nash_dreamer_evaluate.sh

# Evaluate on Leduc Poker, NashConv metric
GAME_NAME="leduc" METRIC="nash_conv" SCALE_FACTOR=13 ./nash_dreamer_evaluate.sh

# Or call directly:
export PYTHONPATH="$(pwd)/src"
uv run python -m eval.actor_critic_evaluate \
  --base_path trained_networks \
  --game_name goofspiel_3 \
  --seeds "(42, 99, 160)" \
  --restore_step 10000 \
  loaded --metric nash_conv \
  --algo_dirs "NashDreamer=nash_dreamer_rnad RNaD=rnad"
```

Pass `--restore_step -1` to evaluate all saved checkpoints in the directory.

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
export PYTHONPATH="$(pwd)/src"
uv run python -m eval.head_to_head_evaluate \
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
export PYTHONPATH="$(pwd)/src"
uv run python -m plotting.plot_metrics        # NashConv / return curves
```

Head-to-head and Phantom TTT exploitability comparison plots (`plot_head_to_head.py`,
`plot_pttt_exploitability.py`, ...) live under `src/local_plotting/` — untracked, one-off scripts, not
a stable part of the pipeline.

## Pre-computed Metrics

The `precomputed_metrics/` directory contains pre-computed evaluation results (NashConv, head-to-head win rates) across all games and seeds reported.
