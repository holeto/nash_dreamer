# Training

Detailed training reference for `src/train/`. See the root [README.md](../../README.md) for the
shell-script workflow (`nash_dreamer_train.sh`, `rnad_train.sh`, `mmd_train.sh`, `ppo_train.sh`) and
install instructions — this page covers calling the per-game entry points directly, plus the full
flag reference.

Each game has a dedicated entry-point module here (`goofspiel_train.py`, `leduc_train.py`,
`battleships_train.py`, `pcm_train.py`, `rps_train.py`, `strength_duel_train.py`,
`phantom_ttt_train.py`, `perturbed_rps_train.py`). Every one accepts a positional argument selecting
the algorithm:

- `nash_dreamer` — world model + actor-critic (NashDreamer)
- `rnad` — standalone RNaD baseline (no world model)
- `mmd` — standalone MMD baseline (no world model)
- `ppo` — single-agent PPO best response against a frozen, already-trained opponent

Trained models are saved under `trained_networks/<algo>/<game>/seed_<N>/` by default.

## Calling training scripts directly

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

# Standalone PPO best response against a frozen, already-trained opponent. The opponent
# checkpoint must have been trained with --use_original_infoset.
uv run python -m train.goofspiel_train ppo \
  --opponent_path trained_networks/rnad/goofspiel_3/seed_42/step_1000.pkl \
  --player_id 0 --num_steps 1000 --seeds "(42,)"
```

Game flags (below) come *before* the `nash_dreamer|rnad|mmd|ppo` positional; algorithm-specific flags
(`--eta`, `--beta_imagination`, ...) come after it.

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

**Key training flags** (full list in [parsing_utils.py](parsing_utils.py)):

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

## `--clean_dir` is destructive and on by default in the shell scripts

All four `*_train.sh` scripts default `EXPERIMENT_ADD_FLAGS="--clean_dir --save_first"`, and
`--clean_dir` does `rmtree(model_save_dir)` ([joint_train.py](joint_train.py)), wiping
`trained_networks/<algo>/<game>/seed_<N>/` before training. It also swallows every exception, so a
failure to delete only prints a message. Prefer calling the module directly for smoke tests, and
never add `--clean_dir` to a run over a directory whose contents matter.

## The shared training loop

[joint_train.py](joint_train.py)`::train_loop` is shared by every algorithm and is the only place
that knows about checkpointing, seeds and `--continue_train`. `--continue_train` resumes from the
latest checkpoint in the directory instead of starting fresh — combine it with dropping
`--clean_dir` from `EXPERIMENT_ADD_FLAGS`.

## Decentralized world model

`nash_dreamer_decentralized` (in `src/nash_dreamer/*_decentralized.py`) is an experimental,
deliberately-unstable ablation: one RSSM per player instead of one shared world model, RNaD only,
real infosets only. It has no shell-script wrapper yet and is currently wired up for one game
(Perturbed RPS):

```bash
export PYTHONPATH="$(pwd)/src"
uv run python -m train.perturbed_rps_train nash_dreamer_decentralized --use_original_infoset \
  --num_steps 1000 --seeds "(42,)" rnad --eta 0.2
```
