# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

NashDreamer: a multi-agent DreamerV3 world model for two-player zero-sum imperfect-information games,
plus model-free baselines (RNaD, MMD) and a PPO best-response learner. An experimental decentralized
variant of the world model (one RSSM per player, RNaD only) also exists. Everything is JAX + Flax NNX;
games, losses and trajectory sampling are all jitted and vmapped.

## Repository layout

As of 2026-09-07 the Python source lives under `src/`, one flat package per directory (not nested
under a single top-level package) — `train.foo`, `eval.foo`, `envs.foo` etc. all resolve directly, not
`src.train.foo`. This is a straight split of what used to be a flat set of `*.py` files plus an
`experiments/` and `games/` directory at the repo root:

| Package | Was | Contents |
|---|---|---|
| `src/nash_dreamer/` | root `*.py` | The learners themselves: `dreamer_ma.py`, `ma_rssm.py`, `networks.py`, `train_utils.py`, `replay_buffer.py`, `sim_rnad.py`/`sim_mmd.py`/`sim_ppo.py`, `rnad_dreamer.py`/`mmd_dreamer.py`, `optimizer.py`, `distributions.py`, and every `*_decentralized.py` sibling |
| `src/envs/` | `games/` | `JaxGame` and every game implementation (`jax_leduc.py`, `jax_goofspiel.py`, ...), plus `model_game.py` |
| `src/train/` | `experiments/*_train.py`, `joint_train*.py`, `parsing_utils*.py` | Per-game entry points and the shared training loop |
| `src/eval/` | `experiments/{actor_critic,head_to_head,ppo_exploitability,...}.py` | Evaluation scripts, plus `src/eval/pttt_exploitability/` (was `experiments/pttt_exploitability/` — see its own [CLAUDE.md](src/eval/pttt_exploitability/CLAUDE.md)) |
| `src/plotting/` | `experiments/plot_*.py`, `world_model_plots/*.py` | `plot_metrics.py` and the world-model plot scripts; `plot_utils.py` is their shared helper module |
| `src/world_model_experiments/` | root `world_model_experiments/` | The three world-model quality checks (see Evaluation below) — name unchanged, only the parent directory moved |
| `src/tests/` | root `tests/` | pytest suite. Still gitignored (`tests/` in `.gitignore` matches at any depth) — machine-local, absent after a fresh clone |
| `src/local_plotting/` | root `local_plotting/` | Untracked one-off plot scripts (gitignored, `**/*/local_plotting/`) — not part of the pipeline |
| `src/debug/`, `src/tabular_experiments/` | root `debug/`, `tabular_experiments/` | Scratch. `debug/` is gitignored (`**/*/debug/`); `tabular_experiments/` currently is **not** — see Repo-specific traps |

Root-level and unaffected by the move: all `*.sh` wrapper scripts, `game_instances/`, `cluster_mmd/`
(gitignored cluster backport, see its own [CLAUDE.md](cluster_mmd/CLAUDE.md)), and every generated-output
directory (`trained_networks/`, `metrics/`, `precomputed_metrics/`, `world_model_metrics/`,
`world_model_plots/`, `plots/`).

## Commands

### Install

Dependency management moved to `uv` (`pyproject.toml` + `uv.lock`) on 2026-09-07; `requirements.txt`
is kept only as a legacy mirror of the same audited set and is no longer the source of truth.

```bash
uv sync                        # creates/updates .venv from uv.lock — 63 packages, pinned exactly
uv run python -m train.goofspiel_train --num_cards 3 rnad --num_steps 1000 --seeds "(42,)"
```

**Prefer `uv run` over activating `.venv` and calling `python` directly.** `uv run` always resolves
*this* project's own locked environment from the nearest `pyproject.toml`, regardless of what venv
(if any) is already active in the shell — a bare `python` on `PATH` will happily run against a
completely unrelated project's venv if one happens to be activated, silently using the wrong package
versions with no error. Every root `*.sh` script already does this (`uv run python -m ...`); do the
same in ad hoc commands. `uv run` prints a one-line warning (`` `VIRTUAL_ENV=...` does not match the
project environment path ``) when some other venv is active — that warning is expected and means the
override worked, not that something is wrong.

`requires-python = ">=3.12.1"` in `pyproject.toml` (`.python-version` pins the same). JAX is built for
CUDA 12 (the `nvidia-cu12-*` / `jax-cuda12-*` pins) — this is a GPU-only dependency set, not
conditional on platform.

**Dependency history, so a future `uv lock` doesn't accidentally re-widen this:**
- TensorFlow (`tensorflow`, `tensorboard`, `keras`, `tensorflow_probability`, and everything reachable
  only through them — about half the original pin count) was removed 2026-09-07: nothing in the repo
  imports any of them, they were a holdover from a since-removed visualization. One real consequence:
  **`jax.profiler` traces can no longer be viewed** (that needs `xprof`, the TensorBoard profiler
  plugin, which is what pulled in the whole `google-cloud-*`/`aiohttp`/`requests` stack in the first
  place). Nothing in the repo calls `jax.profiler` today; add `xprof` back if that changes.
- `orbax-checkpoint` is pinned explicitly (`==0.11.39`) even though nothing in the repo imports it —
  checkpointing here is plain `pickle` (`nash_dreamer.train_utils.save_model`/`load_model`). It cannot
  be removed: `flax==0.10.7` itself declares `orbax-checkpoint` as an unconditional, non-optional
  dependency, so it and its own dependency chain (`etils`, `tensorstore`, `msgpack`, `humanize`,
  `simplejson`, `nest-asyncio`, `protobuf`, `fsspec`, `aiofiles`, `uvloop`, ...) get installed via flax
  regardless of what `pyproject.toml` says. Leaving the version unpinned lets the resolver drift to
  whatever the newest compatible release happens to be — that is exactly how `aiofiles` and `uvloop`
  entered this lock in the first place (orbax `0.11.19`→`0.11.39` added them as new hard deps between
  those two releases). Pinning it is what makes `uv lock` reproducible instead of silently picking up
  a newer orbax on every re-lock.

If a fresh `uv sync` starts failing on a missing module, one of the two points above is the likely
culprit — reinstall the one package actually needed rather than restoring the pruned set.

### Running code

**Always run from the repo root**, with `src/` reachable as an import root. `train.joint_train`
builds checkpoint paths as `os.getcwd() + "/trained_networks/..."`, so the working directory still has
to be the repo root; separately, `src/tests/` has no `conftest.py` or `pytest.ini`, and none of
`train.`/`eval.`/`plotting.`/`envs.`/`nash_dreamer.`/`world_model_experiments.` resolve unless `src/`
is on `PYTHONPATH`. Every root `*.sh` script already exports `PYTHONPATH="$(pwd)/src:$PYTHONPATH"` for
this reason; do the same for ad hoc commands (`PYTHONPATH=src uv run python -m ...`).

`src/tests/` currently holds `test_leduc_round1.py`, `test_phantom_ttt.py`, `test_perturbed_rps.py`,
`test_pttt_eas_equivalence.py` (cross-checks `eval/pttt_exploitability/encoding.py` against the
external `eas` library, see Evaluation) and `test_trajectory_bias.py` (a chi-square check that sampled
trajectories match declared chance/action probabilities — the same class of bug as the reward-alignment
issue below). `pytest` itself is not a pinned dependency (pre-existing, not new to the `uv` migration —
install it separately). There is no other in-repo mechanism for "did my change break anything".

```bash
# Tests (whole file, then a single test)
PYTHONPATH=src uv run python -m pytest src/tests/test_leduc_round1.py -q
PYTHONPATH=src uv run python -m pytest src/tests/test_phantom_ttt.py::TestTerminal::test_draw_on_full_board -q

# Training, called directly. First positional = algorithm; nash_dreamer takes a
# SECOND positional (reinforce|rnad|mmd) after all its flags.
PYTHONPATH=src uv run python -m train.goofspiel_train --num_cards 3 nash_dreamer \
  --num_steps 1000 --seeds "(42,)" --encoded_classes 1 --encoded_categories 3 rnad --eta 0.2
PYTHONPATH=src uv run python -m train.goofspiel_train --num_cards 3 rnad --num_steps 1000 --seeds "(42,)"

# Standalone PPO best response against a frozen, already-trained opponent (checkpointed,
# see the 4th-algorithm note below). The opponent must have been trained with --use_original_infoset.
PYTHONPATH=src uv run python -m train.goofspiel_train --num_cards 3 ppo \
  --opponent_path trained_networks/rnad/goofspiel_3/seed_42/step_1000.pkl \
  --player_id 0 --num_steps 1000 --seeds "(42,)"

# Training via the shell wrappers (PYTHONPATH/uv run and env-var overrides handled internally,
# see below for the --clean_dir hazard)
GAME=leduc ALGO=reinforce ./nash_dreamer_train.sh
GAME=goofspiel EXPERIMENT_ADD_FLAGS="--continue_train" ./nash_dreamer_train.sh

# Decentralized world model (one RSSM per player, RNaD only, real infosets only). No shell
# wrapper exists yet, and it is currently only wired up for one game:
PYTHONPATH=src uv run python -m train.perturbed_rps_train nash_dreamer_decentralized --use_original_infoset \
  --num_steps 1000 --seeds "(42,)" rnad --eta 0.2

# Evaluation and plots
PYTHONPATH=src uv run python -m eval.actor_critic_evaluate --game_name goofspiel_3 --seeds "(42,)" \
  --restore_step 10000 loaded --metric nash_conv --algo_dirs "NashDreamer=nash_dreamer_rnad RNaD=rnad"
PYTHONPATH=src uv run python -m eval.head_to_head_evaluate --game_name goofspiel_4 ...
PYTHONPATH=src uv run python -m plotting.plot_metrics

# Budgeted approximate exploitability: trains PPO best responses in memory and plays them
# head-to-head in one process, writing metrics only (no checkpoints). See below.
ALGO_DIR=mmd GAME_NAME=goofspiel_3 RESTORE_STEP=1000 ./ppo_exploitability.sh

# World model quality checks against a trained DreamerMA checkpoint. See below.
./chance_marginal_evaluate.sh
./posterior_collapse_evaluate.sh
./world_model_sampling_evaluate.sh
```

There is no build, no linter, and no CI config.

### `--clean_dir` is destructive and on by default in the shell scripts

All four `*_train.sh` scripts (`nash_dreamer_train.sh`, `rnad_train.sh`, `mmd_train.sh`,
`ppo_train.sh`) default `EXPERIMENT_ADD_FLAGS="--clean_dir --save_first"`, and `--clean_dir` does
`rmtree(model_save_dir)` ([src/train/joint_train.py](src/train/joint_train.py)), wiping
`trained_networks/<algo>/<game>/seed_<N>/` before training. It also swallows every exception, so a
failure to delete only prints a message. Prefer calling the module directly for smoke tests, and never
add `--clean_dir` to a run over a directory whose contents matter. `ppo_exploitability.sh` is the one
related script that never touches `trained_networks/` — it trains its PPO best responses in memory and
writes metrics only.

## Architecture

### NashDreamer, RNaD, MMD and PPO, one training loop

[src/train/joint_train.py](src/train/joint_train.py)`::train_loop` is shared by all of them and is the
only place that knows about checkpointing, seeds and `--continue_train`. It uses the game object for
exactly one thing: `game.to_compact_str()` as the directory name.

- **NashDreamer** (`DreamerMA`) = an `MARSSM` world model + an actor-critic head, chosen by config type
  in `nash_dreamer/dreamer_ma.py::init`: `MMDConfig` → `MMDDreamer`, else `use_rnad` → `RNaDDreamer`,
  else `DreamerActorCritic` (REINFORCE). One `train_step` = sample the buffer, update the world model,
  then step the actor-critic on *both* real and imagined trajectories (`--beta_real` /
  `--beta_imagination`).
- **`SimRNaD`** / **`SimMMD`** (`nash_dreamer/sim_rnad.py`, `nash_dreamer/sim_mmd.py`) are standalone
  baselines: no world model, the actor reads the game's real infoset tensor directly. `MMDDreamer` and
  `SimMMD` share one `MMDConfig` — the Dreamer-only fields (`beta_imagination`, `wm_warm_up_period`,
  etc.) are simply ignored by `SimMMD`, exactly the dual-use pattern `RNaDConfig` already has for
  `RNaDDreamer`/`SimRNaD`. MMD is Magnetic Mirror Descent with the magnet fixed at the uniform policy:
  a PPO-style clipped surrogate (`sim_mmd.clipped_surrogate`) plus an explicit proximal-KL term and a
  magnet-KL term toward uniform. `nash_dreamer/mmd_dreamer.py` imports its loss primitives directly
  from `sim_mmd.py`, the same way `rnad_dreamer.py` imports v-trace/NeuRD from `sim_rnad.py` — the loss
  math lives once in the standalone learner and is reused by its Dreamer integration.
- **`SimPPO`** (`nash_dreamer/sim_ppo.py`) is different in kind, not just implementation: it does not
  learn a policy for the game, it learns a best response *to one frozen, already-trained opponent*
  loaded from `--opponent_path` (a `DreamerMA`/`SimRNaD`/`SimMMD` checkpoint trained with
  `--use_original_infoset`). It is strictly on-policy — no replay buffer, no `buffer_config` attribute
  at all — standard PPO clipping is its only correction. `PPOConfig.entropy_coeff` defaults to **0** on
  purpose: entropy regularization softens the best response, which *under-estimates* the opponent's
  exploitability, so the default favors "the best-response value is the number I want" over
  exploration. Run it standalone via `src/train/<game>_train.py ppo` (checkpointed through the same
  `train_loop`, under `trained_networks/ppo_p{player_id}/...`), or sweep it in-memory with
  `eval/ppo_exploitability.py` (see Evaluation).

### Decentralized world model (`nash_dreamer/*_decentralized.py`)

A second, parallel world-model implementation. `DecentralizedMARSSM`
(`nash_dreamer/ma_rssm_decentralized.py`) gives each player its **own** RSSM, fed only that player's
own observation and own action (weights shared via `nnx.vmap` over the player axis) — the opposite of
`MARSSM`'s one shared recurrent state over the joint observation. There is no latent-infoset network in
this variant at all, only real infosets: the constructor asserts `--use_original_infoset` and
`information_state_tensor_shape() == observation_tensor_shape()`. Per-player rewards are reconciled
from two separate reward heads as `(r0 - r1) / 2`, and the critic reads one player's own infoset rather
than the joint state.

This is a revival, not an extension, of the idea abandoned in commit `9b8e6e5` ("Decentralized model too
unstable. Switched back to centralized model..."), and unlike that abandonment the current code is
explicit that the instability is the point — `src/train/parsing_utils_decentralized.py`'s help text for
`nash_dreamer_decentralized` reads *"Expected to be unstable -- that is what it is for."* Treat it as a
deliberate ablation target, not a working alternative to `MARSSM`.

Only RNaD is implemented decentralized (`nash_dreamer/rnad_dreamer_decentralized.py`; both
`dreamer_ma_decentralized.py::init` and `train/joint_train_decentralized.py::train_nash_dreamer_decentralized`
assert `train_mode == "rnad"`) — there is no decentralized MMD or REINFORCE.
`DecentralizedWMReplayBuffer` (`nash_dreamer/replay_buffer_decentralized.py`) only implements the
arrival-reward convention (no decentralized `ActorReplayBuffer`, no `filter_chance_rewards`); it does
not override `__init__`, so it still gets `check_reward_alignment_once` for free from `WMReplayBuffer`.
No shell script references any of this yet, and no evaluation script (`eval/policy_eval_utils.py`,
`eval/head_to_head_evaluate.py`) knows about the `Decentralized*` classes — the `python -m` command in
Commands above is currently the only way to run or inspect it. The world-model update and v-trace logic
in `rnad_dreamer_decentralized.py` are hand-duplicated from the centralized versions, not shared, so a
bugfix in `dreamer_ma.py` / `rnad_dreamer.py` does not propagate here automatically.

The only game currently wired to it is Perturbed RPS (`src/envs/jax_rps_perturbed.py::JaxPerturbedRPS`,
a `JaxRPS` subclass, launched via `src/train/perturbed_rps_train.py`), which exists specifically to
make exploitability measurable at all: plain `JaxRPS`'s zero-initialized policy head already **is** the
uniform equilibrium, so an untrained run and a converged run look identical on that metric.
`JaxPerturbedRPS` reweights one antisymmetric payoff pair (Scissors beats Paper 2:0 instead of 1:0),
which keeps the game value at 0 but moves the unique equilibrium to `(1/2, 1/4, 1/4)` — something
convergence can actually be measured against.

### Latent vs. real infosets

By default the actor does *not* see the game's infoset tensor — `MARSSM` learns a latent infoset per
player and the actor-critic runs on that. `--use_original_infoset` switches to the real one and asserts
`information_state_tensor_shape() == observation_tensor_shape()`. This is why
`src/envs/model_game.py` exists: `DreamerModelGame` exposes the *learned* model as a `JaxGame` (latent
stochastic state becomes chance outcomes), and `InformedRealGame` walks the real game while carrying
latent infosets alongside. Evaluation scripts pick between `model.game` and `InformedRealGame(model)`
based on `use_real_infoset`. (The decentralized variant above skips this choice entirely — it has no
latent-infoset path and always requires real infosets.)

### Replay buffer scans fixed-length trajectories

`nash_dreamer/replay_buffer.py` unrolls exactly `max_trajectory_length()` steps with `nnx.scan`,
regardless of when the game ends, then filters chance nodes down to
`max_trajectory_lenght_no_chance()`. Two consequences that every game must respect: **terminal must be
absorbing** (sticky flag, zero reward afterwards), and **the legal mask must never be all-zero** —
actions are drawn with `jax.random.choice(key, n, p=pi)` over the mask on every step, including steps
past the end of the game.

### Two different reward conventions, one per buffer

The two buffers store *different* things under `reward`, and the chance-node filtering differs to match:

- `WMReplayBuffer` stores `carry.reward` — the reward for **arriving** at a step — and filters it with
  the same plain `non_chance` mask as `obs`, so the two stay aligned.
  `train_utils.wm_timestep_to_timestep` then does the `reward[1:]` shift to actor-critic convention.
- `ActorReplayBuffer` (and `sim_ppo.py`) store `next_rewards` — the reward for **acting** at a step —
  so each surviving non-chance step must absorb its own reward plus the rewards of any chance steps
  that follow it before the next decision. That is `replay_buffer.filter_chance_rewards`, a segment
  sum, which is independent of how chance and play nodes interleave.

Until 2026-07-31 `ActorReplayBuffer` instead used `jnp.nonzero(~jnp.roll(is_chance, 1), ...)`.
`jnp.roll(x, 1)` shifts *right*, so despite the name that mask tested the **previous** step, crediting
every step with the reward of the step *after* it. Consequences: chance-free games were unaffected
(the mask is the identity); Leduc kept its total but credited the terminal reward one step early; and
in `JaxRandomGoofspiel`, where chance and play strictly alternate, every play was credited with a
chance node's reward — always zero — so **100 % of the return was destroyed and nothing ever learned**.
A cluster `SimMMD` checkpoint at 30 000 steps on `goofspiel_random_13` is uniform to within 3.9e-5.
**Every `goofspiel_random_*` result predating that date is meaningless.**.

`replay_buffer.check_reward_alignment_once` now runs at every learner construction: it rolls out a few
uniform trajectories and raises if filtering loses reward. It also catches an under-declared
`max_trajectory_length()` (game never terminates inside the scan) or `max_trajectory_lenght_no_chance()`
(trailing segments dropped). Do not remove it — this failure mode is silent, and it costs under a second.

### `legal_policy` shifts by the maximum over LEGAL actions

Softmax is invariant to the shift constant, but the choice matters in float32. Until 2026-07-31
`legal_policy` / `legal_log_policy` shifted by `logit.max(-1)` over **all** actions, illegal ones
included. Illegal actions get no gradient — the policy is masked to zero there — so their logits drift
freely, and in any game where an action is illegal in one state but the network's favourite in another
(a card already played, a cell already shot) they grow without bound. Once an illegal logit exceeds every
legal one by ~88, `exp(shifted)` underflows for every legal action:

- gap ≳ 88: the normalization enters the denormal range and the **backward pass returns NaN**;
- gap ≳ 90: the normalization is exactly 0, the policy is exactly all-zero, and its gradient is
  silently 0. `jax.random.choice(key, n, p=0)` then returns action 0 every time, so every trajectory
  becomes the same degenerate line.

`_legal_shifted_logit` (`nash_dreamer/train_utils.py`) now shifts by the max over legal actions, so the
best legal action contributes `exp(0) = 1` and the normalization is always ≥ 1. Verified equal to the
old code within 1.8e-7 (float32 eps) on random inputs, and the goofspiel_3 known-answer PPO check
reproduces its pre-fix values to all six decimals. It affects every learner, since they all reach it
through `ActorNetwork`.

This is what made `SimPPO` NaN at ~2000 steps on battleships and goofspiel_random_13, reported as
`BR = 0.000000 ± 0.000000` (all draws) and `-1.000000 ± 0.000000` (always plays card 0) — a standard
error of exactly zero across 100 000 games is the signature. MMD and RNaD survived it only because their
magnet and KL terms stop the policy peaking that hard; they were exposed to the same failure.
`SimPPO.step` also carries a non-finite tripwire now (first 5 updates, then every 500th).

## The game layer (`src/envs/`)

[src/envs/jax_game.py](src/envs/jax_game.py) defines `JaxGame` with 13 abstract methods. Read
`jax_goofspiel.py::JaxGoofspiel` for the minimal deterministic case, `jax_leduc.py` for chance nodes
and turn-based play, `jax_rps.py` for the smallest possible example.

Contract details that are not visible from the abstract signatures:

- `get_info(state)` returns `(state_tensor, p1_infoset, p2_infoset, public_state)`, each a flat 1-D
  float vector of exactly the declared shape. Convention: prefix each player tensor with a player-id
  one-hot; zero every tensor at chance nodes.
- `apply_action(state, actions)` returns `(new_state, terminal, reward, legals)` where `legals` is
  `[num_players, num_distinct_actions]`, `terminal` is a scalar bool, and `reward` is a **scalar
  float32 from player 1's perspective** (p2's is its negation). Normalize into `[-1, 1]`.
- **Turn-based games** reserve action id `0` as a dummy `INVALID`, give the non-acting player exactly
  that one-hot as their legal mask, and index all effects by the acting player. `one_hot(action - 1, ...)`
  makes the dummy encode as all zeros. See `jax_leduc.py` and `jax_phantom_ttt.py`.
- **Deterministic games override none of the chance methods** — the `jax_game.py` defaults are correct.
  But `apply_action` still gets *traced* with the dummy joint action, because the buffer's
  `lax.cond` traces its chance branch either way. Guard against that changing state.
- State is a `@chex.dataclass(frozen=True)` PyTree of fixed-shape arrays; methods carry
  `@functools.partial(jax.jit, static_argnums=(0,))` and must survive being vmapped over both the state
  axis and the action axis (see the double vmap in `eval/policy_eval_utils.py`).

### Adding a new game

There is **no game registry**. Dispatch is by module name — `nash_dreamer_train.sh` literally runs
`uv run python -m train."$GAME"_train`. So a new game is two files:

1. `src/envs/jax_<name>.py` implementing `JaxGame`.
2. `src/train/<name>_train.py` — copy [src/train/goofspiel_train.py](src/train/goofspiel_train.py),
   add game-specific flags, dispatch on `args.experiment_type`. Game flags must come *before* the
   `nash_dreamer|rnad|mmd|ppo` positional.

`to_compact_str()` = `game_name()` + `params_dict()` **values** joined by `_`. It is both the checkpoint
directory name and the `--game_name` every evaluate/plot script takes, so changing `params_dict` silently
relocates checkpoints and breaks evaluation paths.

## Evaluation

[src/eval/policy_eval_utils.py](src/eval/policy_eval_utils.py) computes `nash_conv` / best responses by
an **exhaustive tree walk** that materializes every history at a depth at once and expands each by all
`A²` joint actions before filtering. This is only viable for small games (Goofspiel-3, Leduc, RPS). It
OOMs within a few plies on large ones — Phantom TTT dies at depth 4 of 17. For anything bigger, use
`eval/head_to_head_evaluate.py` for an approximate empirical win rate, or one of the two exact/budgeted
alternatives below for an actual exploitability number.

`src/eval/goofspiel_nash.pkl` and `leduc_nash.pkl` are reference equilibria for the `nash` subcommand.
Everything under `metrics/`, `precomputed_metrics/`, `world_model_metrics/` is generated output.

### Exact exploitability for Phantom TTT (`src/eval/pttt_exploitability/`)

Not a scaled-up version of the tree walk above — a three-stage bridge to `eas` ("exp-a-spiel"), an
external compiled C++ treeplex solver that is **not a pinned dependency** and must be built separately.
Driven end-to-end by the root `pttt_exploitability.sh`. It produces an exact NashConv in the same sense
as `policy_eval_utils.nash_conv`, and works for **real-infoset actors only** — it rejects a
latent-infoset `MARSSM` outright.

See [src/eval/pttt_exploitability/CLAUDE.md](src/eval/pttt_exploitability/CLAUDE.md) before touching
anything in that directory or the root `pttt_exploitability.sh` — in particular, that script
deliberately carries no cluster preamble, and must not gain one.

### Budgeted exploitability (`src/eval/ppo_exploitability.py`)

`eval/ppo_exploitability.py` trains `SimPPO` best responses in memory and plays each head-to-head
against the frozen opponent in the same process, so it writes metrics only and never touches
`trained_networks/`. It sweeps the **seed** directories of one algorithm at one fixed `--restore_step`
(or takes a single `--opponent_path`), and reports `max_seed BR(p0) + max_seed BR(p1)`.

The PPO budget is **part of the metric, not a shortcoming of it**. Best responding to a frozen opponent
is far easier than solving the game, so an unbounded budget saturates and step 0 and step 30000 of a hard
game both score the maximum. Consequences: the number is a *budgeted lower bound* and must never be
compared against the exact treeplex numbers from `src/eval/pttt_exploitability/` above (hence its
output is `approx_exploitability.tsv`, not `exploitability.tsv`); and `--num_steps` / `--ppo_seeds` /
`--batch_size` / `--num_epochs` must be identical across everything compared, which every row records and
the script warns about on mismatch. A negative `expl_p` means the search never matched the profile's own
value at that budget; it is reported, not clamped.

One hazard it warns about: `SimPPO`'s rollout always feeds the opponent `symlog(obs)` (as every training
path does), but `MARSSM.get_policy` — which `head_to_head_play` uses — symlogs only when `obs_loss_bce`
is False. A real-infoset `DreamerMA` stored with `obs_loss_bce=True` is trained against one policy and
played against another. The repair is to correct the stored flag on the checkpoint — the weights
themselves are fine, since `obs_loss_bce` is a plain bool affecting no network's shape, only
`get_decoder`'s output convention and `get_encoder`'s input convention.

### World model quality (`src/world_model_experiments/`)

Three checks against a trained `DreamerMA` checkpoint, each with a matching root-level
`<name>_evaluate.sh` wrapper (env-var configured, e.g. `--algo_dir nash_dreamer_rnad --game_dir leduc`).
All three walk the tree under the model's own policy, pruned by `--policy_eps`, and accept
`--restore_step -1` to sweep every checkpoint. `chance_marginal_eval.py` compares the model's filtered
prior dynamics distribution against real reachable next-observations (L∞ distance / total variation,
split by chance vs. deterministic nodes). `posterior_collapse_eval.py` instead computes
KL(posterior‖prior) at each chance node against the *realized* outcome, which catches a model that
ignores chance-outcome information even when reconstruction looks fine (an encoder-incompatible
checkpoint raises rather than silently producing a meaningless number). `world_model_sampling_eval.py`
samples full imagined rollouts (own policy, prior-sampled latents) in parallel with the real game and
flags steps where obs/reward/terminal/legal diverge past a threshold.

## Repo-specific traps

- `max_trajectory_lenght_no_chance` — the typo is part of the API, spelled that way everywhere.
- `src/train/frozen_lake_train.py` is dead code: it imports a `train` symbol that does not exist in
  `joint_train.py`, and uses `args` before assigning it. Do not copy it as a template.
- `src/eval/pttt_exploitability/` needs the external `eas` package built separately (see Evaluation
  above) — `import eas` fails after a bare `uv sync`.
- `src/debug/` is scratch, not a source of truth. Edits belong in the root copies.
- `src/tabular_experiments/` is currently **untracked** (unlike `src/debug/` and `src/tests/`, no
  `.gitignore` pattern covers `tabular_experiments/`, so `git add` would pick it up) even though its
  contents were tracked before the `src/` reorg. Resolve this deliberately — either add a `.gitignore`
  entry or re-add the directory — rather than letting a stray `git add -A` decide it.
- README says Goofspiel takes `--obs_only`; the real flag is `--observation_only`. The README also
  predates MMD, PPO, the decentralized world model, `strength_duel`, `phantom_ttt` and `perturbed_rps`.
- Per-game hyperparameters live in hardcoded bash `if [ "$GAME" == ... ]` blocks duplicated across
  `nash_dreamer_train.sh`, `rnad_train.sh`, `mmd_train.sh`, `ppo_train.sh` and their `src/debug/`
  copies — there are no yaml/hydra configs anywhere.
- `--wm_warm_up` defaults to 1000, so short NashDreamer runs never exercise the imagination path. Set it
  to 1 when smoke-testing a new game.
