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

### Three phases: stage one, the warm-up, then imagination

A NashDreamer run passes through up to three phases, in order. Stage one is **orthogonal to the
warm-up** — it runs first and ends dynamically, and only when it is over does the fixed
`--wm_warm_up` budget start counting.

| phase | ends when | world-model loss | actor-critic | imagination |
|---|---|---|---|---|
| 1. stage one | the WM loss plateaus (dynamic) | recon + infoset, `rep` only if `--vq_vae_posterior`, no `dyn` | soft: trains on real / hard and complete: frozen | off |
| 2. warm-up | `--wm_warm_up` steps after phase 1 | recon + infoset + `dyn`; **no `rep` ever**. Under complete, only `dyn` *trains* | trains on real | off |
| 3. full | — | same as phase 2 | trains | on |

**Without a two-stage flag phase 1 is empty**, `stage_one_end_step` is 0, the warm-up counts from
step 0, and none of the freeze below applies — exactly the original behaviour, where `--wm_warm_up`
(default 1000) gated imagination alone and both `dyn` and `rep` ran from step 0.

There are three mutually exclusive variants, ordered by how much they freeze:

| flag | stage one | stage two |
|---|---|---|
| `--soft_two_stage` | AC trains on real | posterior frozen |
| `--hard_two_stage` | AC update skipped entirely | posterior frozen |
| `--complete_two_stage` | identical to hard | **whole world model frozen except the prior** |

Stage one drops the dynamics loss, since the prior is not being trained yet, and with it the
KL-balancing representation loss, which only pulls the posterior toward that untrained prior.
`--vq_vae_posterior` (renamed from `--l2_posterior`) substitutes its commitment term — cross entropy
against the posterior's own argmax, which never references the prior — to sharpen the posterior
during stage one. What remains is reconstruction/prediction (`dec`, `con`, `leg`, `rew`) plus the
four latent-infoset terms. Soft and hard differ only in the actor-critic: soft keeps training it on
real trajectories, hard skips `actor_critic.step` outright — no real loss, no imagination loss, no
critic or target-network update. `--complete_two_stage` reuses hard's stage one unchanged and
differs only in stage two.

**After stage one the posterior is frozen**, and that takes *two* independent changes — dropping the
representation loss alone is not enough.

1. **No term aimed at the posterior.** The VQ commitment term stops with stage one, and the
   KL-balancing representation loss never runs at all under two-stage training:

   ```python
   two_stage = soft_two_stage or hard_two_stage or complete_two_stage
   compute_dynamics = not in_stage_one
   compute_representation = (vq_vae_posterior and in_stage_one) if two_stage else True
   ```

2. **`stop_gradient` on the Encoder→Observer output.** `sample_categorical`
   (`nash_dreamer/distributions.py`) is a **straight-through estimator** —
   `stop_gradient(one_hot) + (probs - stop_gradient(probs))` — so gradient from the decoder, the
   reward/done/legal heads and the recurrent chain flows back through the sampled state into the
   observer and encoder regardless of which losses are on. `update_world_model` therefore detaches
   `get_encoder_no_jit(...)` (which *is* Encoder→Observer) whenever `freeze_posterior` holds.

`dyn` is deliberately untouched: it carries `stop_gradient(posterior)` already, so it trains the
prior *toward* the frozen posterior without being able to move it. Everything else keeps
training — the sequential network still gets gradient because `recurrent_state` feeds the decoder,
the predictors, the dynamics net and the next-recurrent call *directly*, not only through the
detached posterior. Verify with `--report_gradnorms`: `enc` and `observer` are exactly `0` after the
boundary and nonzero before it.

So under `--vq_vae_posterior` the reported `rep` goes nonzero → **0** at the stage-one boundary, and
without it `rep` is 0 for the whole run. Both are intended; the startup banner says which to expect.
`freeze_posterior` is static like `two_stage_warm_up`, so the switch is a compile-time branch and
costs one trace, not a per-step test.

#### `--complete_two_stage`: freeze everything but the prior

The literal reading of "separate representation learning from generative learning". Stage one is
hard's; stage two freezes the *whole* world model except `dyn` — `enc`, `observer`, `seq`, `dec`,
`rew`, `term`, `leg` and all three infoset networks stop learning, and only the dynamics network and
the actor-critic keep training. Three mechanisms, and **all three are needed**:

1. **Detach every loss term but `dyn`.** They are still computed (the forward pass produces the
   prediction step the actor-critic consumes anyway) and still *reported*, so `dec`/`con`/`leg`/
   `rew`/`is_*` keep printing — read them as drift of a fixed representation, not as progress. Since
   `stop_gradient` is the identity forward, the printed compound loss stays comparable across the
   boundary.
2. **`stop_gradient` on `dyn`'s input.** The dynamics loss is the one term still training, and its
   input is `recurrent_state` — without this it would keep training `seq`, and through it everything
   upstream.
3. **Restore the frozen parameters after each `train_step`.** A zero gradient is *not* enough:
   `DreamerMA.optimizer` is one optimizer shared by the world-model and actor-critic updates, and
   LaProp carries momentum across both, so a network that stops receiving gradient still drifts on
   leftover momentum. Measured on goofspiel_3 that tail is ~2.5e-5 max |Δw| over the first ten steps
   past the boundary, decaying geometrically (3e-6, 2e-7, 7e-10) — small, but not frozen.
   `snapshot_frozen_networks`/`restore_frozen_networks` rebind the arrays around both updates, which
   is what `optimizer.update` itself does and costs no device work. With it the frozen networks are
   **bit-identical** across 80 steps while `dyn`, `actor` and `critic` keep moving.

The frozen set is derived as `[k for k in self.network_keys if k != 'dyn']`, not listed, so the
decentralized model (no infoset networks) gets the right set with nothing to keep in sync. Every
network is `nnx.Param`-only, so the Param-filtered snapshot misses nothing.

**And at the boundary, the optimizer's moments are reset** (`--complete_two_stage` only).
`dyn` receives exactly zero gradient throughout stage one and starts at zero, so its `nu`/`mu`
are still at their init values when stage one ends — but the bias-correction divisor is not:
`nu_hat = nu / (1 - beta2**step)`, and `step` has been counting all along. The first real
dynamics gradient is then normalized by an `nu` built from one sample against a divisor that
assumes thousands, overshooting by up to `sqrt(1/(1-beta2))` = **31.6x** and staying mis-scaled
for ~1000 steps while `nu` re-warms. That is the error in the rms *normalization*; the realized
parameter movement is smaller, because momentum contributes only `(1 - beta1)` of it on the first
step and AGC clips the gradient first. Measured on goofspiel_3 with a 3000-step stage one, `dyn`
moves **4.0x-8.8x** further per step over the first eight stage-two steps without the reset than
with it, peaking around step five:

| stage two step | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| no reset | 1.8e-3 | 3.2e-3 | 4.3e-3 | 7.4e-3 | 7.7e-3 | 7.0e-3 | 6.2e-3 | 4.5e-3 |
| with reset | 4.4e-4 | 4.8e-4 | 5.6e-4 | 8.7e-4 | 8.7e-4 | 9.1e-4 | 9.4e-4 | 8.9e-4 |

Those counters are **single scalars shared by every parameter** (see `make_opt` in
`nash_dreamer/optimizer.py`), so this cannot be fixed for `dyn` alone — resetting the counter
while leaving other networks' warm `nu` in place crushes *their* updates by the same factor in
reverse. Moments and counters must go together, which is exactly why the reset is gated to
`--complete_two_stage`: there every other world-model network is frozen from that point and the
actor-critic has never stepped, so nothing useful is discarded. Soft and hard keep the optimizer
they have; their `dyn` still takes the over-scaled first steps, which is part of what makes them
the looser variants.

Not reset, deliberately: the **LR schedule's own count** (restarting it would replay
`OptimizerConfig.warmup`, 1000 by default, from lr 0, and would restart a linear/cosine anneal),
and `nnx.Optimizer.step` (nothing reads it). `reset_optimizer_moments` asserts the chain layout
so a change to `make_opt` fails loudly rather than silently zeroing the schedule.

One global worth knowing about while reading any of these counters: **`optimizer.update` is
called once per world-model step, once by the actor-critic's real loss and once more by its
imagination loss**, so the shared counters advance at **1x** during a hard/complete stage one,
**2x** during the warm-up and **3x** afterwards. The LR schedule is therefore keyed to a rate
that changes at every phase boundary — `--wm_warm_up`-scale numbers in `OptimizerConfig` are not
in learner steps.

One hazard: the free-bits clamp still applies to `dyn`, so if the dynamics loss sits below
`--free_bits_threshold` the world model stops learning **altogether** in stage two — `dyn` is the
only term left that could train anything. Pass `--free_bits_threshold 0` if the latent is small
enough for that to be a risk.

**How stage one ends.** `DreamerMA.update_stage_one` keeps the last `--loss_check_window` (100)
values of the compound WM loss in a `deque`, fits a least-squares line once it is full, and ends
stage one when the drop that line predicts across the window falls below `--stage_one_tol` (1e-3) as
a *fraction* of the current loss level: `abs(slope * window) < tol * abs(mean(window))`. Three things
follow from that formula:

- The tolerance is **relative** on purpose. The compound loss is dominated by `dec`/`is_obs_dec`/
  `is_rec_pred`, whose scale follows the observation size, so an absolute delta would need retuning
  per game.
- A **steeply rising** loss does not satisfy the test and will keep stage one running.
  `--stage_one_max_steps` (default -1, no cap) is the backstop.
- Stage one can never end before the window has filled, so `--loss_check_window` doubles as its
  minimum length. Fitting a line to a mean of *absolute* successive differences was rejected: at a
  genuine plateau that statistic settles at the minibatch noise floor rather than near zero, so the
  tolerance would encode sampling noise instead of convergence.

**Reading a run.** `dyn` prints as exactly `0` during stage one and nonzero after; `rep` is nonzero
during stage one only under `--vq_vae_posterior` and `0` everywhere else; under a hard or complete
stage one *every* actor-critic metric is 0 until it ends. So the printed metrics say which phase a
run is in — except under `--complete_two_stage`, where the stage-two losses no longer track what is
learning; `--report_gradnorms` does (every world-model network but `dyn` reads exactly `0`).
The transition is also logged (`Stage one ended at step N because ...`, naming the step imagination
will start at).

Three consequences worth knowing before changing any of this:

- `DreamerMA.train_step` is the **single source of truth** for `should_imagine`, which it passes into
  `actor_critic.step(...)`. The learners used to each compute it from their own `learner_steps`, which
  cannot express this ordering: under a soft stage one the learner steps throughout it, so its counter
  would reach `wm_warm_up_period` while still inside stage one and it would imagine far too early.
  The `hard_two_stage` mirror fields on `RNaDConfig`/`MMDConfig` are the remnant of that scheme — now
  deprecated and read by nothing, kept only because a `chex.dataclass` tolerates a *missing* field but
  not a stored field the class no longer declares.
- `DreamerActorCritic` (REINFORCE) had no imagination gate at all and imagined from step 0. It now
  honours the same boundary, via a new static `imagine` argument on `update_paramaters_and_model`.
  When skipping, the incoming `return_range` must be threaded through to `real_loss` — normally the
  imagination branch advances it first.
- Both `two_stage_warm_up` (in `update_world_model`) and `imagine` are **static** jit arguments, so
  each phase transition costs one extra trace, not a per-step branch. The stage-one latch is monotone
  for exactly this reason: it must never flap.

Because stage one is dynamic, the step at which imagination starts is **model state**, not a config
value. `DreamerMA.imagination_start_step()` reports it, `__getstate__` persists it alongside the latch
and window (so `--continue_train` resumes in the right phase), and `eval/actor_critic_evaluate.py`
writes it into `config.json` as `imagination_start_step` for `plotting/plot_total_steps_comparison.py`
— which otherwise recomputes the boundary from `wm_warm_up_period` alone and would draw it in the
wrong place.

`dyn`/`rep` still pass through `jnp.maximum(free_bits_clip_threshold, ...)`, so a VQ-VAE term that
sits under `--free_bits_threshold` (easy with a small latent) contributes no gradient during stage
one even though it is reported — and, because of the clamp, a disabled term and a clamped one both
print as the threshold times their beta rather than a bare 0. Pass `--free_bits_threshold 0` when you
need the printed `rep`/`dyn` to distinguish "off" from "clamped".

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
bugfix in `dreamer_ma.py` / `rnad_dreamer.py` does not propagate here automatically. It does inherit
`DreamerMA.train_step`, so the two-stage warm-up above applies unchanged, and its `update_world_model`
mirrors both the gating and the `--vq_vae_posterior` branch (added there 2026-09-09; before that the
decentralized copy only ever computed the KL representation loss and silently ignored the flag).

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
- `--wm_warm_up` defaults to 1000, so short NashDreamer runs never exercise the imagination path. Set
  it to 1 when smoke-testing a new game. With `--soft_two_stage`/`--hard_two_stage` there is a second
  boundary in front of it: `--loss_check_window` defaults to 100, which is the minimum length of stage
  one, so also drop that (and raise `--stage_one_tol`, or set `--stage_one_max_steps`) or a short run
  never leaves stage one at all.
