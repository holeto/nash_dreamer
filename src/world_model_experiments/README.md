# World Model Experiments

Three quality checks against a trained `DreamerMA` checkpoint, each walking the game tree under the
model's own policy (pruned by `--policy_eps`) and each with a matching root-level `<name>_evaluate.sh`
wrapper. Plotted by the scripts under [src/plotting/](../plotting/README.md).

- **`chance_marginal_eval.py`** — compares the world model's predicted next-observation distribution
  against the true one at every reached node. Each model continuation is decoded to an observation and
  snapped to the closest real reachable outcome; a continuation too far from any real outcome (by
  L∞ distance) is counted as an error. Measures raw reconstruction accuracy of the dynamics. 
- **`posterior_collapse_eval.py`** — at every chance node, checks whether the encoder's posterior
  (which sees the true realized outcome) actually differs from the prior (which doesn't), via
  KL(posterior‖prior). Catches a model that has stopped using its stochastic state to represent
  chance-outcome information at all. Primarily meant to diagnose collapse (the KL vanishes) or representation drift (the KL significantly higher than ground truth expected KL)

- **`world_model_sampling_eval.py`** — samples full imagined rollouts (own policy, latents from the
  filtered prior) in parallel with the real game, and flags steps where the reconstructed
  observation, reward, terminal flag or legal mask diverge from the real ones past a threshold.
  Measures whether imagined trajectories — what the actor-critic actually trains on — stay valid over
  a full rollout, not just one step. Only measures whether generated imagined trajectories map onto some real trajectories, not whether the probability of the trajectories is correct.


All three accept `--restore_step -1` to sweep every checkpoint in a seed directory.
