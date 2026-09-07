import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from copy import deepcopy


def my_softmax(logits: np.ndarray):
  logits_shifted = logits - np.max(logits)
  exp_logits = np.exp(logits_shifted)
  normalization = np.sum(exp_logits)

  p = exp_logits / (normalization + (normalization == 0))
  return p

def best_reponses(matrix: np.ndarray, policies: list[np.ndarray]):
  p1_action_values, p2_action_values = np.sum(matrix * policies[1][None, ...], axis=-1), -np.sum(matrix * policies[0][..., None], axis=0)
  p1_br_action, p2_br_action = np.argmax(p1_action_values), np.argmax(p2_action_values)
  p1_br_val, p2_br_val = p1_action_values[p1_br_action], p2_action_values[p2_br_action]
  p1_br, p2_br = np.eye(matrix.shape[0])[p1_br_action], np.eye(matrix.shape[1])[p2_br_action]
  return p1_br, p1_br_val, p2_br, p2_br_val

def get_nash_conv(matrix: np.ndarray, policies: list[np.ndarray]):
  p1_br, p1_br_val, p2_br, p2_br_val = best_reponses(matrix, policies)
  return p1_br_val + p2_br_val


def reinforce_update(matrix: np.ndarray, rng_gen: np.random.Generator,
                     old_logits: list[np.ndarray], old_values: np.ndarray
                     , lr: float = 0.1, eta:float = 0.2, alpha: float=0.1):
 
  new_values = np.zeros_like(old_values)
  advantages = np.zeros_like(old_values)
  new_logits = [None for l in old_logits]
  policies = [my_softmax(l) for l in old_logits]
  actions = [rng_gen.choice(matrix.shape[i], p = pols) for i, pols in enumerate(policies)]
  achieved_util = matrix[actions[0], actions[1]]
  # We compute adv as Q(taken_action) - V
  # Gradient of the action that was
  # taken = (1 - pi(a)) * adv = -adv * (pi(a) - 1)
  # for action that was not taken
  # -adv * pi(a)

  #The derivation of the entropy bonus
  # is pi * (sum(pi * logit) - logit)
  # or, alternatively, since logit is
  # just log(pi) + log of the normalization,
  # (which we can ignore due to softmax shift invariance)
  # we can write this as -pi * (ent(pi) + log(pi))

  #So, the final update rule can
  # then be written as -lr * (adv * (pi - one_hot(a)) + eta * pi * (ent(pi) + log(pi)))
  for i, (l, val, pi) in enumerate(zip(old_logits, old_values, policies)):
    #2p0s game
    util = achieved_util * (1 - 2 * (i))
    adv = util - val
    #Update our estimates
    # We update our value estimate
    # with TD update
    # in our matrix game, the
    # next state is always terminal
    new_value = val + alpha * (util - val)
    new_values[i] = new_value
    advantages[i] = adv
    #Safe log
    log_pi = np.where(pi > 1e-8, np.log(pi), 0)
    ent_pi = -np.sum(pi * log_pi)

    policy_update_dir = adv * (pi - np.eye(matrix.shape[i])[actions[i]])
    entropy_update_dir = eta * pi * (ent_pi + log_pi)
    new_logit = l  - lr *(policy_update_dir + entropy_update_dir)
    new_logits[i] = new_logit


  return new_logits, advantages, new_values

  


def run_experiment(matrix: np.ndarray,
                   init_logits: np.ndarray, 
                   etas: list[float], 
                   init_seed: int = 42, 
                   num_seeds: int = 5,
                   num_iters: int = 1000, 
                   lr: float = 0.01,
                   alpha: float = 0.01):
    
    results = {}
    
    for eta in etas:
        print(f"Running for eta: {eta}")
        eta_nash_convs = []
        
        for seed_offset in range(num_seeds):
            seed = init_seed + seed_offset
            rng_gen = np.random.default_rng(seed)
            values = np.zeros(2)
            logits = deepcopy(init_logits)
            run_nash_conv = []
            
            for _ in range(num_iters):
                # Update step
                logits, _, values = reinforce_update(matrix, rng_gen, logits, values, lr, eta, alpha)
                
                # Logging
                policies = [my_softmax(l) for l in logits]
                nash_conv = get_nash_conv(matrix, policies)
                run_nash_conv.append(nash_conv)
            
            eta_nash_convs.append(run_nash_conv)
        
        # Store result as (num_seeds, num_iters) array
        results[eta] = np.array(eta_nash_convs)
        
    return results

def plot_results(results: dict, plot_name: str):
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))

    for idx, (eta, data) in enumerate(results.items()):
        # data shape: (num_seeds, num_iters)
        steps = np.arange(data.shape[1])

        mean_conv = np.mean(data, axis=0)
        std_conv = np.std(data, axis=0)

        ax.plot(steps, mean_conv, label=f"eta={eta}", color=colors[idx])
        ax.fill_between(steps,
                        mean_conv - std_conv,
                        mean_conv + std_conv,
                        color=colors[idx], alpha=0.2)

    ax.set_xlabel("Iterations", fontsize=20)
    ax.set_ylabel("NashConv", fontsize=20)
    #ax.set_yscale('log')
    #ymin, ymax = ax.get_ylim()
    #ax.set_ylim(bottom=min(ymin, 1e-1), top=max(ymax, 1.0))
    #ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))
    #ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
    ax.tick_params(axis='both', labelsize=15)
    ax.xaxis.get_offset_text().set_fontsize(15)
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend(fontsize=15, loc='upper right')
    plt.tight_layout()
    plt.savefig(f"{plot_name}.pdf")
    plt.close()

def main():
    rps = np.asarray([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
    
    etas_to_test = [3e-4, 0.2]
    R, C = rps.shape
    logits = [np.zeros(R), np.zeros(C)]
    pure_logits = deepcopy(logits)
    pure_logits[0][0] = 5
    pure_logits[1][0] = 5
    
    pure_results = run_experiment(
        matrix=rps,
        init_logits=pure_logits,
        etas=etas_to_test,
        init_seed=42,
        num_seeds=10,
        num_iters=10000,
        lr=0.01,
        alpha=0.01
    )
    # Plot
    plot_results(pure_results, "rps_reinforce_pure_init")

    uniform_results = run_experiment(
        matrix=rps,
        init_logits=logits,
        etas=etas_to_test,
        init_seed=42,
        num_seeds=10,
        num_iters=10000,
        lr=0.01,
        alpha=0.01
    )
    # Plot
    plot_results(uniform_results, "rps_reinforce_uniform_init")

if __name__ == "__main__":
    main()