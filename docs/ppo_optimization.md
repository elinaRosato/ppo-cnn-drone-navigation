# PPO Optimization Techniques

A practical guide to PPO stability and hyperparameter optimization, with analysis
of what applies to this drone obstacle avoidance project.

---

## Why PPO is sensitive

PPO stability is not a property of the algorithm — it is a property of your exact
configuration. The algorithm relies on four approximations working together:

- Clipped policy objective (keeps updates inside a trust region)
- Advantage estimates (noisy and scale-dependent)
- Value bootstrap (can drift when rewards shift)
- Batching (changes gradient noise and effective step size)

**The core mental model:** every config change is ultimately a gradient scale change.
Scale changes everything: how far the policy moves, how well the value function tracks
returns, how much entropy survives.

---

## The 9 config knobs that flip outcomes

### 1. Clip range
Controls how much the policy probability ratio can change before the objective stops
rewarding larger steps.

- **Too small (e.g. 0.1):** clip fraction near 0, learning is slow, entropy barely
  moves — the policy is being held back even when it could safely improve.
- **Too large (e.g. 0.3+):** KL spikes, reward climbs then collapses — the policy
  takes steps that look good short-term but destabilize the distribution.

**Diagnostic:** track `approx_kl` and `clip_fraction` together. High clip fraction
+ KL spike = stepping off a cliff, not learning faster.

**Typical range:** 0.1–0.2. Start at 0.2, drop to 0.1 if KL is unstable.

---

### 2. Learning rate and schedule
A 2× change in LR is obvious. The painful failures come from subtler sources:

- LR decay with too short a horizon (policy freezes before it converges)
- Warmup that ends before advantages settle
- Adam epsilon changes that alter early-phase behavior

**Diagnostic:** log gradient norms separately for the policy head and value head. If
value gradients dominate early, PPO becomes "value fitting with a side of policy."

**Typical range:** 1e-4 to 3e-4. Smaller LRs are more stable; the W&B sweep study
found LR is the single most important hyperparameter.

---

### 3. Batch size, minibatch size, and epochs
These three interact in non-obvious ways. You can hold LR constant and still change
the effective gradient step by changing any of them.

- **More epochs** = more reuse of stale rollout data = overfitting the old policy =
  KL climbs within the update, performance collapses after looking strong.
- **Tiny minibatches** = high gradient noise = unstable value fitting = oscillatory
  rewards.
- **Larger total batch** = more stable advantage estimates = smoother updates.

**Rule of thumb:** increase batch size before increasing epochs when you need more
signal. Fix batch size first, then tune epochs.

**Typical values:** n_epochs=4–10, minibatch=64–256, n_steps=2048–4096.

---

### 4. Target KL
Adds a hard stop: if the KL divergence exceeds the target during a batch update,
stop further epochs on that batch.

- **Too strict:** stable but stagnant — entropy stays high, rewards plateau, the
  policy is being throttled even when it could improve.
- **Too loose (or absent):** sudden KL runaway when advantages spike, usually followed
  by reward collapse.

**This is one of the most reliable stability fixes.** The pattern:
```python
if approx_kl > target_kl:
    break  # stop epochs on this batch
```
Often turns "sometimes diverges" into "predictable."

**Typical range:** 0.01–0.05. Start at 0.02.

---

### 5. Advantage normalization
A single flag that changes the entire training regime.

Without normalization, advantage scale depends on reward magnitude, episode length,
and GAE parameters. A few high-reward episodes can spike gradients and cause KL
runaway. With normalization (per minibatch), gradient scale is implicitly controlled
and the policy is insensitive to reward magnitude.

**Symptoms without normalization:**
- Policy loss magnitude swings wildly between iterations
- Sudden KL spikes after a few high-reward episodes

**Practical move:** normalize per minibatch (SB3 does this by default). Monitor raw
advantage mean/std anyway — normalization can hide reward scaling problems.

---

### 6. GAE lambda (λ) and gamma (γ)
These look like theory knobs but behave as stability knobs:

- **High λ (e.g. 0.98):** low bias, high variance — noisy advantages → oscillating
  policy updates, unstable KL.
- **Low λ (e.g. 0.90):** smoother advantages, but the policy gets mis-credited for
  actions it didn't take. Trains stably but may plateau early.

**Diagnostic:** if KL is unstable and advantage std is high, try lowering λ slightly
(0.95 → 0.90) before reaching for heavier fixes.

**Typical values:** γ=0.99, λ=0.90–0.98.

---

### 7. Value function coefficient and value clipping
The value loss can dominate the optimizer, especially after a curriculum shift or
reward rescaling.

- **vf_coef too high:** value gradients crowd out policy gradients, policy stops
  moving, entropy collapses.
- **vf_coef too low:** value function lags, advantages stay noisy, policy gets bad
  credit assignment.
- **Value clipping (clip_range_vf):** mirrors the policy clip — prevents the value
  function from making overly large updates. Useful when returns are non-stationary
  (e.g. curriculum shifts).

**Diagnostic:** `explained_variance`. Persistent negative values = red alarm, not a
curiosity. Oscillating variance = value function chasing a moving target.

**Typical values:** vf_coef=0.5, clip_range_vf=0.2–0.5.

---

### 8. Entropy coefficient
Entropy bonus prevents premature policy collapse. Too much washes out the advantage
signal — the policy looks stable but learns nothing.

- **Too low:** entropy collapses early, policy becomes overconfident, KL spikes when
  the confident policy gets a bad batch.
- **Too high:** entropy stays high regardless of rewards, policy ignores signal,
  rewards plateau.

**The principle:** entropy at the right time. High early for exploration, lower later
for exploitation. An adaptive schedule (rise when performance declines, decay when
it improves) is more robust than a fixed schedule.

**Typical range:** 0.0–0.1, schedule from high to low over training.

---

### 9. Reward scaling and normalization
Reward scale changes advantage scale, which changes gradient scale, which changes KL.
This is the entire chain — a tiny rescaling (×0.1) can turn "diverging" into "stable."

**Symptoms of reward scale problems:**
- Advantage std grows monotonically with training (bigger rewards → bigger updates)
- KL gradually trends upward as the policy improves
- Different tasks "need" different LRs for no obvious reason

**Options:**
1. Manual scaling (e.g. divide all rewards by 100)
2. Running normalization via `VecNormalize` (normalizes observations and/or rewards
   with a running mean/std)
3. Return normalization (normalize the bootstrapped returns, not just rewards)

**Always log:** reward mean/std, return mean/std, advantage std.

---

## Diagnostic checklist

When a run flips, triage in this order:

| Signal | What to look for |
|--------|-----------------|
| `approx_kl` + `clip_fraction` | Stepping too far? Not stepping at all? |
| Advantage mean/std | Is normalization on? Are raw values blowing up? |
| `explained_variance` | Is the critic tracking reality? |
| Entropy curve | Collapsing too early? Staying too high? |
| `n_epochs` + minibatch size | Over-optimizing stale data? |

**Best single addition:** plot `approx_kl` across epochs within one update. If KL
climbs each epoch on the same batch, you are over-optimizing stale rollout data —
reduce epochs or add target_kl early stopping.

---

## Hyperparameter optimization strategy

Trying all combinations is infeasible. A 500-run grid search over 12 hyperparameters
would take months. The efficient approach is **Bayesian optimization**: use past
trials to guide where to search next, converging on good regions in 50–150 trials
instead of thousands.

### Why Bayesian over grid/random
- Grid search wastes budget on unimportant dimensions
- Random search is better but blind
- Bayesian builds a surrogate model of "which config gives good reward" and samples
  intelligently from it

### What the W&B sweep study found (209 runs, PickAndPlace task)
1. **Learning rate** is the most important hyperparameter overall — lower is more stable
2. Among good runs, **clip_range** and **target_kl** are the key differentiators —
   both control how large policy updates are allowed to be
3. Among the best runs, **higher target_kl** wins — it allows larger updates when the
   policy can handle them, leading to faster convergence
4. **Normalize advantages** consistently appears in top runs

### Recommended tool: Optuna
Optuna is the simplest way to run Bayesian hyperparameter search with SB3 without
needing a W&B account. It runs trials in sequence (or parallel), prunes bad runs
early, and plots importance rankings.

```python
import optuna
from stable_baselines3 import PPO

def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.15, 0.2, 0.25])
    n_epochs = trial.suggest_int("n_epochs", 3, 10)
    gae_lambda = trial.suggest_float("gae_lambda", 0.90, 0.99)
    target_kl = trial.suggest_float("target_kl", 0.005, 0.05, log=True)
    vf_coef = trial.suggest_float("vf_coef", 0.3, 1.0)

    model = PPO("CnnPolicy", env, learning_rate=lr, clip_range=clip_range,
                n_epochs=n_epochs, gae_lambda=gae_lambda,
                target_kl=target_kl, vf_coef=vf_coef)
    model.learn(total_timesteps=300_000)

    # Evaluate over N deterministic episodes
    mean_reward = evaluate(model, n_episodes=20)
    return mean_reward

study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=50)
```

Each trial runs for fewer steps (300K instead of 2M) to quickly filter bad configs.
50 trials × 300K steps = 15M steps total — the same as ~7–8 full training runs.

---

## Current config analysis (drone avoidance project)

### Current PPO hyperparameters

| Parameter | Current value | Assessment |
|-----------|--------------|------------|
| `learning_rate` | 1e-4 | Good — on the conservative side |
| `n_steps` | 2048 | Fine for 1 env |
| `batch_size` | 256 | Fine — 8 minibatches per rollout |
| `n_epochs` | 5 | Reasonable |
| `gamma` | 0.99 | Standard |
| `gae_lambda` | 0.95 | Standard — watch advantage std |
| `clip_range` | 0.2 | Standard |
| `ent_coef` | 0.05 adaptive | Good — adaptive system in place |
| `vf_coef` | 0.5 | Standard |
| `max_grad_norm` | 0.5 | Good safety net |
| `target_kl` | None | **Missing — high priority** |
| `clip_range_vf` | None | Missing — consider adding |
| Reward normalization | None | **Missing — rewards are ±100** |
| Advantage normalization | SB3 default (on) | Good |

### Changes ranked by expected impact

**High priority:**

1. **Add `target_kl`** — the single most reliable stability fix from the blog.
   Value loss spikes and KL runaway have both appeared in past runs. `target_kl=0.01`
   would stop epochs early when the policy is being updated too aggressively.

2. **Reward normalization** — current rewards span a large range: +100 (goal),
   -100 (collision), small per-step values (-2 to +0.05). This creates high advantage
   variance and is likely contributing to the noisy value loss seen in logs. Options:
   - Divide all rewards by 100 (simple, interpretable)
   - Wrap with `VecNormalize(norm_reward=True)` (adaptive)

**Medium priority:**

3. **Add `clip_range_vf=0.2`** — value loss has spiked to 270+ in past runs. Clipping
   value updates prevents the critic from chasing non-stationary returns after
   curriculum transitions.

4. **Lower `gae_lambda` to 0.90`** if advantage std is high. The sparse long-horizon
   episodes (up to 2000 steps) make high-lambda estimates noisy — actions near the
   start get blended credit from much later outcomes.

**Lower priority / experimental:**

5. **Optuna sweep** for learning_rate, clip_range, n_epochs, gae_lambda, vf_coef,
   target_kl — run 50 trials × 300K steps each to find the optimal config for this
   specific environment before committing to a 2M-step run.

6. **Increase `n_steps` to 4096** — longer rollouts give more stable advantage
   estimates for long episodes. Trade-off: less frequent policy updates.
