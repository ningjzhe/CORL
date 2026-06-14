# Model Architecture and Hyperparameters (Algorithm Owner)

Source: `/code/CORL/algorithms/offline/iql.py`, `/code/analysis/iad_iql/scripts/iad_common.py`, saved `config.yaml` under each `value_seed*/`.

---

## Network architectures

All MLP blocks use **ReLU** activations between layers (CORL `MLP` default). Output layers have no activation unless noted.

### TwinQ (critic)

- Input: concatenated `[state, action]`
- Architecture: `[state_dim + action_dim] → 256 → 256 → 1` (**two** independent Q networks `q1`, `q2`)
- Default: `hidden_dim=256`, `n_hidden=2`
- Forward: `Q(s,a) = min(Q1(s,a), Q2(s,a))`

### ValueFunction V

- Input: `state`
- Architecture: `[state_dim] → 256 → 256 → 1`
- Default: `hidden_dim=256`, `n_hidden=2`

### GaussianPolicy (actor)

- Input: `state`
- MLP trunk: `[state_dim] → 256 → 256 → action_dim` with **Tanh** on mean output
- Learnable `log_std` per action dimension (clamped to `[LOG_STD_MIN, LOG_STD_MAX]` = **[-20, 2]**)
- Action: `tanh(mean) * max_action` at eval (deterministic mean in our actor extraction eval)
- Default: `hidden_dim=256`, `n_hidden=2`, `actor_dropout=None`

---

## IQL value training (seed0 / seed1 per env)

| Hyperparameter | Value |
|----------------|------:|
| Optimizer | Adam |
| Q learning rate (`qf_lr`) | 3e-4 |
| V learning rate (`vf_lr`) | 3e-4 |
| Actor learning rate (`actor_lr`) | 3e-4 |
| Batch size | 256 |
| Discount γ (`discount`) | 0.99 |
| Target soft update τ (`tau`) | 0.005 |
| IQL expectile τ (`iql_tau`) | 0.7 |
| Advantage temperature β (`beta`) | 3.0 |
| Max timesteps | 1,000,000 |
| Eval frequency | 5,000 |
| Mid-training eval episodes | 10 |
| Replay buffer size | 2,000,000 |
| State normalization | **True** (mean/std from dataset, eps=1e-3) |
| Reward normalization | **False** |
| Checkpoints | every 5k → `checkpoint_{step}.pt` |

Value checkpoints store `qf`, `vf`, `actor` state dicts at step 999999.

---

## Actor extraction (frozen Q/V)

| Hyperparameter | Value |
|----------------|------:|
| Trainable module | **Actor only** (Q/V frozen, verified max param delta = 0) |
| Updates | 100,000 |
| Batch size | 256 |
| Actor optimizer | Adam, lr = **3e-4** |
| β (weight temperature) | 3.0 |
| EXP_ADV_MAX (weight clip) | **100.0** |
| Eval during training | every 5,000 steps, **10 episodes**, eval seed 0 |
| Final reporting eval | **50 episodes**, eval seeds **0–49**, deterministic mean action |
| Replicates | Rep A (actor seed 12345, batch seed 42); Rep B (54321, 43) |

---

## Online advantage (all methods)

For each value system `i`:

\[
A_i(s,a) = \min(Q_{i,1}(s,a), Q_{i,2}(s,a)) - V_i(s)
\]

Computed **online** from frozen networks (no precomputed advantage table, no per-step flatten).

---

## Method: `standard_iql`

Uses **seed0** value system only:

\[
w(s,a) = \exp(\beta \cdot A_0(s,a)), \quad w \leftarrow \min(w, \text{EXP\_ADV\_MAX})
\]

Actor loss: `mean(-w * log π(a|s))`.

---

## Method: `ensemble_mean` (main)

Uses **seed0 + seed1**:

\[
\mu_A = \frac{A_0 + A_1}{2}, \quad w = \exp(\beta \cdot \mu_A), \quad w \leftarrow \min(w, \text{EXP\_ADV\_MAX})
\]

---

## Method: `iad_lambda_1.0`

\[
\mu_A = \frac{A_0 + A_1}{2}, \quad u_A = \frac{|A_0 - A_1|}{2}
\]
\[
\hat{u}_A = \mathrm{clip}\left(\frac{u_A}{\mathbb{E}[u_A]+\epsilon}, 0, 5\right)
\]
\[
\log w = \beta \cdot \mu_A - \lambda \cdot \hat{u}_A, \quad \lambda = 1.0
\]

Then mass-matching to base ensemble weight mean, clip to EXP_ADV_MAX.

Implementation: `iad_common.compute_weights()` variant `iad_lambda_1.0`.

---

## Method: `shuffled_iad_lambda_1.0`

Same as IAD, but **`hat_u_a` is permuted randomly across the batch** before penalty:

\[
\hat{u}_A^{\text{shuf}} = \hat{u}_A[\pi], \quad \pi \sim \text{uniform permutation}
\]

Mechanism control: destroys cross-sample disagreement structure.

---

## Best-checkpoint protocol

- Save `actor_step_{5k..100k}.pt`, track best 10-ep D4RL → `best_actor.pt`
- Save `actor_final.pt` at step 100k
- Re-evaluate **best** and **final** with 50 episodes each
- Report **Best 50-ep** (primary) and **Final 50-ep** (overtraining diagnostic)

---

## Not in main results

- seed2 value systems
- `ensemble_mean_3q`
- λ=0.5 IAD under best-checkpoint protocol
- CQL / DT / UA / UGH (experiment owner)
