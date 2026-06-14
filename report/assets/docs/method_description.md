# Method Description (Algorithm Owner)

## 1. Original IQL actor extraction

Standard Implicit Q-Learning (IQL) trains a Q-function, value function, and policy jointly. For **policy extraction**, IQL uses advantage-weighted behavioral cloning:

\[
\mathcal{L}_{\text{actor}} = \mathbb{E}_{(s,a) \sim \mathcal{D}}\left[ -\log \pi(a|s) \cdot w(s,a) \right]
\]

where the weight is derived from the **online** advantage using the **same** trained Q and V:

\[
A(s,a) = \min(Q_1(s,a), Q_2(s,a)) - V(s), \quad w = \exp(\beta \cdot A), \quad w \leftarrow \min(w, w_{\max})
\]

In our codebase this corresponds to **`standard_iql`**: one frozen value system (seed0 Q/V only).

---

## 2. Frozen Q/V actor extraction

After IQL value training to 1M steps, we **freeze** Q and V and train only the actor for 100k gradient steps:

- No Q/V gradient updates; freeze verified at start/end of actor training
- Same batch size (256), same state normalization as value training
- TwinQ min − V for online advantage (CORL IQL convention)
- β = 3.0, EXP_ADV_MAX clipping as in CORL

This isolates **policy extraction** from value learning.

---

## 3. `standard_iql`

Uses **one** value system (checkpoint seed0):

\[
A_1(s,a) = Q_0(s,a) - V_0(s), \quad w = \exp(\beta A_1)
\]

Baseline for our experiments: single-value-system weighted BC with frozen Q/V.

---

## 4. `ensemble_mean` (main method)

Train **two independent** IQL value systems (seed0 and seed1, same env/hyperparams, different RNG). At actor extraction:

\[
A_0 = Q_0(s,a) - V_0(s), \quad A_1 = Q_1(s,a) - V_1(s)
\]
\[
\mu_A = \frac{A_0 + A_1}{2}, \quad w = \exp(\beta \cdot \mu_A), \quad w \leftarrow \min(w, w_{\max})
\]

**Interpretation:** Ensemble the advantage estimates from two independently trained value systems, then apply standard IQL exponential weighting. No explicit disagreement penalty.

**Why it works:** Complementary value estimates reduce variance in advantage targets; simpler and more stable than penalizing disagreement.

---

## 5. IAD (λ = 1.0)

**Inter-system Advantage Disagreement** weighting uses mean advantage minus a normalized disagreement penalty:

\[
\mu_A = \frac{A_0 + A_1}{2}, \quad u_A = \frac{|A_0 - A_1|}{2}
\]
\[
\hat{u}_A = \mathrm{clip}\left(\frac{u_A}{\mathbb{E}[u_A] + \epsilon}, 0, 5\right)
\]
\[
\log w = \beta \cdot \mu_A - \lambda \cdot \hat{u}_A, \quad w = \exp(\mathrm{clip}(\log w))
\]

With **mass matching** so mean weight matches the ensemble base weight, then clip to EXP_ADV_MAX.

Our experiments use **λ = 1.0** (`iad_lambda_1.0`).

---

## 6. `shuffled_iad_lambda_1.0` (mechanism control)

Same as IAD, but **shuffle** \(\hat{u}_A\) across the batch before applying the penalty:

\[
\hat{u}_A^{\text{shuf}} = \hat{u}_A[\pi], \quad \pi \text{ random permutation}
\]

If IAD’s gain comes from the **disagreement structure** (not just down-weighting high-magnitude batches), shuffled should perform worse than IAD. If IAD ≈ shuffled, the penalty may not use meaningful disagreement signal.

---

## 7. Why `ensemble_mean` is the main method (not IAD)

| Criterion | ensemble_mean | IAD λ=1.0 |
|-----------|---------------|-----------|
| vs standard_iql (3 envs, Best & Final 50-ep) | **Consistently positive Δ** | Mixed; Final often worse than ensemble on Hopper/Walker2d |
| Cross-replicate stability | Better on Hopper Rep A; Walker2d Rep B stable | Large overtraining on Walker2d Rep A (gap 18.9); Hopper Rep B noisy |
| vs shuffled control | N/A (no shuffled variant) | Does not clearly beat shuffled on Final 50-ep (Walker2d Δ = −8.47) |
| Complexity | Simple mean of advantages | Extra hyperparameter λ + normalization + mass matching |
| Original project name | Aligns with “ensemble policy extraction” framing | “IAD-IQL” — retained as **ablation** |

**Conclusion:** We report **`ensemble_mean`** as the primary proposed method. IAD remains an ablation studying whether disagreement-aware down-weighting helps beyond averaging.

---

## Implementation reference

- Shared utilities: `/code/analysis/iad_iql/scripts/iad_common.py` (`compute_weights`, `compute_online_advantages`)
- Actor training: `/code/analysis/iad_iql/scripts/train_frozen_iad_actor_best.py`
- Value training: `/code/analysis/iad_iql/scripts/train_value_system.py` (CORL IQL, 1M steps)
