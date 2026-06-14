# Algorithm Owner Results Summary

**Protocol:** Best-checkpoint actor extraction (frozen Q/V, 100k updates, Rep A/B, Best/Final 50-ep re-eval)  
**Environments:** halfcheetah-medium-v2, hopper-medium-v2, walker2d-medium-v2  
**Methods:** standard_iql, ensemble_mean, iad_lambda_1.0, shuffled_iad_lambda_1.0  
**Value systems:** IQL seed0 + seed1 per env (1M steps each); **not** seed2  
**Actor replicates:** N=2 per method per env (Rep A: actor seed 12345, batch 42; Rep B: 54321, 43)

---

## Headline findings

1. **`ensemble_mean` vs `standard_iql`:** Positive Best 50-ep and Final 50-ep deltas in **all three environments** (see Table C).
2. **IAD λ=1.0:** Does **not** consistently beat `ensemble_mean`; unstable across envs and replicates → **ablation only**.
3. **Overtraining:** Hopper shows large best–final gaps (avg ~17.4 across all Hopper runs); Walker2d moderate (avg ~3.2); HalfCheetah minimal (~0.16).
4. **Best-checkpoint protocol:** Required for fair peak performance reporting; prior final-only Hopper runs could not re-evaluate best actor (see stability audit).

---

## Cross-environment comparison (mean over Rep A/B, Best 50-ep)

| Env | standard_iql | ensemble_mean | iad λ=1.0 | shuffled IAD |
|-----|-------------:|--------------:|----------:|-------------:|
| HalfCheetah | 47.05 | 48.84 | 48.89 | 49.08 |
| Hopper | 46.35 | **67.58** | 62.63 | 51.27 |
| Walker2d | 81.72 | **86.78** | 87.54 | 87.02 |

**Main method ranking (our side):** `ensemble_mean` is the recommended primary method because it improves over `standard_iql` consistently without the instability of IAD weighting.

---

## ensemble_mean − standard_iql (Table C summary)

| Env | Best 50-ep Δ | Final 50-ep Δ |
|-----|-------------:|--------------:|
| HalfCheetah | +1.79 | +1.96 |
| Hopper | +21.24 | +16.97 |
| Walker2d | +5.06 | +2.24 |

All positive → supports main claim on our data.

---

## IAD vs ensemble (Table C summary)

| Env | IAD − ensemble (Best) | IAD − ensemble (Final) |
|-----|----------------------:|-----------------------:|
| HalfCheetah | +0.05 | +0.24 |
| Hopper | **−4.95** | −0.74 |
| Walker2d | +0.76 | **−5.86** |

IAD does not reliably dominate ensemble; Final 50-ep especially unstable on Hopper/Walker2d.

---

## Replicate stability notes

- **HalfCheetah:** All methods stable; gaps near zero.
- **Hopper:** High Rep A vs Rep B variance for all methods (ensemble Best: 71.7 vs 63.4; Final: 59.2 vs 33.1).
- **Walker2d:** Rep B more stable; Rep A IAD shows large overtraining (gap 18.9).

---

## Value system quality (reference)

| Env | seed0 final D4RL (value eval) | seed1 final D4RL | Pearson(A0,A1) |
|-----|------------------------------:|-----------------:|---------------:|
| HalfCheetah | ~48.5 | ~48.3 | ~0.53 |
| Hopper | ~60.7 / ~54.8 | (see hopper logs) | ~0.43 |
| Walker2d | 78.01 | 78.06 | 0.237 |

Low Pearson is expected; compatibility checks passed on structure/preprocessing/finite stats.

---

## What we cannot claim (pending experiment owner)

- Superiority vs CQL, DT, UA_CQL, UGH_CQL
- 3-seed mean ± std tables aligned with baseline section
- Training-speed comparisons under their hardware schedule

See `baseline_section_placeholder.md` and `questions_for_experiment_owner.md`.
