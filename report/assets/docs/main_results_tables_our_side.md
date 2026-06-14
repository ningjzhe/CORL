# Main Results Tables (Algorithm Owner)

**Source:** `/code/analysis/iad_iql/best_checkpoint_protocol/results/best_protocol_all_runs.csv`  
**Metric:** D4RL normalized score (0–100 scale). **Best/Final 50-ep** = deterministic mean action, eval seeds 0–49.  
**N:** 2 actor replicates per method per env (Rep A, Rep B). **Not** 3 value seeds.

---

## Table A: Per-run results

| Env | Rep | Method | Best Step | Best 10-ep | Best 50-ep | Final 50-ep | Best-Final Gap |
|-----|-----|--------|----------:|-----------:|-----------:|------------:|---------------:|
| halfcheetah | A | ensemble_mean | 100000 | 49.04 | 48.94 | 48.94 | 0.00 |
| halfcheetah | A | iad_lambda_1.0 | 100000 | 48.83 | 48.85 | 48.85 | 0.00 |
| halfcheetah | A | shuffled_iad_lambda_1.0 | 100000 | 49.42 | 49.29 | 49.29 | 0.00 |
| halfcheetah | A | standard_iql | 80000 | 47.03 | 47.13 | 46.68 | 0.45 |
| halfcheetah | B | ensemble_mean | 40000 | 48.85 | 48.74 | 48.37 | 0.37 |
| halfcheetah | B | iad_lambda_1.0 | 100000 | 49.15 | 48.92 | 48.92 | 0.00 |
| halfcheetah | B | shuffled_iad_lambda_1.0 | 75000 | 49.07 | 48.86 | 48.67 | 0.19 |
| halfcheetah | B | standard_iql | 75000 | 47.20 | 46.97 | 46.72 | 0.25 |
| hopper | A | ensemble_mean | 55000 | 76.47 | 71.72 | 59.23 | 12.48 |
| hopper | A | iad_lambda_1.0 | 55000 | 67.37 | 63.68 | 55.84 | 7.84 |
| hopper | A | shuffled_iad_lambda_1.0 | 55000 | 64.87 | 68.91 | 49.88 | 19.03 |
| hopper | A | standard_iql | 70000 | 60.37 | 52.90 | 34.97 | 17.94 |
| hopper | B | ensemble_mean | 85000 | 70.03 | 63.45 | 33.08 | 30.37 |
| hopper | B | iad_lambda_1.0 | 50000 | 54.93 | 61.58 | 34.98 | 26.60 |
| hopper | B | shuffled_iad_lambda_1.0 | 75000 | 32.96 | 33.63 | 25.06 | 8.58 |
| hopper | B | standard_iql | 75000 | 37.61 | 39.79 | 23.40 | 16.38 |
| walker2d | A | ensemble_mean | 60000 | 89.51 | 89.69 | 84.21 | 5.48 |
| walker2d | A | iad_lambda_1.0 | 20000 | 87.78 | 85.57 | 66.70 | 18.87 |
| walker2d | A | shuffled_iad_lambda_1.0 | 95000 | 89.34 | 87.10 | 86.19 | 0.91 |
| walker2d | A | standard_iql | 100000 | 86.59 | 80.85 | 80.85 | 0.00 |
| walker2d | B | ensemble_mean | 95000 | 87.06 | 83.87 | 83.71 | 0.16 |
| walker2d | B | iad_lambda_1.0 | 100000 | 90.54 | 89.51 | 89.51 | 0.00 |
| walker2d | B | shuffled_iad_lambda_1.0 | 100000 | 89.58 | 86.95 | 86.95 | 0.00 |
| walker2d | B | standard_iql | 100000 | 86.93 | 82.59 | 82.59 | 0.00 |

*24 runs total (3 envs × 4 methods × 2 replicates). Values rounded to 2 decimals; full precision in CSV.*

---

## Table B: Mean ± std by env/method (N=2 replicates)

| Env | Method | Best 50-ep Mean ± Std | Final 50-ep Mean ± Std | Avg Best-Final Gap |
|-----|--------|----------------------|------------------------|-------------------:|
| halfcheetah | standard_iql | 47.05 ± 0.11 | 46.70 ± 0.03 | 0.35 |
| halfcheetah | ensemble_mean | 48.84 ± 0.14 | 48.65 ± 0.40 | 0.19 |
| halfcheetah | iad_lambda_1.0 | 48.89 ± 0.05 | 48.89 ± 0.05 | 0.00 |
| halfcheetah | shuffled_iad_lambda_1.0 | 49.08 ± 0.31 | 48.98 ± 0.44 | 0.10 |
| hopper | standard_iql | 46.35 ± 9.27 | 29.19 ± 8.18 | 17.16 |
| hopper | ensemble_mean | 67.58 ± 5.84 | 46.16 ± 18.50 | 21.43 |
| hopper | iad_lambda_1.0 | 62.63 ± 1.48 | 45.41 ± 14.75 | 17.22 |
| hopper | shuffled_iad_lambda_1.0 | 51.27 ± 24.95 | 37.47 ± 17.55 | 13.80 |
| walker2d | standard_iql | 81.72 ± 1.23 | 81.72 ± 1.23 | 0.00 |
| walker2d | ensemble_mean | 86.78 ± 4.11 | 83.96 ± 0.35 | 2.82 |
| walker2d | iad_lambda_1.0 | 87.54 ± 2.79 | 78.10 ± 16.13 | 9.44 |
| walker2d | shuffled_iad_lambda_1.0 | 87.02 ± 0.10 | 86.57 ± 0.54 | 0.46 |

**Note:** Std is across **2 actor replicates**, not 3 value seeds.

---

## Table C: Method comparison (mean over Rep A/B)

| Env | Comparison | Best 50-ep Δ | Final 50-ep Δ |
|-----|------------|-------------:|--------------:|
| halfcheetah | ensemble_mean − standard_iql | **+1.79** | **+1.96** |
| halfcheetah | iad_lambda_1.0 − ensemble_mean | +0.05 | +0.24 |
| halfcheetah | iad_lambda_1.0 − shuffled_iad_lambda_1.0 | −0.19 | −0.09 |
| hopper | ensemble_mean − standard_iql | **+21.24** | **+16.97** |
| hopper | iad_lambda_1.0 − ensemble_mean | −4.95 | −0.74 |
| hopper | iad_lambda_1.0 − shuffled_iad_lambda_1.0 | +11.36 | +7.94 |
| walker2d | ensemble_mean − standard_iql | **+5.06** | **+2.24** |
| walker2d | iad_lambda_1.0 − ensemble_mean | +0.76 | −5.86 |
| walker2d | iad_lambda_1.0 − shuffled_iad_lambda_1.0 | +0.51 | −8.47 |

**Primary claim (our side):** All `ensemble_mean − standard_iql` deltas are **positive** on both Best and Final 50-ep.

---

## Table D: Overtraining / stability by environment

| Env | Avg best–final gap (all runs) | Largest gap | Run with largest gap | Best-checkpoint necessary? |
|-----|------------------------------:|------------:|----------------------|----------------------------|
| halfcheetah | 0.16 | 0.45 | standard_iql Rep A | Low priority (gaps < 0.5) |
| hopper | 17.40 | **30.37** | ensemble_mean Rep B | **Yes — critical** |
| walker2d | 3.18 | **18.87** | iad_lambda_1.0 Rep A | **Yes — moderate** |

**Interpretation:**

- **Hopper:** Severe late-stage score collapse for most methods; reporting Final-only would understate peak and overstate collapse. Best-checkpoint + Final dual reporting mandatory.
- **Walker2d:** `ensemble_mean` gaps modest (5.5 Rep A, 0.2 Rep B); IAD Rep A shows large gap (18.9) → IAD unstable as main method.
- **HalfCheetah:** Actor extraction stable at 100k; best-checkpoint still good practice but gaps negligible.

---

## Raw data paths

- Per-run summaries: `best_checkpoint_protocol/{halfcheetah,hopper,walker2d}_medium_v2/results/*_summary.json`
- Aggregated CSV: `best_checkpoint_protocol/results/best_protocol_all_runs.csv`
