# Offline RL Training Results Summary

**Date**: 2026-06-13 09:23
**Total experiments**: 45 (5 algorithms × 3 datasets × 3 seeds)
**Training steps per experiment**: 1,000,000
**Datasets**: halfcheetah-medium-v2, hopper-medium-v2, walker2d-medium-v2
**Algorithms**: CQL, IQL, DT, UA_CQL, UGH_CQL

---

## Overall Progress

| Metric | Value |
|--------|-------|
| Total experiments | 45 |
| Completed (1M steps) | 32 |
| Running | 13 |
| Average progress | 93.6% |
| Min step | 610k |
| Max step | 1,000k |

---

## Results by Dataset

### halfcheetah-medium-v2

| Algorithm | Seed | Progress | Last D4RL | Best D4RL | Best@Step |
|-----------|------|----------|-----------|-----------|-----------|
| CQL | 0 | 1,000k (100%) | 48.91 | 49.30 | 655k |
| CQL | 1 | 1,000k (100%) | 48.99 | 49.34 | 505k |
| CQL | 2 | 720k (72%) | — | — | — |
| IQL | 0 | 870k (87%) | 47.74 | 48.67 | 710k |
| IQL | 1 | 730k (73%) | 47.72 | 48.47 | 715k |
| IQL | 2 | 1,000k (100%) | 48.56 | 48.61 | 575k |
| DT | 0 | 690k (69%) | 40.83 | 44.36 | 200k |
| DT | 1 | 725k (72%) | 42.49 | 45.27 | 90k |
| DT | 2 | 610k (61%) | 41.86 | 45.79 | 380k |
| UA_CQL | 0 | 1,000k (100%) | 42.38 | 44.97 | 20k |
| UA_CQL | 1 | 1,000k (100%) | 42.61 | 45.83 | 20k |
| UA_CQL | 2 | 1,000k (100%) | 42.94 | 46.23 | 15k |
| UGH_CQL | 0 | 1,000k (100%) | 42.23 | 44.99 | 35k |
| UGH_CQL | 1 | 1,000k (100%) | 43.09 | 44.69 | 40k |
| UGH_CQL | 2 | 1,000k (100%) | 42.73 | 44.45 | 20k |

**Algorithm averages (best D4RL)**:

| Algorithm | Mean Best | Mean Last |
|-----------|-----------|-----------|
| CQL | 49.32 | 48.95 |
| IQL | 48.58 | 48.01 |
| DT | 45.14 | 41.73 |
| UA_CQL | 45.68 | 42.64 |
| UGH_CQL | 44.71 | 42.68 |

---

### hopper-medium-v2

| Algorithm | Seed | Progress | Last D4RL | Best D4RL | Best@Step |
|-----------|------|----------|-----------|-----------|-----------|
| CQL | 0 | 990k (99%) | — | — | — |
| CQL | 1 | 1,000k (100%) | 54.71 | 69.20 | 390k |
| CQL | 2 | 1,000k (100%) | 49.47 | 70.40 | 410k |
| IQL | 0 | 1,000k (100%) | 53.12 | 72.89 | 530k |
| IQL | 1 | 970k (97%) | 51.00 | 68.03 | 200k |
| IQL | 2 | 625k (62%) | 53.99 | 62.58 | 210k |
| DT | 0 | 1,000k (100%) | 46.82 | 99.63 | 465k |
| DT | 1 | 1,000k (100%) | 54.48 | 79.82 | 740k |
| DT | 2 | 1,000k (100%) | 63.87 | 97.51 | 900k |
| UA_CQL | 0 | 1,000k (100%) | 40.41 | 66.01 | 5k |
| UA_CQL | 1 | 1,000k (100%) | 43.89 | 54.85 | 5k |
| UA_CQL | 2 | 1,000k (100%) | 41.13 | 67.85 | 5k |
| UGH_CQL | 0 | 1,000k (100%) | 30.83 | 56.34 | 30k |
| UGH_CQL | 1 | 1,000k (100%) | 46.38 | 54.90 | 175k |
| UGH_CQL | 2 | 1,000k (100%) | 32.50 | 60.83 | 90k |

**Algorithm averages (best D4RL)**:

| Algorithm | Mean Best | Mean Last |
|-----------|-----------|-----------|
| CQL | 69.80 | 52.09 |
| IQL | 67.83 | 52.70 |
| DT | 92.32 | 55.06 |
| UA_CQL | 62.90 | 41.81 |
| UGH_CQL | 57.35 | 36.57 |

---

### walker2d-medium-v2

| Algorithm | Seed | Progress | Last D4RL | Best D4RL | Best@Step |
|-----------|------|----------|-----------|-----------|-----------|
| CQL | 0 | 725k (72%) | — | — | — |
| CQL | 1 | 890k (89%) | — | — | — |
| CQL | 2 | 1,000k (100%) | 82.95 | 84.98 | 625k |
| IQL | 0 | 1,000k (100%) | 86.05 | 88.18 | 640k |
| IQL | 1 | 1,000k (100%) | 81.04 | 87.19 | 750k |
| IQL | 2 | 820k (82%) | 86.65 | 87.50 | 650k |
| DT | 0 | 775k (78%) | 78.51 | 87.35 | 690k |
| DT | 1 | 1,000k (100%) | 71.53 | 85.14 | 5k |
| DT | 2 | 1,000k (100%) | 78.55 | 84.62 | 330k |
| UA_CQL | 0 | 1,000k (100%) | 20.28 | 80.95 | 35k |
| UA_CQL | 1 | 1,000k (100%) | 46.25 | 80.83 | 75k |
| UA_CQL | 2 | 1,000k (100%) | 33.12 | 78.03 | 20k |
| UGH_CQL | 0 | 1,000k (100%) | 16.36 | 79.48 | 60k |
| UGH_CQL | 1 | 1,000k (100%) | 26.53 | 77.68 | 60k |
| UGH_CQL | 2 | 1,000k (100%) | 46.91 | 79.40 | 30k |

**Algorithm averages (best D4RL)**:

| Algorithm | Mean Best | Mean Last |
|-----------|-----------|-----------|
| CQL | 84.98 | 82.95 |
| IQL | 87.62 | 84.58 |
| DT | 85.70 | 76.20 |
| UA_CQL | 79.94 | 33.21 |
| UGH_CQL | 78.85 | 29.93 |

---

## Cross-Dataset Algorithm Ranking (by mean Best D4RL)

| Algorithm | halfcheetah | hopper | walker2d | **Average** |
|-----------|-------------|--------|----------|-------------|
| IQL | 48.58 | 67.83 | 87.62 | **68.01** |
| CQL | 49.32 | 69.80 | 84.98 | **68.03** |
| DT | 45.14 | 92.32 | 85.70 | **74.39** |
| UA_CQL | 45.68 | 62.90 | 79.94 | **62.84** |
| UGH_CQL | 44.71 | 57.35 | 78.85 | **60.30** |

---

## Experimental Configuration

- **Hardware**: 4× NVIDIA A800-SXM4-80GB
- **Concurrency**: 8 tasks per GPU (subprocess.Popen)
- **Environment**: Gymnasium MuJoCo v4
- **State normalization**: mean/std from dataset
- **Eval frequency**: every 5,000 steps, 10 episodes
- **D4RL score**: normalized 0–100+ scale (0=random, 100=expert)
- **DT target returns**: halfcheetah=6000, hopper=3600, walker2d=5000
- **DT reward_scale**: 0.001
- **DT episode_len**: inferred from dataset (halfcheetah=1020, hopper=1020, walker2d=1020)
