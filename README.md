# CORL — Offline RL Course Project

Fork of [CORL](https://github.com/tinkoff-ai/CORL) extended with **Independent Value-System Ensemble Policy Extraction** experiments on D4RL Gym MuJoCo **medium-v2** tasks.

**Repository:** https://github.com/ningjzhe/CORL.git

---

## Project overview

This repository supports a course project on offline reinforcement learning. We reproduce standard baselines (**CQL**, **IQL**, **DT**) and explore uncertainty-aware variants (**UA_CQL**, **UGH_CQL**) on the experiment-owner side, while the algorithm-owner contribution focuses on **frozen Q/V actor extraction** with two independent IQL value systems and an **ensemble_mean** weighting scheme.

### Our method: Independent Value-System Ensemble Policy Extraction

Instead of extracting a policy from a single IQL value system, we:

1. Train **two independent IQL value systems** (seed0, seed1) per environment.
2. **Freeze** Q and V during actor-only extraction.
3. Compute online advantages \(A_0, A_1\) and ensemble their mean: \(\mu_A = (A_0 + A_1)/2\).
4. Apply standard IQL exponential weighting: \(w = \exp(\beta \cdot \mu_A)\), clipped at `EXP_ADV_MAX`.

**Ablation:** IAD (disagreement-aware weighting, λ=1.0) and shuffled-IAD controls are included but **not** promoted as the main method.

---

## Main conclusions (algorithm owner, best-checkpoint protocol)

| Finding | Detail |
|---------|--------|
| **ensemble_mean vs standard_iql** | **Positive** Best 50-ep and Final 50-ep deltas on **HalfCheetah, Hopper, and Walker2d** |
| **IAD** | Unstable across envs/replicates → **ablation only** |
| **Overtraining** | Hopper and Walker2d show large best–final gaps → **best-checkpoint protocol required** (save `best_actor.pt` + 50-ep re-eval) |

Detailed tables: [`report/assets/tables/`](report/assets/tables/) · Figures: [`report/assets/figures/`](report/assets/figures/) · Docs: [`report/assets/docs/`](report/assets/docs/)

**Not in main results:** seed2 value systems, 3Q ensemble (future work).

---

## Baselines and explored methods

| Category | Methods | Owner |
|----------|---------|-------|
| Baselines | CQL, IQL, DT | Experiment owner |
| Explored | UA_CQL, UGH_CQL | Experiment owner |
| **Our contribution** | `standard_iql`, `ensemble_mean`, `iad_lambda_1.0`, `shuffled_iad_lambda_1.0` | Algorithm owner |

Baseline numbers for the final report are **pending** from the experiment owner (see `report/assets/docs/` and baseline placeholder in writing materials).

---

## Environment & dependencies

| Component | Version (reference run) |
|-----------|-------------------------|
| Python | 3.8 |
| PyTorch | 1.11.0+cu113 |
| CUDA | 11.3 |
| GPU | 2× NVIDIA A800-SXM4-80GB |
| Gym | 0.23.0 |
| D4RL | 1.1 |
| MuJoCo | 3.2.3 |
| Conda env | `iql` |

Full stack: [`report/assets/experiment_environment_our_side.md`](report/assets/experiment_environment_our_side.md)

Install CORL dependencies:

```bash
pip install -r requirements/requirements_dev.txt
```

D4RL datasets: download to `/root/.d4rl/datasets/` (not committed; see `.gitignore`).

---

## Code layout

| Path | Description |
|------|-------------|
| `algorithms/offline/iql.py` | CORL IQL implementation (Q, V, actor training) |
| `algorithms/offline/cql.py` | CQL baseline |
| `algorithms/offline/dt.py` | Decision Transformer baseline |
| `analysis/iad_iql/scripts/iad_common.py` | Shared frozen-actor utilities, ensemble/IAD weights |
| `analysis/iad_iql/scripts/train_value_system.py` | IQL value-system training wrapper |
| `analysis/iad_iql/scripts/train_frozen_iad_actor_best.py` | **Best-checkpoint** actor extraction |
| `analysis/iad_iql/scripts/summarize_best_checkpoint_protocol.py` | Aggregate results → CSV/JSON |
| `analysis/iad_iql/best_checkpoint_protocol/run_best_protocol_pipeline.sh` | 24-run orchestration (reference) |

Hyperparameters & network architecture: [`report/assets/model_and_hyperparameters_our_side.md`](report/assets/model_and_hyperparameters_our_side.md)

---

## Report materials (submitted assets)

| Asset | Path |
|-------|------|
| **Index** | [`report/assets/README_algorithm_owner_assets.md`](report/assets/README_algorithm_owner_assets.md) |
| **Tables** | `report/assets/tables/best_protocol_*.csv`, `best_protocol_summary.json` |
| **Figures** | `report/assets/figures/*.png` |
| **Methods & protocol** | `report/assets/docs/method_description.md`, `experiment_protocol_our_side.md` |
| **Main tables (markdown)** | `report/assets/docs/main_results_tables_our_side.md` |
| **Ablation & stability** | `report/assets/docs/ablation_and_stability_analysis.md` |
| **Conclusion draft** | `report/assets/docs/final_conclusion_draft_our_side.md` |
| **GitHub metadata** | `report/assets/github_info.md` |

---

## Reproduce algorithm-owner actor runs (reference)

Value checkpoints and batch-index files are **local only** (not in git). After training value seed0/seed1:

```bash
export WANDB_MODE=offline MUJOCO_GL=osmesa D4RL_SUPPRESS_IMPORT_ERROR=1
conda activate iql

# Example: best-checkpoint ensemble_mean on Hopper Rep A
python analysis/iad_iql/scripts/train_frozen_iad_actor_best.py \
  --variant ensemble_mean --env hopper-medium-v2 \
  --run-name hopper_repA_ensemble_mean --replicate A \
  --actor-seed 12345 --batch-seed 42 \
  --max-updates 100000 --eval-freq 5000 --batch-size 256 \
  --checkpoint-a <local>/value_seed0/checkpoint_999999.pt \
  --checkpoint-b <local>/value_seed1/checkpoint_999999.pt \
  --norm-stats <local>/norm_stats.npz \
  --shared-actor-init <local>/shared_actor_init_seed12345.pt \
  --batch-indices <local>/actor_training_batch_indices_seed42.npy \
  --output-dir <local>/checkpoints --log-dir <local>/logs \
  --results-dir <local>/results --device cuda
```

Regenerate committed tables/figures from local result JSON summaries:

```bash
python analysis/iad_iql/scripts/summarize_best_checkpoint_protocol.py
python analysis/iad_iql/scripts/generate_best_protocol_figures.py
```

---

## Future work

- seed2 value systems and 3-value `ensemble_mean_3q` supplement
- pair02 / pair12 ensemble ablations
- Merge with experiment-owner baseline tables (CQL / DT / UA / UGH)

---

## Upstream CORL

Original CORL library by Tinkoff AI. See [tinkoff-ai/CORL](https://github.com/tinkoff-ai/CORL) and citation in upstream README.

## License

Apache 2.0 (see [LICENSE](LICENSE)).
