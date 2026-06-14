# Final Report Assets (Algorithm Owner)

Export for course report and PPT. **No new experiments.** Baseline/CQL/DT data not included.

## Contents

| Folder / file | Description |
|---------------|-------------|
| `tables/` | CSV + JSON from best-checkpoint protocol |
| `figures/` | PNG plots for main results and stability |
| `docs/` | Method, protocol, tables, ablation, conclusion drafts |
| `experiment_environment_our_side.md` | Software/hardware stack |
| `model_and_hyperparameters_our_side.md` | Networks and hyperparameters |
| `github_info.md` | Repro paths; GitHub TBD |

## Primary claim (our side)

`ensemble_mean` > `standard_iql` on Best & Final 50-ep in HalfCheetah, Hopper, Walker2d (see `tables/best_protocol_comparison.csv`).

## Source of truth

`/code/analysis/iad_iql/best_checkpoint_protocol/` (24 completed actor runs, 2 replicates × 4 methods × 3 envs).

Generated: 2026-06-14
