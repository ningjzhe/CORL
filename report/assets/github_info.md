# GitHub Repository Metadata

| Field | Value |
|-------|-------|
| **Repository** | https://github.com/ningjzhe/CORL.git |
| **Branch** | main |
| **Commit** | dd0cb9c (report assets); metadata commit updated below after push |

## Report assets paths (in repo)

| Asset | Path |
|-------|------|
| Main report assets index | `report/assets/README_algorithm_owner_assets.md` |
| Main result tables | `report/assets/tables/` |
| Main figures | `report/assets/figures/` |
| Methods & protocol docs | `report/assets/docs/` |
| Environment info | `report/assets/experiment_environment_our_side.md` |
| Model & hyperparameters | `report/assets/model_and_hyperparameters_our_side.md` |

## Algorithm-owner code (in repo)

| Component | Path |
|-----------|------|
| IAD-IQL scripts | `analysis/iad_iql/scripts/` |
| Best-protocol pipeline (reference) | `analysis/iad_iql/best_checkpoint_protocol/run_best_protocol_pipeline.sh` |
| CORL IQL | `algorithms/offline/iql.py` |

## Local-only (not in git)

- `*.pt` checkpoints under `checkpoints/` or `**/checkpoints/`
- D4RL `*.hdf5` datasets
- `wandb/`, full training `logs/`
- Large `*.tar.gz` archives

## Remote

```
origin  git@github.com:ningjzhe/CORL.git
```

HTTPS equivalent: https://github.com/ningjzhe/CORL.git
