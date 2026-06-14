# Experiment Environment (Algorithm Owner)

Captured from the machine used for IQL value training and best-checkpoint actor extraction (`conda env: iql`).

## Hardware

| Item | Value |
|------|-------|
| GPU | NVIDIA A800-SXM4-80GB × **2** |
| GPU driver | 570.124.06 |
| Typical usage | Value training: 1 process/GPU; actor extraction: 1 process/GPU (max 2 concurrent actor jobs) |

## Software stack

| Component | Version |
|-----------|---------|
| Python | 3.8.20 |
| PyTorch | 1.11.0+cu113 |
| CUDA (PyTorch build) | 11.3 |
| CUDA toolkit (`nvcc`, if queried) | 12.1 build string on system |
| NumPy | 1.23.1 |
| SciPy | 1.10.1 |
| Gym | 0.23.0 |
| MuJoCo | 3.2.3 |
| mujoco-py | 2.1.2.14 |
| pybullet | 3.2.7 |
| d4rl | 1.1 (package); no `__version__` attribute |
| pyrallis | 0.3.1 |
| wandb | 0.12.21 (offline mode during runs) |
| Conda environment name | **`iql`** |

## D4RL datasets

Local paths (validated):

| Environment | Dataset file |
|-------------|--------------|
| halfcheetah-medium-v2 | `/root/.d4rl/datasets/halfcheetah_medium-v2.hdf5` |
| hopper-medium-v2 | `/root/.d4rl/datasets/hopper_medium-v2.hdf5` |
| walker2d-medium-v2 | `/root/.d4rl/datasets/walker2d_medium-v2.hdf5` |

Rendering for eval: `MUJOCO_GL=osmesa` (headless).

## Code base

| Item | Path |
|------|------|
| CORL / IQL implementation | `/code/CORL/algorithms/offline/iql.py` |
| IAD-IQL actor extraction | `/code/analysis/iad_iql/scripts/` |
| Best-checkpoint results | `/code/analysis/iad_iql/best_checkpoint_protocol/` |

Upstream IQL reference: [gwthomas/IQL-PyTorch](https://github.com/gwthomas/IQL-PyTorch) (noted in CORL source header).

## Environments evaluated

- `halfcheetah-medium-v2` (state_dim=17, action_dim=6)
- `hopper-medium-v2` (state_dim=11, action_dim=3)
- `walker2d-medium-v2` (state_dim=17, action_dim=6)

## Notes for report/PPT

- We use **Gym 0.23** + **D4RL v1.1** (not Gymnasium) — align footnotes if comparing to experiment-owner baselines on different stacks.
- Actor extraction runs used **2× A800**; value training used **2× A800** (one seed per GPU).
- No new experiments should be started for report writing; this file documents the completed runs only.
