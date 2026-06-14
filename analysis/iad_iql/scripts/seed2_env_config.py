"""Environment paths for seed2 supplement experiment."""
from pathlib import Path

ROOT = Path("/code/analysis/iad_iql")
SUPP = ROOT / "seed2_supplement"

ENVS = {
    "halfcheetah": {
        "env_name": "halfcheetah-medium-v2",
        "project_dir": str(ROOT),
        "ckpt_dir": str(ROOT / "checkpoints"),
        "log_dir": str(ROOT / "logs"),
        "norm_stats": str(ROOT / "checkpoints" / "norm_stats.npz"),
        "batch42": str(ROOT / "best_checkpoint_protocol" / "halfcheetah_medium_v2" / "checkpoints" / "actor_training_batch_indices_seed42.npy"),
        "batch43": str(ROOT / "best_checkpoint_protocol" / "halfcheetah_medium_v2" / "checkpoints" / "actor_training_batch_indices_seed43.npy"),
        "init12345": str(ROOT / "best_checkpoint_protocol" / "halfcheetah_medium_v2" / "checkpoints" / "shared_actor_init_seed12345.pt"),
        "init54321": str(ROOT / "best_checkpoint_protocol" / "halfcheetah_medium_v2" / "checkpoints" / "shared_actor_init_seed54321.pt"),
        "run_prefix": "hc",
    },
    "hopper": {
        "env_name": "hopper-medium-v2",
        "project_dir": str(ROOT / "hopper_medium_v2"),
        "ckpt_dir": str(ROOT / "hopper_medium_v2" / "checkpoints"),
        "log_dir": str(ROOT / "hopper_medium_v2" / "logs"),
        "norm_stats": str(ROOT / "hopper_medium_v2" / "checkpoints" / "norm_stats.npz"),
        "batch42": str(ROOT / "best_checkpoint_protocol" / "hopper_medium_v2" / "checkpoints" / "actor_training_batch_indices_seed42.npy"),
        "batch43": str(ROOT / "best_checkpoint_protocol" / "hopper_medium_v2" / "checkpoints" / "actor_training_batch_indices_seed43.npy"),
        "init12345": str(ROOT / "best_checkpoint_protocol" / "hopper_medium_v2" / "checkpoints" / "shared_actor_init_seed12345.pt"),
        "init54321": str(ROOT / "best_checkpoint_protocol" / "hopper_medium_v2" / "checkpoints" / "shared_actor_init_seed54321.pt"),
        "run_prefix": "hopper",
    },
    "walker2d": {
        "env_name": "walker2d-medium-v2",
        "project_dir": str(ROOT / "walker2d_medium_v2"),
        "ckpt_dir": str(ROOT / "walker2d_medium_v2" / "checkpoints"),
        "log_dir": str(ROOT / "walker2d_medium_v2" / "logs"),
        "norm_stats": str(ROOT / "walker2d_medium_v2" / "checkpoints" / "norm_stats.npz"),
        "batch42": str(ROOT / "best_checkpoint_protocol" / "walker2d_medium_v2" / "checkpoints" / "actor_training_batch_indices_seed42.npy"),
        "batch43": str(ROOT / "best_checkpoint_protocol" / "walker2d_medium_v2" / "checkpoints" / "actor_training_batch_indices_seed43.npy"),
        "init12345": str(ROOT / "best_checkpoint_protocol" / "walker2d_medium_v2" / "checkpoints" / "shared_actor_init_seed12345.pt"),
        "init54321": str(ROOT / "best_checkpoint_protocol" / "walker2d_medium_v2" / "checkpoints" / "shared_actor_init_seed54321.pt"),
        "run_prefix": "walker",
    },
}


def ckpt_path(env_key: str, seed: int, step: int = 999999) -> str:
    cfg = ENVS[env_key]
    return str(Path(cfg["ckpt_dir"]) / f"value_seed{seed}" / f"checkpoint_{step:06d}.pt")


def actor_output_dirs(env_key: str):
    base = SUPP / env_key
    return {
        "checkpoints": str(base / "checkpoints"),
        "logs": str(base / "logs"),
        "results": str(base / "results"),
    }
