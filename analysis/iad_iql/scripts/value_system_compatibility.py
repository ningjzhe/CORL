#!/usr/bin/env python3
"""Full compatibility check between two IQL value systems for IAD-IQL."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, "/code/CORL")
sys.path.insert(0, "/code/analysis/iad_iql/scripts")

from iad_common import (  # noqa: E402
    build_env_and_buffer,
    compute_online_advantages,
    get_batch_from_indices,
    load_norm_stats,
    load_value_bundle,
)


def inspect_checkpoint(path: str, seed: int, device: str) -> dict:
    sd = torch.load(path, map_location="cpu")
    info = {
        "path": path,
        "seed": seed,
        "total_it": int(sd.get("total_it", -1)),
        "keys": sorted(sd.keys()),
        "has_qf": "qf" in sd,
        "has_vf": "vf" in sd,
        "has_actor": "actor" in sd,
        "nan_inf": False,
    }
    for name in ("qf", "vf", "actor"):
        if name not in sd:
            continue
        for k, v in sd[name].items():
            t = v if isinstance(v, torch.Tensor) else torch.tensor(v)
            if torch.isnan(t).any() or torch.isinf(t).any():
                info["nan_inf"] = True
                info[f"{name}.{k}_bad"] = True

    cfg_path = Path(path).parent / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        info["config"] = {
            "env": cfg.get("env"),
            "beta": cfg.get("beta"),
            "iql_tau": cfg.get("iql_tau"),
            "normalize": cfg.get("normalize"),
            "normalize_reward": cfg.get("normalize_reward"),
            "batch_size": cfg.get("batch_size"),
        }
    return info


def find_final_eval_score(log_path: Path) -> float:
    if not log_path.exists():
        return float("nan")
    scores = []
    for line in log_path.read_text().splitlines():
        if "D4RL score:" in line:
            try:
                scores.append(float(line.split("D4RL score:")[-1].strip()))
            except ValueError:
                pass
    return float(scores[-1]) if scores else float("nan")


def main():
    default_ckpt_dir = "/code/analysis/iad_iql/checkpoints"
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="halfcheetah-medium-v2")
    parser.add_argument(
        "--checkpoint-a",
        default=f"{default_ckpt_dir}/value_seed0/checkpoint_999999.pt",
    )
    parser.add_argument(
        "--checkpoint-b",
        default=f"{default_ckpt_dir}/value_seed1/checkpoint_999999.pt",
    )
    parser.add_argument(
        "--norm-stats",
        default="/code/analysis/iad_iql/checkpoints/norm_stats.npz",
    )
    parser.add_argument(
        "--batch-indices",
        default="/code/analysis/iad_iql/checkpoints/actor_training_batch_indices.npy",
    )
    parser.add_argument(
        "--value-log-dir",
        default="/code/analysis/iad_iql/logs",
        help="Directory containing value_seed{N}.log for final eval scores",
    )
    parser.add_argument("--n-transitions", type=int, default=100000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output",
        default="/code/analysis/iad_iql/results/value_system_compatibility.json",
    )
    args = parser.parse_args()

    ckpt0 = args.checkpoint_a
    ckpt1 = args.checkpoint_b
    for p in (ckpt0, ckpt1):
        if not Path(p).exists():
            raise FileNotFoundError(f"Missing checkpoint: {p}")

    info0 = inspect_checkpoint(ckpt0, 0, args.device)
    info1 = inspect_checkpoint(ckpt1, 1, args.device)

    state_mean, state_std = load_norm_stats(args.norm_stats)
    _, _, replay_buffer, state_dim, action_dim = build_env_and_buffer(
        args.env, args.device, state_mean, state_std
    )

    bundle0 = load_value_bundle(ckpt0, state_dim, action_dim, args.device, 0)
    bundle1 = load_value_bundle(ckpt1, state_dim, action_dim, args.device, 1)

    batch_indices = np.load(args.batch_indices)
    n_batches = min(args.n_transitions // batch_indices.shape[1], batch_indices.shape[0])
    all_a1, all_a2, all_mu, all_u, all_hat = [], [], [], [], []

    with torch.no_grad():
        for i in range(n_batches):
            idx = batch_indices[i]
            batch = get_batch_from_indices(replay_buffer, idx)
            states, actions = batch[0].to(args.device), batch[1].to(args.device)
            a1 = compute_online_advantages(bundle0, states, actions)
            a2 = compute_online_advantages(bundle1, states, actions)
            mu_a = (a1 + a2) / 2.0
            u_a = (a1 - a2).abs() / 2.0
            batch_mean_ua = u_a.mean()
            hat_u_a = torch.clamp(u_a / (batch_mean_ua + 1e-6), 0.0, 5.0)
            all_a1.append(a1.cpu().numpy())
            all_a2.append(a2.cpu().numpy())
            all_mu.append(mu_a.cpu().numpy())
            all_u.append(u_a.cpu().numpy())
            all_hat.append(hat_u_a.cpu().numpy())

    a1 = np.concatenate(all_a1)
    a2 = np.concatenate(all_a2)
    mu_a = np.concatenate(all_mu)
    u_a = np.concatenate(all_u)
    hat_u = np.concatenate(all_hat)

    pearson = float(pearsonr(a1, a2)[0])
    spearman = float(spearmanr(a1, a2)[0])

    cfg0 = info0.get("config", {})
    cfg1 = info1.get("config", {})
    preprocessing_match = (
        cfg0.get("env") == cfg1.get("env")
        and cfg0.get("normalize") == cfg1.get("normalize")
        and cfg0.get("normalize_reward") == cfg1.get("normalize_reward")
        and cfg0.get("beta") == cfg1.get("beta")
        and cfg0.get("iql_tau") == cfg1.get("iql_tau")
    )

    qf_params0 = sum(p.numel() for p in bundle0.qf.parameters())
    qf_params1 = sum(p.numel() for p in bundle1.qf.parameters())
    vf_params0 = sum(p.numel() for p in bundle0.vf.parameters())
    vf_params1 = sum(p.numel() for p in bundle1.vf.parameters())

    structure_match = (
        state_dim == state_dim
        and action_dim == action_dim
        and qf_params0 == qf_params1
        and vf_params0 == vf_params1
    )

    stats_finite = bool(
        np.isfinite(a1).all()
        and np.isfinite(a2).all()
        and np.isfinite(mu_a).all()
        and np.isfinite(u_a).all()
        and np.isfinite(hat_u).all()
    )

    compatible = (
        info0["has_qf"] and info0["has_vf"]
        and info1["has_qf"] and info1["has_vf"]
        and not info0.get("nan_inf") and not info1.get("nan_inf")
        and preprocessing_match
        and structure_match
        and stats_finite
    )

    log_dir = Path(args.value_log_dir)
    result = {
        "compatible": compatible,
        "env": args.env,
        "checkpoints": {"seed0": info0, "seed1": info1},
        "state_dim": state_dim,
        "action_dim": action_dim,
        "preprocessing_match": preprocessing_match,
        "structure_match": structure_match,
        "stats_finite": stats_finite,
        "n_transitions_analyzed": int(len(a1)),
        "pearson_A1_A2": pearson,
        "spearman_A1_A2": spearman,
        "A1": {
            "mean": float(a1.mean()),
            "std": float(a1.std()),
        },
        "A2": {
            "mean": float(a2.mean()),
            "std": float(a2.std()),
        },
        "mu_A": {
            "mean": float(mu_a.mean()),
            "std": float(mu_a.std()),
        },
        "disagreement_abs_A1_minus_A2_over_2": {
            "mean": float(u_a.mean()),
            "median": float(np.median(u_a)),
            "p90": float(np.quantile(u_a, 0.9)),
            "p99": float(np.quantile(u_a, 0.99)),
            "max": float(u_a.max()),
        },
        "normalized_disagreement": {
            "mean": float(hat_u.mean()),
            "p90": float(np.quantile(hat_u, 0.9)),
            "p99": float(np.quantile(hat_u, 0.99)),
        },
        "final_eval_d4rl_score": {
            "seed0": find_final_eval_score(log_dir / "value_seed0.log"),
            "seed1": find_final_eval_score(log_dir / "value_seed1.log"),
        },
        "note": "Low correlation is expected and is not a failure criterion.",
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    sys.exit(0 if compatible else 1)


if __name__ == "__main__":
    main()
