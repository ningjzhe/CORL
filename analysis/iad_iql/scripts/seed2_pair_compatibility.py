#!/usr/bin/env python3
"""Pair compatibility check for two value system checkpoints."""
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

from iad_common import (
    build_env_and_buffer,
    compute_online_advantages,
    get_batch_from_indices,
    load_norm_stats,
    load_value_bundle,
)


def inspect_checkpoint(path: str, seed: int) -> dict:
    sd = torch.load(path, map_location="cpu")
    info = {
        "path": path,
        "seed": seed,
        "has_qf": "qf" in sd,
        "has_vf": "vf" in sd,
        "nan_inf": False,
    }
    for name in ("qf", "vf"):
        if name not in sd:
            continue
        for k, v in sd[name].items():
            t = v if isinstance(v, torch.Tensor) else torch.tensor(v)
            if torch.isnan(t).any() or torch.isinf(t).any():
                info["nan_inf"] = True
    cfg_path = Path(path).parent / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        info["config"] = {
            "env": cfg.get("env"),
            "normalize": cfg.get("normalize"),
            "normalize_reward": cfg.get("normalize_reward"),
            "beta": cfg.get("beta"),
            "iql_tau": cfg.get("iql_tau"),
        }
    return info


def run_pair(env, ckpt_i, ckpt_j, seed_i, seed_j, norm_stats, batch_indices, device, n_transitions):
    info_i = inspect_checkpoint(ckpt_i, seed_i)
    info_j = inspect_checkpoint(ckpt_j, seed_j)
    state_mean, state_std = load_norm_stats(norm_stats)
    _, _, replay_buffer, state_dim, action_dim = build_env_and_buffer(env, device, state_mean, state_std)
    bundle_i = load_value_bundle(ckpt_i, state_dim, action_dim, device, seed_i)
    bundle_j = load_value_bundle(ckpt_j, state_dim, action_dim, device, seed_j)

    n_batches = min(n_transitions // batch_indices.shape[1], batch_indices.shape[0])
    all_ai, all_aj, all_u, all_hat = [], [], [], []
    with torch.no_grad():
        for b in range(n_batches):
            idx = batch_indices[b]
            batch = get_batch_from_indices(replay_buffer, idx)
            states, actions = batch[0].to(device), batch[1].to(device)
            ai = compute_online_advantages(bundle_i, states, actions)
            aj = compute_online_advantages(bundle_j, states, actions)
            u = (ai - aj).abs() / 2.0
            hat = torch.clamp(u / (u.mean() + 1e-6), 0.0, 5.0)
            all_ai.append(ai.cpu().numpy())
            all_aj.append(aj.cpu().numpy())
            all_u.append(u.cpu().numpy())
            all_hat.append(hat.cpu().numpy())

    ai = np.concatenate(all_ai)
    aj = np.concatenate(all_aj)
    u = np.concatenate(all_u)
    hat = np.concatenate(all_hat)

    cfg_i = info_i.get("config", {})
    cfg_j = info_j.get("config", {})
    preprocessing_match = (
        cfg_i.get("env") == cfg_j.get("env")
        and cfg_i.get("normalize") == cfg_j.get("normalize")
        and cfg_i.get("normalize_reward") == cfg_j.get("normalize_reward")
    )
    stats_finite = bool(np.isfinite(ai).all() and np.isfinite(aj).all())
    compatible = (
        info_i["has_qf"] and info_i["has_vf"]
        and info_j["has_qf"] and info_j["has_vf"]
        and not info_i["nan_inf"] and not info_j["nan_inf"]
        and preprocessing_match and stats_finite
    )

    return {
        "compatible": compatible,
        "pair": f"seed{seed_i}_seed{seed_j}",
        "seed_i": seed_i,
        "seed_j": seed_j,
        "pearson": float(pearsonr(ai, aj)[0]),
        "spearman": float(spearmanr(ai, aj)[0]),
        f"A{seed_i}": {"mean": float(ai.mean()), "std": float(ai.std())},
        f"A{seed_j}": {"mean": float(aj.mean()), "std": float(aj.std())},
        "disagreement": {
            "mean": float(u.mean()),
            "median": float(np.median(u)),
            "p90": float(np.quantile(u, 0.9)),
            "p99": float(np.quantile(u, 0.99)),
            "max": float(u.max()),
        },
        "normalized_disagreement": {
            "mean": float(hat.mean()),
            "p90": float(np.quantile(hat, 0.9)),
            "p99": float(np.quantile(hat, 0.99)),
        },
        "stats_finite": stats_finite,
        "preprocessing_match": preprocessing_match,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True)
    p.add_argument("--checkpoint-i", required=True)
    p.add_argument("--checkpoint-j", required=True)
    p.add_argument("--seed-i", type=int, required=True)
    p.add_argument("--seed-j", type=int, required=True)
    p.add_argument("--norm-stats", required=True)
    p.add_argument("--batch-indices", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--n-transitions", type=int, default=100000)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    batch_indices = np.load(args.batch_indices)
    result = run_pair(
        args.env, args.checkpoint_i, args.checkpoint_j,
        args.seed_i, args.seed_j, args.norm_stats, batch_indices,
        args.device, args.n_transitions,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["compatible"] else 1)


if __name__ == "__main__":
    main()
