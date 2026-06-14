#!/usr/bin/env python3
"""Inspect IQL value checkpoints for IAD-IQL compatibility."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, "/code/CORL")
sys.path.insert(0, "/code/analysis/iad_iql/scripts")
from iad_common import load_norm_stats, load_value_bundle, build_env_and_buffer  # noqa: E402


def inspect_one(
    path: str,
    seed: int,
    env: str,
    norm_stats: str,
    device: str = "cpu",
) -> dict:
    sd = torch.load(path, map_location="cpu")
    info = {
        "path": path,
        "seed": seed,
        "total_it": int(sd.get("total_it", -1)),
        "keys": sorted(sd.keys()),
        "has_qf": "qf" in sd,
        "has_vf": "vf" in sd,
        "has_q_target": "q_target" in sd,
    }
    for name in ("qf", "vf", "actor"):
        if name not in sd:
            continue
        for k, v in sd[name].items():
            t = v if isinstance(v, torch.Tensor) else torch.tensor(v)
            if torch.isnan(t).any() or torch.isinf(t).any():
                info[f"{name}.{k}_bad"] = True
    state_mean, state_std = load_norm_stats(norm_stats)
    _, _, _, state_dim, action_dim = build_env_and_buffer(
        env, device, state_mean, state_std
    )
    bundle = load_value_bundle(path, state_dim, action_dim, device, seed)
    info["state_dim"] = state_dim
    info["action_dim"] = action_dim
    info["qf_params"] = sum(p.numel() for p in bundle.qf.parameters())
    info["vf_params"] = sum(p.numel() for p in bundle.vf.parameters())
    cfg_path = Path(path).parent / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        info["beta"] = cfg.get("beta")
        info["iql_tau"] = cfg.get("iql_tau")
        info["normalize"] = cfg.get("normalize")
        info["env"] = cfg.get("env")
    return info


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
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        default="/code/analysis/iad_iql/results/checkpoint_inspection.json",
    )
    args = parser.parse_args()

    checks = []
    for seed, ckpt in ((0, args.checkpoint_a), (1, args.checkpoint_b)):
        if Path(ckpt).exists():
            checks.append(inspect_one(ckpt, seed, args.env, args.norm_stats, args.device))
        else:
            checks.append({"seed": seed, "path": ckpt, "status": "missing"})

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(checks, f, indent=2)
    for c in checks:
        print(json.dumps(c, indent=2))


if __name__ == "__main__":
    main()
