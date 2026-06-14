#!/usr/bin/env python3
"""Triple value system statistics (A0, A1, A2, mean, std)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/code/CORL")
sys.path.insert(0, "/code/analysis/iad_iql/scripts")

from iad_common import (
    build_env_and_buffer,
    compute_online_advantages,
    get_batch_from_indices,
    load_norm_stats,
    load_value_bundle,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True)
    p.add_argument("--checkpoint-a", required=True)
    p.add_argument("--checkpoint-b", required=True)
    p.add_argument("--checkpoint-c", required=True)
    p.add_argument("--norm-stats", required=True)
    p.add_argument("--batch-indices", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--n-transitions", type=int, default=100000)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    state_mean, state_std = load_norm_stats(args.norm_stats)
    _, _, replay_buffer, state_dim, action_dim = build_env_and_buffer(
        args.env, args.device, state_mean, state_std
    )
    b0 = load_value_bundle(args.checkpoint_a, state_dim, action_dim, args.device, 0)
    b1 = load_value_bundle(args.checkpoint_b, state_dim, action_dim, args.device, 1)
    b2 = load_value_bundle(args.checkpoint_c, state_dim, action_dim, args.device, 2)
    batch_indices = np.load(args.batch_indices)
    n_batches = min(args.n_transitions // batch_indices.shape[1], batch_indices.shape[0])

    all_a0, all_a1, all_a2, all_mean, all_std = [], [], [], [], []
    with torch.no_grad():
        for i in range(n_batches):
            idx = batch_indices[i]
            batch = get_batch_from_indices(replay_buffer, idx)
            states, actions = batch[0].to(args.device), batch[1].to(args.device)
            a0 = compute_online_advantages(b0, states, actions)
            a1 = compute_online_advantages(b1, states, actions)
            a2 = compute_online_advantages(b2, states, actions)
            mean_3q = (a0 + a1 + a2) / 3.0
            std_3q = torch.stack([a0, a1, a2], dim=-1).std(dim=-1)
            all_a0.append(a0.cpu().numpy())
            all_a1.append(a1.cpu().numpy())
            all_a2.append(a2.cpu().numpy())
            all_mean.append(mean_3q.cpu().numpy())
            all_std.append(std_3q.cpu().numpy())

    a0 = np.concatenate(all_a0)
    a1 = np.concatenate(all_a1)
    a2 = np.concatenate(all_a2)
    mean_3q = np.concatenate(all_mean)
    std_3q = np.concatenate(all_std)

    result = {
        "env": args.env,
        "n_transitions": int(len(a0)),
        "forward_pass_ok": bool(
            np.isfinite(a0).all() and np.isfinite(a1).all()
            and np.isfinite(a2).all() and np.isfinite(mean_3q).all()
            and np.isfinite(std_3q).all()
        ),
        "A0": {"mean": float(a0.mean()), "std": float(a0.std())},
        "A1": {"mean": float(a1.mean()), "std": float(a1.std())},
        "A2": {"mean": float(a2.mean()), "std": float(a2.std())},
        "A_mean_3q": {"mean": float(mean_3q.mean()), "std": float(mean_3q.std())},
        "A_std_3q": {
            "mean": float(std_3q.mean()),
            "median": float(np.median(std_3q)),
            "p90": float(np.quantile(std_3q, 0.9)),
            "p99": float(np.quantile(std_3q, 0.99)),
            "max": float(std_3q.max()),
        },
        "note": "A_std_3q is descriptive only; not used for training penalty.",
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["forward_pass_ok"] else 1)


if __name__ == "__main__":
    main()
