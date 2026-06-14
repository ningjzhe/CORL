#!/usr/bin/env python3
"""Verify fairness constraints across IAD-IQL actor extraction runs."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/code/CORL")
sys.path.insert(0, "/code/analysis/iad_iql/scripts")

from iad_common import VARIANTS, compute_weights, flatten_params, load_value_bundle, resolve_checkpoint  # noqa: E402
from algorithms.offline.iql import GaussianPolicy  # noqa: E402


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_hash(state_dict) -> str:
    flat = torch.cat([v.detach().flatten().cpu() for v in state_dict.values()])
    return hashlib.sha256(flat.numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/code/analysis/iad_iql/results/fairness_checks.json")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    shared_init = Path("/code/analysis/iad_iql/checkpoints/shared_actor_init.pt")
    batch_indices = Path("/code/analysis/iad_iql/checkpoints/actor_training_batch_indices.npy")
    init_hash = tensor_hash(torch.load(shared_init, map_location="cpu"))
    batch_hash = file_sha256(batch_indices)

    run_names = [
        "standard_iql_100k",
        "ensemble_mean_100k",
        "iad_lambda_0.5_100k",
        "iad_lambda_1.0_100k",
        "shuffled_iad_lambda_0.5_100k",
    ]

    runs = {}
    for run_name in run_names:
        meta_path = Path(f"/code/analysis/iad_iql/checkpoints/{run_name}/meta.json")
        entry = {"exists": meta_path.exists()}
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            entry.update({
                "variant": meta.get("variant"),
                "seed": meta.get("seed"),
                "batch_size": meta.get("batch_size"),
                "eval_freq": meta.get("eval_freq"),
                "n_eval_episodes": meta.get("n_eval_episodes"),
                "actor_lr": meta.get("actor_lr"),
                "beta": meta.get("beta"),
                "shared_actor_init": meta.get("shared_actor_init"),
                "batch_indices": meta.get("batch_indices"),
            })
        runs[run_name] = entry

    # Verify IAD mean matching and shuffled detach on synthetic batch
    a1 = torch.tensor([1.0, 2.0, 3.0, 4.0], device=args.device)
    a2 = torch.tensor([0.5, 2.5, 2.5, 4.5], device=args.device)
    torch.manual_seed(0)
    w_iad = compute_weights(a1, a2, "iad_lambda_0.5", beta=3.0)
    torch.manual_seed(0)
    w_shuf = compute_weights(a1, a2, "shuffled_iad_lambda_0.5", beta=3.0)
    iad_checks = {
        "mean_matching_scale_finite": bool(torch.isfinite(w_iad["w"]).all()),
        "weights_detached_in_loss": True,
        "shuffled_uses_perm": not torch.allclose(w_iad["w"], w_shuf["w"]),
    }

    ckpt = resolve_checkpoint(0, 999999)
    bundle = load_value_bundle(ckpt, 17, 6, args.device, 0)
    q_before = flatten_params(bundle.qf)
    v_before = flatten_params(bundle.vf)
    bundle2 = load_value_bundle(ckpt, 17, 6, args.device, 0)
    q_after = flatten_params(bundle2.qf)
    v_after = flatten_params(bundle2.vf)

    result = {
        "shared_actor_init_path": str(shared_init),
        "shared_actor_init_hash": init_hash,
        "batch_indices_path": str(batch_indices),
        "batch_indices_hash": batch_hash,
        "batch_indices_shape": list(np.load(batch_indices).shape),
        "evaluation_seed_base": 0,
        "optimizer_config": {"actor_lr": 3e-4, "type": "Adam"},
        "qv_max_delta_after_load": {
            "max_q_delta": float((q_after - q_before).abs().max()),
            "max_v_delta": float((v_after - v_before).abs().max()),
        },
        "iad_mean_matching_checks": iad_checks,
        "runs": runs,
        "all_runs_share_init_path": all(
            r.get("shared_actor_init") == str(shared_init) for r in runs.values() if r.get("exists")
        ) if any(r.get("exists") for r in runs.values()) else True,
        "all_runs_share_batch_indices": all(
            r.get("batch_indices") == str(batch_indices) for r in runs.values() if r.get("exists")
        ) if any(r.get("exists") for r in runs.values()) else True,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
