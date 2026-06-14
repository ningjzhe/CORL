#!/usr/bin/env python3
"""Zero-disagreement unit test and smoke test runner for IAD-IQL pipeline."""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/code/CORL")
sys.path.insert(0, "/code/analysis/iad_iql/scripts")

from iad_common import (  # noqa: E402
    VARIANTS,
    actor_bc_loss,
    build_env_and_buffer,
    compute_online_advantages,
    compute_weights,
    flatten_params,
    get_batch_from_indices,
    load_norm_stats,
    load_value_bundle,
    resolve_checkpoint,
)
from algorithms.offline.iql import GaussianPolicy, set_seed  # noqa: E402


def run_zero_disagreement_test(device="cuda"):
    state_mean, state_std = load_norm_stats()
    _, _, replay_buffer, state_dim, action_dim = build_env_and_buffer(
        "halfcheetah-medium-v2", device, state_mean, state_std
    )
    ckpt = resolve_checkpoint(0, 999999)
    b0 = load_value_bundle(ckpt, state_dim, action_dim, device, 0)
    b1 = load_value_bundle(ckpt, state_dim, action_dim, device, 0)

    batch_indices = np.load("/code/analysis/iad_iql/checkpoints/actor_training_batch_indices.npy")
    idx = batch_indices[0]
    batch = get_batch_from_indices(replay_buffer, idx)
    states, actions = batch[0].to(device), batch[1].to(device)

    with torch.no_grad():
        a1 = compute_online_advantages(b0, states, actions)
        a2 = compute_online_advantages(b1, states, actions)
        max_diff = float((a1 - a2).abs().max())

    losses = {}
    weights_by_variant = {}
    for variant in VARIANTS:
        with torch.no_grad():
            wstats = compute_weights(a1, a2, variant, beta=3.0)
            w = wstats["w"]
        actor = GaussianPolicy(state_dim, action_dim, 1.0).to(device)
        actor.load_state_dict(
            torch.load("/code/analysis/iad_iql/checkpoints/shared_actor_init.pt", map_location=device)
        )
        loss = actor_bc_loss(actor, states, actions, w)
        losses[variant] = float(loss.item())
        weights_by_variant[variant] = w

    w_std = weights_by_variant["standard_iql"]
    w_em = weights_by_variant["ensemble_mean"]
    w_iad05 = weights_by_variant["iad_lambda_0.5"]
    w_iad10 = weights_by_variant["iad_lambda_1.0"]
    w_shuf = weights_by_variant["shuffled_iad_lambda_0.5"]

    results = {
        "max_abs_A1_minus_A2": max_diff,
        "losses": losses,
        "max_weight_diff_standard_vs_ensemble": float((w_std - w_em).abs().max()),
        "max_weight_diff_ensemble_vs_iad05": float((w_em - w_iad05).abs().max()),
        "max_weight_diff_ensemble_vs_iad10": float((w_em - w_iad10).abs().max()),
        "max_weight_diff_ensemble_vs_shuffled": float((w_em - w_shuf).abs().max()),
        "max_loss_diff": max(losses.values()) - min(losses.values()),
    }
    return results


def check_frozen_qv(device="cuda"):
    state_mean, state_std = load_norm_stats()
    _, _, replay_buffer, state_dim, action_dim = build_env_and_buffer(
        "halfcheetah-medium-v2", device, state_mean, state_std
    )
    ckpt = resolve_checkpoint(0, 999999)
    bundle = load_value_bundle(ckpt, state_dim, action_dim, device, 0)
    q_before = flatten_params(bundle.qf)
    v_before = flatten_params(bundle.vf)

    actor = GaussianPolicy(state_dim, action_dim, 1.0).to(device)
    actor.load_state_dict(
        torch.load("/code/analysis/iad_iql/checkpoints/shared_actor_init.pt", map_location=device)
    )
    opt = torch.optim.Adam(actor.parameters(), lr=3e-4)
    batch_indices = np.load("/code/analysis/iad_iql/checkpoints/actor_training_batch_indices.npy")
    idx = batch_indices[0]
    batch = get_batch_from_indices(replay_buffer, idx)
    states, actions = batch[0].to(device), batch[1].to(device)
    with torch.no_grad():
        a1 = compute_online_advantages(bundle, states, actions)
        w = compute_weights(a1, a1, "standard_iql")["w"]
    loss = actor_bc_loss(actor, states, actions, w)
    opt.zero_grad()
    loss.backward()
    opt.step()
    q_after = flatten_params(bundle.qf)
    v_after = flatten_params(bundle.vf)
    return {
        "q_unchanged": bool(torch.allclose(q_before, q_after)),
        "v_unchanged": bool(torch.allclose(v_before, v_after)),
        "max_q_delta": float((q_after - q_before).abs().max()),
        "max_v_delta": float((v_after - v_before).abs().max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-smoke", action="store_true")
    args = parser.parse_args()

    print("=== Zero Disagreement Unit Test ===")
    zd = run_zero_disagreement_test(args.device)
    for k, v in zd.items():
        print(f"  {k}: {v}")

    print("\n=== Frozen Q/V Test ===")
    fr = check_frozen_qv(args.device)
    for k, v in fr.items():
        print(f"  {k}: {v}")

    tol = 1e-5
    ok = (
        zd["max_abs_A1_minus_A2"] < tol
        and zd["max_weight_diff_standard_vs_ensemble"] < tol
        and zd["max_weight_diff_ensemble_vs_iad05"] < tol
        and zd["max_weight_diff_ensemble_vs_iad10"] < tol
        and zd["max_weight_diff_ensemble_vs_shuffled"] < tol
        and fr["q_unchanged"]
        and fr["v_unchanged"]
    )
    print(f"\nUnit test passed: {ok}")

    if args.run_smoke:
        print("\n=== Running 1000-step smoke test (standard_iql) ===")
        subprocess.check_call([
            sys.executable,
            "/code/analysis/iad_iql/scripts/train_frozen_iad_actor.py",
            "--variant", "standard_iql",
            "--smoke-test",
            "--device", args.device,
            "--use-same-checkpoint",
        ])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
