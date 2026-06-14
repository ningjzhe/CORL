#!/usr/bin/env python3
"""Frozen 3-value-system ensemble_mean actor extraction with best-checkpoint protocol."""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "4")))
try:
    torch.set_num_interop_threads(int(os.environ.get("TORCH_NUM_INTEROP_THREADS", "1")))
except RuntimeError:
    pass

sys.path.insert(0, "/code/CORL")
sys.path.insert(0, "/code/analysis/iad_iql/scripts")

from algorithms.offline.iql import GaussianPolicy, set_seed  # noqa: E402
from iad_common import (  # noqa: E402
    actor_bc_loss,
    build_env_and_buffer,
    compute_ensemble3_weights,
    compute_online_advantages,
    ess,
    flatten_params,
    get_batch_from_indices,
    load_norm_stats,
    load_value_bundle,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="ensemble_mean_3q")
    p.add_argument("--checkpoint-a", required=True)
    p.add_argument("--checkpoint-b", required=True)
    p.add_argument("--checkpoint-c", required=True)
    p.add_argument("--norm-stats", required=True)
    p.add_argument("--env", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--replicate", default="A", choices=["A", "B"])
    p.add_argument("--actor-seed", type=int, default=None)
    p.add_argument("--batch-seed", type=int, default=None)
    p.add_argument("--max-updates", type=int, default=100000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eval-freq", type=int, default=5000)
    p.add_argument("--n-eval-episodes", type=int, default=10)
    p.add_argument("--best-eval-episodes", type=int, default=50)
    p.add_argument("--final-eval-episodes", type=int, default=50)
    p.add_argument("--final-eval-seed", type=int, default=0)
    p.add_argument("--beta", type=float, default=3.0)
    p.add_argument("--actor-lr", type=float, default=3e-4)
    p.add_argument("--grad-check-interval", type=int, default=1000)
    p.add_argument("--batch-indices", required=True)
    p.add_argument("--shared-actor-init", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--log-dir", required=True)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--run-name", required=True)
    return p.parse_args()


def qv_parameters(*bundles):
    params = []
    for bundle in bundles:
        params.extend(bundle.qf.parameters())
        params.extend(bundle.vf.parameters())
    return params


def assert_qv_grads_none(qv_params, step: int):
    for p in qv_params:
        if p.grad is not None:
            raise RuntimeError(f"Q/V parameter received grad at step {step}")


def full_freeze_check(bundle, qf_before, vf_before, label: str):
    qf_after = flatten_params(bundle.qf)
    vf_after = flatten_params(bundle.vf)
    max_q = float((qf_after - qf_before).abs().max())
    max_v = float((vf_after - vf_before).abs().max())
    if max_q > 0 or max_v > 0:
        raise RuntimeError(f"Q/V changed at {label}: max_q_delta={max_q}, max_v_delta={max_v}")
    return {"max_q_delta": max_q, "max_v_delta": max_v}


def eval_actor_n_episodes(env, actor, device, n_episodes, eval_seed_base):
    returns, norm_scores = [], []
    for ep_seed in [eval_seed_base + i for i in range(n_episodes)]:
        env.seed(ep_seed)
        actor.eval()
        state, done = env.reset(), False
        ep_ret = 0.0
        while not done:
            action = actor.act(state, device)
            state, reward, done, _ = env.step(action)
            ep_ret += reward
        returns.append(ep_ret)
        norm_scores.append(float(env.get_normalized_score(ep_ret) * 100.0))
    return {
        "n_episodes": n_episodes,
        "eval_seed_base": eval_seed_base,
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "D4RL_normalized_score_mean": float(np.mean(norm_scores)),
        "D4RL_normalized_score_std": float(np.std(norm_scores)),
    }


def load_actor(actor_path, state_dim, action_dim, max_action, device):
    actor = GaussianPolicy(state_dim, action_dim, max_action).to(device)
    actor.load_state_dict(torch.load(actor_path, map_location=device))
    return actor


def main():
    args = parse_args()
    run_name = args.run_name
    out_ckpt = Path(args.output_dir) / run_name
    out_ckpt.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_dir) / f"{run_name}.csv"
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    state_mean, state_std = load_norm_stats(args.norm_stats)
    _, eval_env, replay_buffer, state_dim, action_dim = build_env_and_buffer(
        args.env, args.device, state_mean, state_std
    )
    max_action = float(eval_env.action_space.high[0])

    bundle0 = load_value_bundle(args.checkpoint_a, state_dim, action_dim, args.device, 0)
    bundle1 = load_value_bundle(args.checkpoint_b, state_dim, action_dim, args.device, 1)
    bundle2 = load_value_bundle(args.checkpoint_c, state_dim, action_dim, args.device, 2)
    bundles = (bundle0, bundle1, bundle2)
    qv_params = qv_parameters(*bundles)

    freeze_before = []
    for i, b in enumerate(bundles):
        freeze_before.append((flatten_params(b.qf), flatten_params(b.vf)))
    start_freeze = [
        full_freeze_check(b, freeze_before[i][0], freeze_before[i][1], f"start_seed{i}")
        for i, b in enumerate(bundles)
    ]

    set_seed(args.seed, eval_env)
    actor = GaussianPolicy(state_dim, action_dim, max_action).to(args.device)
    actor.load_state_dict(torch.load(args.shared_actor_init, map_location=args.device))
    actor.train()
    optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

    batch_indices = np.load(args.batch_indices)
    assert batch_indices.shape[1] == args.batch_size

    fieldnames = [
        "training_step", "actor_loss", "mean_A0", "mean_A1", "mean_A2",
        "mean_A_mean_3q", "mean_A_std_3q", "mean_weight", "p90_weight",
        "clamp_ratio", "ESS", "ESS_over_batch_size", "actor_log_std",
        "evaluation_return_mean", "evaluation_return_std", "D4RL_normalized_score",
    ]

    csv_file = open(log_path, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    best_score = float("-inf")
    best_step = 0
    best_10ep_return = float("nan")
    final_10ep_score = float("nan")
    final_10ep_return = float("nan")
    final_metrics = {}

    print(f"[{run_name}] method={args.method} env={args.env} updates={args.max_updates}")
    t_start = time.time()

    for step in range(1, args.max_updates + 1):
        idx = batch_indices[step - 1]
        batch = get_batch_from_indices(replay_buffer, idx)
        states, actions = batch[0].to(args.device), batch[1].to(args.device)

        with torch.no_grad():
            a0 = compute_online_advantages(bundle0, states, actions)
            a1 = compute_online_advantages(bundle1, states, actions)
            a2 = compute_online_advantages(bundle2, states, actions)
            wstats = compute_ensemble3_weights(a0, a1, a2, beta=args.beta)
            weights = wstats["w"]

        loss = actor_bc_loss(actor, states, actions, weights)
        if torch.isnan(loss) or torch.isinf(loss):
            raise RuntimeError(f"NaN/Inf actor loss at step {step}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.grad_check_interval > 0 and step % args.grad_check_interval == 0:
            assert_qv_grads_none(qv_params, step)

        if step % args.eval_freq == 0 or step == args.max_updates:
            with torch.no_grad():
                metrics = {
                    "actor_loss": float(loss.item()),
                    "mean_A0": float(a0.mean()),
                    "mean_A1": float(a1.mean()),
                    "mean_A2": float(a2.mean()),
                    "mean_A_mean_3q": float(wstats["mu_3q"].mean()),
                    "mean_A_std_3q": float(wstats["std_3q"].mean()),
                    "mean_weight": float(weights.mean()),
                    "p90_weight": float(torch.quantile(weights, 0.9)),
                    "clamp_ratio": float(wstats["clamp_ratio"]),
                    "ESS": ess(weights),
                    "ESS_over_batch_size": ess(weights) / args.batch_size,
                    "actor_log_std": float(actor.log_std.mean()),
                }
            scores = eval_actor_n_episodes(
                eval_env, actor, args.device, args.n_eval_episodes, args.seed
            )
            d4rl = scores["D4RL_normalized_score_mean"]
            metrics.update({
                "evaluation_return_mean": scores["return_mean"],
                "evaluation_return_std": scores["return_std"],
                "D4RL_normalized_score": d4rl,
            })
            torch.save(actor.state_dict(), out_ckpt / f"actor_step_{step:06d}.pt")
            if d4rl > best_score:
                best_score = d4rl
                best_step = step
                best_10ep_return = scores["return_mean"]
                torch.save(actor.state_dict(), out_ckpt / "best_actor.pt")
            if step == args.max_updates:
                final_10ep_score = d4rl
                final_10ep_return = scores["return_mean"]
                final_metrics = dict(metrics)
            writer.writerow({"training_step": step, **metrics})
            csv_file.flush()
            print(f"[{run_name}] step={step} d4rl={d4rl}")
            actor.train()

    end_freeze = [
        full_freeze_check(b, freeze_before[i][0], freeze_before[i][1], f"end_seed{i}")
        for i, b in enumerate(bundles)
    ]
    torch.save(actor.state_dict(), out_ckpt / "actor_final.pt")
    csv_file.close()
    eval_env.close()

    _, eval_env50, _, _, _ = build_env_and_buffer(args.env, args.device, state_mean, state_std)
    best_path = out_ckpt / "best_actor.pt"
    final_path = out_ckpt / "actor_final.pt"
    if not best_path.exists():
        torch.save(torch.load(final_path), best_path)
        best_step = args.max_updates

    best_50 = eval_actor_n_episodes(
        eval_env50, load_actor(best_path, state_dim, action_dim, max_action, args.device),
        args.device, args.best_eval_episodes, args.final_eval_seed,
    )
    final_50 = eval_actor_n_episodes(
        eval_env50, load_actor(final_path, state_dim, action_dim, max_action, args.device),
        args.device, args.final_eval_episodes, args.final_eval_seed,
    )
    eval_env50.close()

    summary = {
        "run_name": run_name,
        "env": args.env,
        "method": args.method,
        "replicate": args.replicate,
        "actor_seed": args.actor_seed if args.actor_seed is not None else args.seed,
        "batch_seed": args.batch_seed,
        "value_checkpoints": {"seed0": args.checkpoint_a, "seed1": args.checkpoint_b, "seed2": args.checkpoint_c},
        "updates": args.max_updates,
        "eval_freq": args.eval_freq,
        "best_step": int(best_step),
        "best_10ep_score": float(best_score),
        "best_10ep_return": float(best_10ep_return),
        "best_50ep_score_mean": best_50["D4RL_normalized_score_mean"],
        "best_50ep_score_std": best_50["D4RL_normalized_score_std"],
        "final_50ep_score_mean": final_50["D4RL_normalized_score_mean"],
        "final_50ep_score_std": final_50["D4RL_normalized_score_std"],
        "best_final_gap_50ep": float(
            best_50["D4RL_normalized_score_mean"] - final_50["D4RL_normalized_score_mean"]
        ),
        "ESS_B_final": final_metrics.get("ESS_over_batch_size"),
        "clamp_ratio_final": final_metrics.get("clamp_ratio"),
        "actor_log_std_final": final_metrics.get("actor_log_std"),
        "qv_start_end_max_delta": {
            "start": start_freeze,
            "end": end_freeze,
            "max_q_delta": max(x["max_q_delta"] for x in start_freeze + end_freeze),
            "max_v_delta": max(x["max_v_delta"] for x in start_freeze + end_freeze),
        },
        "status": "completed",
        "script": "train_frozen_ensemble3_actor_best.py",
        "total_seconds": time.time() - t_start,
    }
    with open(results_dir / f"{run_name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_ckpt / "meta.json", "w") as f:
        json.dump({**vars(args), **summary}, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
