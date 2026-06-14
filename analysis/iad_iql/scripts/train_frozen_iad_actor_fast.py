#!/usr/bin/env python3
"""Fast frozen-critic IAD-IQL actor extraction (no per-step flatten freeze check)."""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Limit CPU thread oversubscription before heavy imports.
torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "4")))
try:
    torch.set_num_interop_threads(int(os.environ.get("TORCH_NUM_INTEROP_THREADS", "1")))
except RuntimeError:
    pass

sys.path.insert(0, "/code/CORL")
sys.path.insert(0, "/code/analysis/iad_iql/scripts")

from algorithms.offline.iql import GaussianPolicy, eval_actor, set_seed  # noqa: E402
from iad_common import (  # noqa: E402
    VARIANTS,
    actor_bc_loss,
    build_env_and_buffer,
    compute_online_advantages,
    compute_weights,
    ess,
    flatten_params,
    get_batch_from_indices,
    load_norm_stats,
    load_value_bundle,
    resolve_checkpoint,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, choices=VARIANTS)
    p.add_argument("--value-seed0", type=int, default=0)
    p.add_argument("--value-seed1", type=int, default=1)
    p.add_argument("--value-step", type=int, default=999999)
    p.add_argument(
        "--checkpoints-dir",
        default="/code/analysis/iad_iql/checkpoints",
        help="Directory containing value_seed{N}/ checkpoints",
    )
    p.add_argument(
        "--checkpoint-a",
        default="",
        help="Explicit path to value system A checkpoint (overrides --value-seed0/--value-step)",
    )
    p.add_argument(
        "--checkpoint-b",
        default="",
        help="Explicit path to value system B checkpoint (overrides --value-seed1/--value-step)",
    )
    p.add_argument(
        "--norm-stats",
        default="/code/analysis/iad_iql/checkpoints/norm_stats.npz",
    )
    p.add_argument("--env", default="halfcheetah-medium-v2")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0, help="actor init / eval seed")
    p.add_argument("--max-updates", type=int, default=100000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eval-freq", type=int, default=5000)
    p.add_argument("--n-eval-episodes", type=int, default=10)
    p.add_argument("--beta", type=float, default=3.0)
    p.add_argument("--actor-lr", type=float, default=3e-4)
    p.add_argument(
        "--freeze-check-interval",
        type=int,
        default=0,
        help="0=start/end only; N=also full flatten check every N steps",
    )
    p.add_argument(
        "--grad-check-interval",
        type=int,
        default=1000,
        help="Light Q/V grad-none assert every N steps (no CPU copy)",
    )
    p.add_argument(
        "--batch-indices",
        default="/code/analysis/iad_iql/checkpoints/actor_training_batch_indices.npy",
    )
    p.add_argument(
        "--shared-actor-init",
        default="/code/analysis/iad_iql/checkpoints/shared_actor_init.pt",
    )
    p.add_argument("--output-dir", default="/code/analysis/iad_iql/checkpoints")
    p.add_argument("--log-dir", default="/code/analysis/iad_iql/logs")
    p.add_argument("--run-name", default="", help="output subdir and csv basename")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--use-same-checkpoint", action="store_true")
    return p.parse_args()


def qv_parameters(bundle0, bundle1):
    params = []
    for bundle in (bundle0, bundle1):
        params.extend(bundle.qf.parameters())
        params.extend(bundle.vf.parameters())
    return params


def assert_qv_grads_none(qv_params, step: int):
    for p in qv_params:
        if p.grad is not None:
            raise RuntimeError(f"Q/V parameter received grad at step {step}")


def full_freeze_check(bundle0, qf_before, vf_before, label: str):
    qf_after = flatten_params(bundle0.qf)
    vf_after = flatten_params(bundle0.vf)
    max_q = float((qf_after - qf_before).abs().max())
    max_v = float((vf_after - vf_before).abs().max())
    if max_q > 0 or max_v > 0:
        raise RuntimeError(f"Q/V changed at {label}: max_q_delta={max_q}, max_v_delta={max_v}")
    return {"max_q_delta": max_q, "max_v_delta": max_v}


def log_metrics_row(writer, step, metrics):
    writer.writerow({"training_step": step, **metrics})


def main():
    args = parse_args()
    if args.smoke_test:
        args.max_updates = 1000
        args.eval_freq = 500
        args.use_same_checkpoint = True

    run_name = args.run_name or args.variant
    out_ckpt = Path(args.output_dir) / run_name
    out_ckpt.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_dir) / f"{run_name}.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    state_mean, state_std = load_norm_stats(args.norm_stats)
    _, eval_env, replay_buffer, state_dim, action_dim = build_env_and_buffer(
        args.env, args.device, state_mean, state_std
    )
    max_action = float(eval_env.action_space.high[0])

    need_two = args.variant != "standard_iql" and not args.use_same_checkpoint
    if args.checkpoint_a:
        ckpt0 = args.checkpoint_a
    else:
        ckpt0 = resolve_checkpoint(args.value_seed0, args.value_step, args.checkpoints_dir)
    if args.checkpoint_b:
        ckpt1 = args.checkpoint_b
    elif need_two:
        ckpt1 = resolve_checkpoint(args.value_seed1, args.value_step, args.checkpoints_dir)
    else:
        ckpt1 = ckpt0
    bundle0 = load_value_bundle(ckpt0, state_dim, action_dim, args.device, args.value_seed0)
    bundle1 = load_value_bundle(ckpt1, state_dim, action_dim, args.device, args.value_seed1)
    qv_params = qv_parameters(bundle0, bundle1)

    qf_flat_before = flatten_params(bundle0.qf)
    vf_flat_before = flatten_params(bundle0.vf)
    start_freeze = full_freeze_check(bundle0, qf_flat_before, vf_flat_before, "start")

    set_seed(args.seed, eval_env)
    actor = GaussianPolicy(state_dim, action_dim, max_action).to(args.device)
    actor_init = torch.load(args.shared_actor_init, map_location=args.device)
    actor.load_state_dict(actor_init)
    actor_flat_before = flatten_params(actor)
    actor.train()
    optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

    batch_indices = np.load(args.batch_indices)
    assert batch_indices.shape[1] == args.batch_size

    fieldnames = [
        "training_step", "actor_loss", "mean_A1", "mean_A2", "std_A1", "std_A2",
        "corr_A1_A2", "mean_mu_A", "mean_disagreement", "p50_disagreement",
        "p90_disagreement", "p99_disagreement", "mean_normalized_disagreement",
        "mean_weight", "p50_weight", "p90_weight", "p99_weight", "clamp_ratio",
        "ESS", "ESS_over_batch_size", "top_10_percent_weight_mass",
        "actor_log_std", "actor_std", "mean_w_base", "mean_w_tilde", "mean_w_final",
        "evaluation_return_mean", "evaluation_return_std", "D4RL_normalized_score",
    ]

    write_header = not log_path.exists() or args.smoke_test
    csv_file = open(log_path, "w" if args.smoke_test else "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    print(
        f"Training variant={args.variant}, run={run_name}, updates={args.max_updates}, "
        f"device={args.device}, freeze_check_interval={args.freeze_check_interval}"
    )
    t_start = time.time()
    train_seconds = 0.0
    eval_seconds = 0.0

    for step in range(1, args.max_updates + 1):
        t_step = time.time()
        idx = batch_indices[step - 1]
        batch = get_batch_from_indices(replay_buffer, idx)
        states, actions = batch[0].to(args.device), batch[1].to(args.device)

        with torch.no_grad():
            a1 = compute_online_advantages(bundle0, states, actions)
            a2 = compute_online_advantages(bundle1, states, actions)
            wstats = compute_weights(a1, a2, args.variant, beta=args.beta)
            weights = wstats["w"]

        loss = actor_bc_loss(actor, states, actions, weights)
        if torch.isnan(loss) or torch.isinf(loss):
            raise RuntimeError(f"NaN/Inf actor loss at step {step} variant={args.variant}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.grad_check_interval > 0 and step % args.grad_check_interval == 0:
            assert_qv_grads_none(qv_params, step)

        if (
            args.freeze_check_interval > 0
            and step % args.freeze_check_interval == 0
        ):
            full_freeze_check(bundle0, qf_flat_before, vf_flat_before, f"step {step}")

        train_seconds += time.time() - t_step

        if step % args.eval_freq == 0 or step == 1 or (args.smoke_test and step == args.max_updates):
            t_eval = time.time()
            with torch.no_grad():
                corr = float(np.corrcoef(a1.cpu().numpy(), a2.cpu().numpy())[0, 1])
                metrics = {
                    "actor_loss": float(loss.item()),
                    "mean_A1": float(a1.mean()),
                    "mean_A2": float(a2.mean()),
                    "std_A1": float(a1.std()),
                    "std_A2": float(a2.std()),
                    "corr_A1_A2": corr,
                    "mean_mu_A": float(wstats["mu_a"].mean()),
                    "mean_disagreement": float(wstats["u_a"].mean()),
                    "p50_disagreement": float(torch.quantile(wstats["u_a"], 0.5)),
                    "p90_disagreement": float(torch.quantile(wstats["u_a"], 0.9)),
                    "p99_disagreement": float(torch.quantile(wstats["u_a"], 0.99)),
                    "mean_normalized_disagreement": float(wstats["hat_u_a"].mean()),
                    "mean_weight": float(weights.mean()),
                    "p50_weight": float(torch.quantile(weights, 0.5)),
                    "p90_weight": float(torch.quantile(weights, 0.9)),
                    "p99_weight": float(torch.quantile(weights, 0.99)),
                    "clamp_ratio": float((weights >= 99.999).float().mean()),
                    "ESS": ess(weights),
                    "ESS_over_batch_size": ess(weights) / args.batch_size,
                    "top_10_percent_weight_mass": float(
                        weights[weights >= torch.quantile(weights, 0.9)].sum()
                        / (weights.sum() + 1e-8)
                    ),
                    "actor_log_std": float(actor.log_std.mean()),
                    "actor_std": float(torch.exp(actor.log_std.clamp(-20, 2)).mean()),
                    "mean_w_base": float(
                        wstats.get("mean_w_base", wstats.get("w_base", weights).mean())
                    ),
                    "mean_w_tilde": float(wstats["w_tilde"].mean()) if "w_tilde" in wstats else "",
                    "mean_w_final": float(wstats.get("mean_w_final", weights.mean())),
                }

            if step % args.eval_freq == 0 or (args.smoke_test and step == args.max_updates):
                scores = eval_actor(
                    eval_env, actor, args.device, args.n_eval_episodes, args.seed
                )
                metrics["evaluation_return_mean"] = float(scores.mean())
                metrics["evaluation_return_std"] = float(scores.std())
                metrics["D4RL_normalized_score"] = float(
                    eval_env.get_normalized_score(scores.mean()) * 100.0
                )
            else:
                metrics["evaluation_return_mean"] = ""
                metrics["evaluation_return_std"] = ""
                metrics["D4RL_normalized_score"] = ""

            log_metrics_row(writer, step, metrics)
            csv_file.flush()
            eval_seconds += time.time() - t_eval
            if step == 1000:
                avg1k = 1000.0 / (time.time() - t_start)
                print(f"[{run_name}] first_1000_avg_steps_per_sec={avg1k:.2f}")
            print(f"[{run_name}] step={step} loss={loss.item():.4f} mean_w={weights.mean():.4f}")

    end_freeze = full_freeze_check(bundle0, qf_flat_before, vf_flat_before, "end")
    actor_flat_after = flatten_params(actor)
    actor_changed = not torch.allclose(actor_flat_before, actor_flat_after)
    max_actor_delta = float((actor_flat_after - actor_flat_before).abs().max())

    torch.save(actor.state_dict(), out_ckpt / "actor_final.pt")
    total_seconds = time.time() - t_start
    meta = {
        **vars(args),
        "run_name": run_name,
        "script": "train_frozen_iad_actor_fast.py",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_start)),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_seconds": total_seconds,
        "train_seconds": train_seconds,
        "eval_seconds": eval_seconds,
        "avg_steps_per_sec": args.max_updates / total_seconds,
        "start_freeze_check": start_freeze,
        "end_freeze_check": end_freeze,
        "actor_changed": actor_changed,
        "max_actor_delta": max_actor_delta,
    }
    with open(out_ckpt / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    csv_file.close()
    print(f"Saved actor checkpoint to {out_ckpt / 'actor_final.pt'}")
    print(
        f"[{run_name}] done: {meta['avg_steps_per_sec']:.2f} steps/s, "
        f"train={train_seconds:.1f}s eval={eval_seconds:.1f}s"
    )


if __name__ == "__main__":
    main()
