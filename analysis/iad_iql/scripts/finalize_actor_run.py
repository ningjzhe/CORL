#!/usr/bin/env python3
"""Final 20-episode evaluation and summary for a frozen actor extraction run."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/code/CORL")
sys.path.insert(0, "/code/analysis/iad_iql/scripts")

from algorithms.offline.iql import GaussianPolicy, set_seed  # noqa: E402
from iad_common import (  # noqa: E402
    build_env_and_buffer,
    flatten_params,
    load_norm_stats,
    load_value_bundle,
    resolve_checkpoint,
)


def read_csv_metrics(csv_path: Path) -> dict:
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"Empty CSV: {csv_path}")

    max_step = max(int(r["training_step"]) for r in rows)
    eval_rows = [
        r for r in rows
        if r.get("D4RL_normalized_score") and str(r["D4RL_normalized_score"]).strip()
    ]
    if not eval_rows:
        raise RuntimeError(f"No evaluation rows in {csv_path}")

    def fval(row, key):
        v = row.get(key, "")
        return float(v) if v not in ("", None) else float("nan")

    last = eval_rows[-1]
    best_row = max(eval_rows, key=lambda r: float(r["D4RL_normalized_score"]))
    return {
        "max_training_step": max_step,
        "last_eval_step": int(last["training_step"]),
        "final_d4rl_score": fval(last, "D4RL_normalized_score"),
        "best_d4rl_score": float(best_row["D4RL_normalized_score"]),
        "best_eval_step": int(best_row["training_step"]),
        "final_return_mean": fval(last, "evaluation_return_mean"),
        "final_return_std": fval(last, "evaluation_return_std"),
        "ess_over_batch_size": fval(last, "ESS_over_batch_size"),
        "clamp_ratio": fval(last, "clamp_ratio"),
        "top_10_percent_weight_mass": fval(last, "top_10_percent_weight_mass"),
        "actor_log_std": fval(last, "actor_log_std"),
        "actor_std": fval(last, "actor_std"),
        "mean_disagreement": fval(last, "mean_disagreement"),
        "corr_A1_A2": fval(last, "corr_A1_A2"),
    }


def check_qv_frozen(
    value_seed0: int,
    value_step: int,
    device: str,
    env: str,
    norm_stats: str,
    checkpoints_dir: str,
) -> dict:
    state_mean, state_std = load_norm_stats(norm_stats)
    _, _, _, state_dim, action_dim = build_env_and_buffer(
        env, device, state_mean, state_std
    )
    ckpt = resolve_checkpoint(value_seed0, value_step, checkpoints_dir)
    bundle = load_value_bundle(ckpt, state_dim, action_dim, device, value_seed0)
    q = flatten_params(bundle.qf)
    v = flatten_params(bundle.vf)
    bundle2 = load_value_bundle(ckpt, state_dim, action_dim, device, value_seed0)
    q2 = flatten_params(bundle2.qf)
    v2 = flatten_params(bundle2.vf)
    return {
        "max_q_delta": float((q2 - q).abs().max()),
        "max_v_delta": float((v2 - v).abs().max()),
        "qv_frozen": bool(torch.allclose(q, q2) and torch.allclose(v, v2)),
    }


def run_final_eval(
    actor_ckpt: Path,
    env: str,
    device: str,
    n_episodes: int,
    eval_seed: int,
    norm_stats: str,
) -> dict:
    state_mean, state_std = load_norm_stats(norm_stats)
    _, eval_env, _, state_dim, action_dim = build_env_and_buffer(
        env, device, state_mean, state_std
    )
    max_action = float(eval_env.action_space.high[0])
    actor = GaussianPolicy(state_dim, action_dim, max_action).to(device)
    actor.load_state_dict(torch.load(actor_ckpt, map_location=device))

    episode_seeds = [eval_seed + i for i in range(n_episodes)]
    returns, norm_scores = [], []
    for ep_seed in episode_seeds:
        eval_env.seed(ep_seed)
        actor.eval()
        state, done = eval_env.reset(), False
        ep_ret = 0.0
        while not done:
            action = actor.act(state, device)
            state, reward, done, _ = eval_env.step(action)
            ep_ret += reward
        returns.append(ep_ret)
        norm_scores.append(float(eval_env.get_normalized_score(ep_ret) * 100.0))
    eval_env.close()
    return {
        "n_episodes": n_episodes,
        "eval_seed_base": eval_seed,
        "episode_seeds": episode_seeds,
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "D4RL_normalized_score_mean": float(np.mean(norm_scores)),
        "D4RL_normalized_score_std": float(np.std(norm_scores)),
        "returns": [float(r) for r in returns],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--value-seed0", type=int, default=0)
    parser.add_argument("--value-step", type=int, default=999999)
    parser.add_argument("--env", default="halfcheetah-medium-v2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--output-dir", default="/code/analysis/iad_iql/results")
    parser.add_argument("--checkpoints-dir", default="/code/analysis/iad_iql/checkpoints")
    parser.add_argument("--log-dir", default="/code/analysis/iad_iql/logs")
    parser.add_argument(
        "--norm-stats",
        default="/code/analysis/iad_iql/checkpoints/norm_stats.npz",
    )
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoints_dir) / args.run_name
    actor_ckpt = ckpt_dir / "actor_final.pt"
    csv_path = Path(args.log_dir) / f"{args.run_name}.csv"
    if not actor_ckpt.exists():
        raise FileNotFoundError(f"Missing actor checkpoint: {actor_ckpt}")
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV log: {csv_path}")

    csv_metrics = read_csv_metrics(csv_path)
    qv = check_qv_frozen(
        args.value_seed0,
        args.value_step,
        args.device,
        args.env,
        args.norm_stats,
        args.checkpoints_dir,
    )
    final_eval = run_final_eval(
        actor_ckpt, args.env, args.device, args.n_episodes, args.eval_seed, args.norm_stats
    )

    summary = {
        "run_name": args.run_name,
        "variant": args.variant,
        "actor_checkpoint": str(actor_ckpt),
        "csv_log": str(csv_path),
        **csv_metrics,
        **qv,
        "final_20ep_eval": final_eval,
    }

    out_path = Path(args.output_dir) / f"{args.run_name}_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
