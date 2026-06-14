#!/usr/bin/env python3
"""
Checkpoint evaluator — continuously scans for new checkpoints and evaluates them.
Runs as a separate process alongside training.

Usage:
    python3 eval_checkpoints.py --gpus 0,1,2,3 --watch

Scans /code/results/ for checkpoints, evaluates any that haven't been evaluated yet,
and watches for new checkpoints as training progresses.
"""
import argparse
import os
import sys
import json
import time
import glob
import re
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

sys.path.insert(0, "/code")

import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import TransformObservation

from common import (
    load_hdf5_as_dict, compute_mean_std, normalize_states,
    set_seed, eval_actor, get_normalized_score,
    REF_MIN_SCORE, REF_MAX_SCORE,
)
from corl_algos import (
    ContinuousCQL, CQLFullyConnectedQFunction, CQLTanhGaussianPolicy,
    ImplicitQLearning, IQLTwinQ, IQLValueFunction, IQLGaussianPolicy, IQLDeterministicPolicy,
    DecisionTransformer, DTTrainer,
    soft_update,
)
from ua_cql import UACQL
from ugh_cql import UGHCQL

# ── Dataset config ──
DATASET_INFO = {
    "halfcheetah-medium-v2": {"obs_dim": 17, "act_dim": 6, "gym_env": "HalfCheetah-v4"},
    "hopper-medium-v2":      {"obs_dim": 11, "act_dim": 3, "gym_env": "Hopper-v4"},
    "walker2d-medium-v2":    {"obs_dim": 17, "act_dim": 6, "gym_env": "Walker2d-v4"},
}


def parse_run_dir(dirpath: str) -> dict:
    """Parse run directory name like 'CQL-halfcheetah-medium-v2-s0-abc12345'"""
    basename = os.path.basename(dirpath.rstrip("/"))
    # Pattern: {ALGO}-{dataset}-s{seed}-{uuid}
    match = re.match(r"(.+?)-(halfcheetah|hopper|walker2d)-medium-v2-s(\d)-[a-f0-9]+", basename)
    if not match:
        return None
    return {
        "algo": match.group(1),
        "dataset": f"{match.group(2)}-medium-v2",
        "seed": int(match.group(3)),
    }


def load_env_and_data(dataset_name: str):
    """Load gym env and pre-compute normalization stats."""
    info = DATASET_INFO[dataset_name]
    env = gym.make(info["gym_env"])
    max_action = float(env.action_space.high[0])

    ds = load_hdf5_as_dict(f"/code/data/{dataset_name}.hdf5")
    state_mean, state_std = compute_mean_std(ds["observations"], eps=1e-3)
    env = TransformObservation(env, lambda obs: (obs - state_mean) / state_std, env.observation_space)
    return env, info, max_action


def build_model(algo: str, state_dim: int, act_dim: int, max_action: float, device: str):
    """Build a model for a given algorithm (actor only, for eval)."""
    if algo == "CQL":
        actor = CQLTanhGaussianPolicy(state_dim, act_dim, max_action, orthogonal_init=True).to(device)
        # Build full trainer for state dict loading
        critic_1 = CQLFullyConnectedQFunction(state_dim, act_dim, True, 3).to(device)
        critic_2 = CQLFullyConnectedQFunction(state_dim, act_dim, True, 3).to(device)
        trainer = ContinuousCQL(
            critic_1=critic_1, critic_1_optimizer=torch.optim.Adam(critic_1.parameters(), lr=3e-4),
            critic_2=critic_2, critic_2_optimizer=torch.optim.Adam(critic_2.parameters(), lr=3e-4),
            actor=actor, actor_optimizer=torch.optim.Adam(actor.parameters(), lr=3e-5),
            target_entropy=-act_dim, discount=0.99, soft_target_update_rate=0.005, device=device,
            cql_n_actions=10, cql_importance_sample=True, cql_alpha=5.0)
        return trainer, actor

    elif algo == "IQL":
        q = IQLTwinQ(state_dim, act_dim).to(device)
        v = IQLValueFunction(state_dim).to(device)
        actor = IQLGaussianPolicy(state_dim, act_dim, max_action).to(device)
        trainer = ImplicitQLearning(
            max_action, actor, torch.optim.Adam(actor.parameters(), lr=3e-4),
            q, torch.optim.Adam(q.parameters(), lr=3e-4),
            v, torch.optim.Adam(v.parameters(), lr=3e-4),
            iql_tau=0.7, beta=3.0, max_steps=1000000, discount=0.99, tau=0.005, device=device)
        return trainer, actor

    elif algo == "DT":
        # Episode length will be inferred from checkpoint during load
        ep_len = 1020  # fallback default
        model = DecisionTransformer(
            state_dim=state_dim, action_dim=act_dim, seq_len=20, episode_len=ep_len,
            embedding_dim=128, num_layers=3, num_heads=1,
            attention_dropout=0.1, residual_dropout=0.1, embedding_dropout=0.1,
            max_action=max_action).to(device)
        optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lambda s: min((s+1)/10000, 1))
        trainer = DTTrainer(model=model, optimizer=optim, scheduler=scheduler, seq_len=20, device=device)
        return trainer, model

    elif algo == "UA_CQL":
        from types import SimpleNamespace
        cfg = SimpleNamespace(
            obs_dim=state_dim, act_dim=act_dim, hidden_dim=256, ensemble_size=3,
            lr=3e-4, gamma=0.99, tau=0.005, alpha_init=1.0, alpha_lr=1e-4,
            k_pessimism=1.0, k_adapt_rate=0.005, num_random_actions=4, sac_alpha=0.2,
            device=device)
        trainer = UACQL(cfg).to(device)
        return trainer, trainer.pi

    elif algo == "UGH_CQL":
        from types import SimpleNamespace
        cfg = SimpleNamespace(
            obs_dim=state_dim, act_dim=act_dim, hidden_dim=256, ensemble_size=3,
            lr=3e-4, gamma=0.99, tau=0.005, alpha_init=1.0, alpha_lr=1e-4,
            k_pessimism=1.0, k_adapt_rate=0.005, num_random_actions=4, sac_alpha=0.2,
            expectile=0.7, iql_beta=3.0, unc_temperature=0.3, device=device)
        trainer = UGHCQL(cfg).to(device)
        return trainer, trainer.pi

    raise ValueError(f"Unknown algo: {algo}")


def evaluate_checkpoint(ckpt_path: str, run_info: dict, device: str) -> dict:
    """Evaluate a single checkpoint. Returns results dict."""
    dataset = run_info["dataset"]
    algo = run_info["algo"]
    seed = run_info["seed"]
    info = DATASET_INFO[dataset]

    try:
        env, _, max_action = load_env_and_data(dataset)
        set_seed(seed, env)

        # For DT, infer episode_len from checkpoint embedding size
        if algo == "DT":
            ckpt = torch.load(ckpt_path, map_location="cpu")
            timestep_size = ckpt["model"]["timestep_emb.weight"].shape[0]
            ep_len = timestep_size - 20  # seq_len=20
            model = DecisionTransformer(
                state_dim=info["obs_dim"], action_dim=info["act_dim"],
                seq_len=20, episode_len=ep_len,
                embedding_dim=128, num_layers=3, num_heads=1,
                attention_dropout=0.1, residual_dropout=0.1, embedding_dropout=0.1,
                max_action=max_action).to(device)
            optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4, betas=(0.9, 0.999))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lambda s: min((s+1)/10000, 1))
            trainer = DTTrainer(model=model, optimizer=optim, scheduler=scheduler, seq_len=20, device=device)
            trainer.load_state_dict(ckpt)
            trainer.to(device)

            # DT needs proper rollout eval with target returns, not act()
            target_returns = {"halfcheetah-medium-v2": 6000.0, "hopper-medium-v2": 3600.0,
                            "walker2d-medium-v2": 5000.0}
            target_return = target_returns.get(dataset, 3600.0)
            reward_scale = 0.001

            model.eval()
            episode_returns = []
            for ep in range(10):
                env, _, _ = load_env_and_data(dataset)
                set_seed(seed + ep, env)
                states_t = torch.zeros(1, model.episode_len + 1, model.state_dim, dtype=torch.float, device=device)
                actions_t = torch.zeros(1, model.episode_len, model.action_dim, dtype=torch.float, device=device)
                returns_t = torch.zeros(1, model.episode_len + 1, dtype=torch.float, device=device)
                time_steps_t = torch.arange(model.episode_len, dtype=torch.long, device=device).view(1, -1)

                obs, _ = env.reset()
                states_t[:, 0] = torch.as_tensor(obs if not isinstance(obs, tuple) else obs[0], device=device)
                returns_t[:, 0] = torch.as_tensor(target_return * reward_scale, device=device)

                ep_return = 0.0
                for step in range(model.episode_len):
                    pred = model(
                        states_t[:, :step+1][:, -model.seq_len:],
                        actions_t[:, :step+1][:, -model.seq_len:],
                        returns_t[:, :step+1][:, -model.seq_len:],
                        time_steps_t[:, :step+1][:, -model.seq_len:],
                    )
                    action = pred[0, -1].detach().cpu().numpy()
                    step_res = env.step(action)
                    if len(step_res) == 5:
                        ns, r, term, trunc, _ = step_res
                        done = term or trunc
                    else:
                        ns, r, done, _ = step_res
                    actions_t[:, step] = torch.as_tensor(action)
                    states_t[:, step+1] = torch.as_tensor(ns if not isinstance(ns, tuple) else ns[0])
                    returns_t[:, step+1] = torch.as_tensor(returns_t[:, step] - r * reward_scale)
                    ep_return += float(r)
                    if done:
                        break
                episode_returns.append(ep_return)
                env.close()

            eval_scores = np.array(episode_returns)
            raw_mean = eval_scores.mean()
            raw_std = eval_scores.std()
            norm_score = get_normalized_score(dataset, raw_mean) * 100.0

            return {"ckpt": os.path.basename(ckpt_path), "raw_mean": float(raw_mean),
                    "raw_std": float(raw_std), "norm_score": float(norm_score), "status": "ok"}

        else:
            trainer, actor = build_model(algo, info["obs_dim"], info["act_dim"], max_action, device)
            state_dict = torch.load(ckpt_path, map_location=device)
            trainer.load_state_dict(state_dict)
            trainer.to(device)

        # Eval
        eval_scores = eval_actor(env, actor, device, n_episodes=10, seed=seed)
        raw_mean = eval_scores.mean()
        raw_std = eval_scores.std()
        norm_score = get_normalized_score(dataset, raw_mean) * 100.0

        env.close()
        return {"ckpt": os.path.basename(ckpt_path), "raw_mean": float(raw_mean),
                "raw_std": float(raw_std), "norm_score": float(norm_score), "status": "ok"}
    except Exception as e:
        return {"ckpt": os.path.basename(ckpt_path), "status": "error", "error": str(e)}


def scan_checkpoints(results_dir: str) -> List[Tuple[str, dict]]:
    """Find the latest unevaluated checkpoint from each run directory."""
    pending = []
    for run_dir in sorted(glob.glob(f"{results_dir}/*/")):
        run_info = parse_run_dir(run_dir)
        if run_info is None:
            continue
        # Find all checkpoints in this run
        ckpts = sorted(glob.glob(f"{run_dir}/checkpoint_*.pt"),
                       key=lambda x: int(re.search(r"checkpoint_(\d+)\.pt", x).group(1)))
        if not ckpts:
            continue

        # Only evaluate the latest checkpoint
        latest = ckpts[-1]
        step = int(re.search(r"checkpoint_(\d+)\.pt", latest).group(1))

        # Check if this specific checkpoint was already evaluated
        eval_file = os.path.join(run_dir, "eval_results.json")
        if os.path.exists(eval_file):
            with open(eval_file) as f:
                existing = json.load(f)
            already_evaluated = {e.get("step", 0) for e in existing}
            if step in already_evaluated:
                continue

        pending.append((latest, run_info, step))
    return pending


def watch_and_eval(gpus: List[int], results_dir: str, interval: int = 30):
    """Continuously scan and evaluate checkpoints."""
    gpu_idx = 0

    print(f"Watching {results_dir} for checkpoints (interval={interval}s, GPUs={gpus})")
    print(f"Press Ctrl+C to stop\n")

    evaluated = set()  # (run_dir, step) tuples

    while True:
        pending = scan_checkpoints(results_dir)
        new_pending = [(p, r, s) for p, r, s in pending if (os.path.dirname(p), s) not in evaluated]

        if new_pending:
            print(f"\n[{time.strftime('%H:%M:%S')}] Found {len(new_pending)} new checkpoint(s)")
            for ckpt_path, run_info, step in new_pending:
                gpu = gpus[gpu_idx % len(gpus)]
                gpu_idx += 1
                device = f"cuda:{gpu}"

                tag = f"{run_info['algo']:8s} | {run_info['dataset']:25s} | s={run_info['seed']} | step={step} | GPU={gpu}"
                print(f"  [EVAL] {tag}")

                result = evaluate_checkpoint(ckpt_path, run_info, device)
                result["algo"] = run_info["algo"]
                result["dataset"] = run_info["dataset"]
                result["seed"] = run_info["seed"]
                result["step"] = step

                # Save individual eval result
                run_dir = os.path.dirname(ckpt_path)
                eval_file = os.path.join(run_dir, "eval_results.json")
                existing = []
                if os.path.exists(eval_file):
                    with open(eval_file) as f:
                        existing = json.load(f)
                existing.append(result)
                with open(eval_file, "w") as f:
                    json.dump(existing, f, indent=2)

                if result["status"] == "ok":
                    print(f"    -> Raw={result['raw_mean']:.2f}, D4RL={result['norm_score']:.2f}")
                else:
                    print(f"    -> ERROR: {result.get('error', 'unknown')}")

                evaluated.add((os.path.dirname(ckpt_path), step))

            # Update global summary after each batch
            collect_global_summary(results_dir)

        time.sleep(interval)


def collect_global_summary(results_dir: str):
    """Aggregate all eval results into a single summary file."""
    all_evals = []
    for eval_file in sorted(glob.glob(f"{results_dir}/*/eval_results.json")):
        run_dir = os.path.dirname(eval_file)
        run_info = parse_run_dir(run_dir)
        if run_info is None:
            continue
        with open(eval_file) as f:
            evals = json.load(f)
        for e in evals:
            e["algo"] = run_info["algo"]
            e["dataset"] = run_info["dataset"]
            e["seed"] = run_info["seed"]
            e["run"] = os.path.basename(run_dir)
        all_evals.extend(evals)

    with open(os.path.join(results_dir, "all_evals.json"), "w") as f:
        json.dump(all_evals, f, indent=2)

    # Print best scores per run
    best_per_run = defaultdict(lambda: {"best_norm": -np.inf, "best_step": 0})
    for e in all_evals:
        if e.get("status") != "ok":
            continue
        key = (e["algo"], e["dataset"], e["seed"])
        if e["norm_score"] > best_per_run[key]["best_norm"]:
            best_per_run[key]["best_norm"] = e["norm_score"]
            best_per_run[key]["best_step"] = e.get("step", 0)

    print(f"\n  === Current Best Scores ({len(best_per_run)} runs) ===")
    print(f"  {'Algo':10s} | {'Dataset':25s} | {'Seed':5s} | {'Best D4RL':>10s} | {'Step':>10s}")
    print(f"  {'-'*10} | {'-'*25} | {'-'*5} | {'-'*10} | {'-'*10}")
    for (algo, ds, seed), v in sorted(best_per_run.items()):
        print(f"  {algo:10s} | {ds:25s} | {seed:5d} | {v['best_norm']:10.2f} | {str(v['best_step']):>10s}")


def main():
    parser = argparse.ArgumentParser(description="Checkpoint Evaluator")
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--results-dir", type=str, default="/code/results")
    parser.add_argument("--interval", type=int, default=30, help="Scan interval in seconds")
    parser.add_argument("--once", action="store_true", help="Evaluate all pending checkpoints once and exit")
    args = parser.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]

    if args.once:
        pending = scan_checkpoints(args.results_dir)
        print(f"Found {len(pending)} checkpoints to evaluate")
        for ckpt_path, run_info in pending:
            gpu = gpus[0]
            device = f"cuda:{gpu}"
            tag = f"{run_info['algo']} | {run_info['dataset']} | s={run_info['seed']} | {os.path.basename(ckpt_path)}"
            print(f"  [EVAL] {tag}")
            result = evaluate_checkpoint(ckpt_path, run_info, device)
            run_dir = os.path.dirname(ckpt_path)
            eval_file = os.path.join(run_dir, "eval_results.json")
            existing = []
            if os.path.exists(eval_file):
                with open(eval_file) as f:
                    existing = json.load(f)
            result.update(run_info)
            existing.append(result)
            with open(eval_file, "w") as f:
                json.dump(existing, f, indent=2)
            if result["status"] == "ok":
                print(f"    -> D4RL={result['norm_score']:.2f}")
        collect_global_summary(args.results_dir)
    else:
        watch_and_eval(gpus, args.results_dir, args.interval)


if __name__ == "__main__":
    main()
