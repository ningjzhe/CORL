#!/usr/bin/env python3
"""Evaluate the jointly trained actor from a full IQL checkpoint (reference only)."""
import argparse
import json
import sys
from pathlib import Path

import gym
import numpy as np
import torch

sys.path.insert(0, "/code/CORL")
sys.path.insert(0, "/code/analysis/iad_iql/scripts")

from algorithms.offline.iql import GaussianPolicy, set_seed, wrap_env  # noqa: E402
from iad_common import load_norm_stats  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="/code/analysis/iad_iql/checkpoints/value_seed0/checkpoint_999999.pt",
    )
    parser.add_argument("--env", default="halfcheetah-medium-v2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument(
        "--output",
        default="/code/analysis/iad_iql/results/original_joint_iql_seed0_eval.json",
    )
    args = parser.parse_args()

    state_mean, state_std = load_norm_stats()
    env = wrap_env(gym.make(args.env), state_mean=state_mean, state_std=state_std)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    sd = torch.load(args.checkpoint, map_location=args.device)
    actor = GaussianPolicy(state_dim, action_dim, max_action).to(args.device)
    actor.load_state_dict(sd["actor"])

    episode_seeds = [args.eval_seed + i for i in range(args.n_episodes)]
    returns = []
    norm_scores = []
    for ep_seed in episode_seeds:
        env.seed(ep_seed)
        actor.eval()
        state, done = env.reset(), False
        ep_ret = 0.0
        while not done:
            action = actor.act(state, args.device)
            state, reward, done, _ = env.step(action)
            ep_ret += reward
        returns.append(ep_ret)
        norm_scores.append(float(env.get_normalized_score(ep_ret) * 100.0))

    result = {
        "checkpoint": args.checkpoint,
        "env": args.env,
        "n_episodes": args.n_episodes,
        "eval_seed_base": args.eval_seed,
        "episode_seeds": episode_seeds,
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "D4RL_normalized_score_mean": float(np.mean(norm_scores)),
        "D4RL_normalized_score_std": float(np.std(norm_scores)),
        "returns": [float(r) for r in returns],
        "note": "Jointly trained actor; not directly comparable to frozen-critic extraction.",
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
