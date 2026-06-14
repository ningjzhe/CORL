#!/usr/bin/env python3
"""Compute and save state normalization statistics (same logic as CORL IQL)."""
import argparse
import sys
from pathlib import Path

import gym
import d4rl
import numpy as np

sys.path.insert(0, "/code/CORL")
from algorithms.offline.iql import compute_mean_std  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="halfcheetah-medium-v2")
    parser.add_argument(
        "--output",
        default="/code/analysis/iad_iql/checkpoints/norm_stats.npz",
    )
    args = parser.parse_args()

    env = gym.make(args.env)
    dataset = d4rl.qlearning_dataset(env)
    mean, std = compute_mean_std(dataset["observations"], eps=1e-3)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, state_mean=mean, state_std=std, env=args.env, eps=1e-3)
    print(f"Saved norm stats to {out}")
    print(f"  state_dim={mean.shape[0]}, mean range=[{mean.min():.4f}, {mean.max():.4f}]")


if __name__ == "__main__":
    main()
