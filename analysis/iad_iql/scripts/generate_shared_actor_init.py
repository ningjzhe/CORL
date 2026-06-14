#!/usr/bin/env python3
"""Generate shared actor initialization checkpoint."""
import argparse
import sys

import gym
import torch

sys.path.insert(0, "/code/CORL")
sys.path.insert(0, "/code/analysis/iad_iql/scripts")

from algorithms.offline.iql import GaussianPolicy, set_seed  # noqa: E402
from iad_common import load_norm_stats, build_env_and_buffer  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--output",
        default="/code/analysis/iad_iql/checkpoints/shared_actor_init.pt",
    )
    parser.add_argument(
        "--norm-stats",
        default="/code/analysis/iad_iql/checkpoints/norm_stats.npz",
    )
    parser.add_argument("--env", default="halfcheetah-medium-v2")
    args = parser.parse_args()

    state_mean, state_std = load_norm_stats(args.norm_stats)
    _, eval_env, _, state_dim, action_dim = build_env_and_buffer(
        args.env, "cpu", state_mean, state_std
    )
    max_action = float(eval_env.action_space.high[0])
    set_seed(args.seed, eval_env)
    actor = GaussianPolicy(state_dim, action_dim, max_action)
    torch.save(actor.state_dict(), args.output)
    print(f"Saved shared actor init seed={args.seed} -> {args.output}")


if __name__ == "__main__":
    main()
