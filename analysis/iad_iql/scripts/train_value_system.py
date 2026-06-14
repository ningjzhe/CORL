#!/usr/bin/env python3
"""Launch standard CORL IQL training with fixed checkpoint paths for IAD-IQL."""
import argparse
import os
import sys

sys.path.insert(0, "/code/CORL")

import pyrallis
from algorithms.offline.iql import TrainConfig, train as wrapped_train

# pyrallis.wrap() on iql.train; call underlying function to avoid double CLI parsing.
_train = wrapped_train.__wrapped__


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--project-dir",
        default="/code/analysis/iad_iql",
        help="Project root; checkpoints -> {project_dir}/checkpoints/value_seed{N}",
    )
    pre_args, remaining = pre.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    config = pyrallis.parse(config_class=TrainConfig)
    seed = config.seed
    ckpt_dir = os.path.join(pre_args.project_dir, "checkpoints", f"value_seed{seed}")
    config.name = f"value_seed{seed}"
    config.checkpoints_path = ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, "config.yaml"), "w") as f:
        pyrallis.dump(config, f)
    print(f"Training value system seed={seed} -> {ckpt_dir}")
    _train(config)


if __name__ == "__main__":
    main()
