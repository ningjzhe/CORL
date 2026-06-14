#!/usr/bin/env python3
"""Generate shared batch indices for fair actor extraction comparison."""
import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-size", type=int, default=999000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-updates", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="/code/analysis/iad_iql/checkpoints/actor_training_batch_indices.npy",
    )
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    indices = rng.randint(
        0, args.dataset_size, size=(args.num_updates, args.batch_size), dtype=np.int32
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, indices)
    print(f"Saved batch indices shape={indices.shape} to {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
