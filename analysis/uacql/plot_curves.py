#!/usr/bin/env python3
"""Plot D4RL score vs training steps for all algorithms on each dataset."""
import json, os, re
from glob import glob
from collections import defaultdict
import numpy as np

RESULTS = "/code/results"
DATASETS = ["halfcheetah-medium-v2", "hopper-medium-v2", "walker2d-medium-v2"]
OUTPUT = "/code/results/learning_curves.png"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Collect data ──
# Structure: data[algo][dataset] = {step: [scores from 3 seeds]}
data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

# Primary: results.json (has all eval points per run, saved every 5k steps)
for results_file in sorted(glob(f"{RESULTS}/*/results.json")):
    run_dir = os.path.dirname(results_file)
    name = os.path.basename(run_dir)
    m = re.match(r"(.+?)-(halfcheetah|hopper|walker2d)-medium-v2-s(\d)-[a-f0-9]+", name)
    if not m:
        continue
    algo, ds_short, seed = m.group(1), m.group(2), int(m.group(3))
    dataset = f"{ds_short}-medium-v2"

    with open(results_file) as f:
        d = json.load(f)
    for entry in d.get("results", []):
        step = entry.get("step", 0)
        score = entry.get("norm_score")
        if score is not None and step > 0:
            data[algo][dataset][step].append(score)

# Supplement: eval_results.json (may have newer steps not yet in results.json)
for eval_file in sorted(glob(f"{RESULTS}/*/eval_results.json")):
    run_dir = os.path.dirname(eval_file)
    name = os.path.basename(run_dir)
    m = re.match(r"(.+?)-(halfcheetah|hopper|walker2d)-medium-v2-s(\d)-[a-f0-9]+", name)
    if not m:
        continue
    algo, ds_short, seed = m.group(1), m.group(2), int(m.group(3))
    dataset = f"{ds_short}-medium-v2"

    with open(eval_file) as f:
        evals = json.load(f)
    for e in evals:
        if e.get("status") != "ok":
            continue
        step = e.get("step", 0)
        score = e.get("norm_score")
        if score is not None and step > 0:
            # Add if this step+seed combo doesn't already exist
            existing_seeds = {s for s in data[algo][dataset].get(step, [])}  # can't track seed, just add
            if len(data[algo][dataset].get(step, [])) < 3:
                data[algo][dataset][step].append(score)

# ── Aggregate: average across seeds ──
def aggregate(points):
    """points: dict step -> [scores]; return sorted (steps, means, stds)"""
    steps = sorted(points.keys())
    means = [np.mean(points[s]) for s in steps]
    stds = [np.std(points[s]) if len(points[s]) > 1 else 0.0 for s in steps]
    return np.array(steps), np.array(means), np.array(stds)

# ── Plot ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
colors = {"CQL": "#1f77b4", "IQL": "#ff7f0e", "DT": "#2ca02c",
          "UA_CQL": "#d62728", "UGH_CQL": "#9467bd"}

for i, ds in enumerate(DATASETS):
    ax = axes[i]
    ds_short = ds.replace("-medium-v2", "")
    ax.set_title(ds_short, fontsize=13, fontweight="bold")
    ax.set_xlabel("Training Steps", fontsize=10)
    ax.set_ylabel("D4RL Normalized Score", fontsize=10)
    ax.grid(True, alpha=0.3)

    for algo in sorted(data.keys()):
        if ds not in data[algo]:
            continue
        steps, means, stds = aggregate(data[algo][ds])
        if len(steps) == 0:
            continue
        color = colors.get(algo, "#333333")
        ax.plot(steps, means, color=color, linewidth=2.0, label=algo, alpha=0.95)
        # Only draw band where we have ≥2 seeds
        valid = np.array([len(data[algo][ds].get(s, [])) for s in steps]) >= 2
        if valid.any():
            ax.fill_between(steps, means - stds, means + stds, where=valid,
                           color=color, alpha=0.2, linewidth=0)

    ax.legend(fontsize=8, loc="lower right")

fig.suptitle("Offline RL Learning Curves (3 seeds, mean ± std)", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(OUTPUT, dpi=150, bbox_inches="tight")
print(f"Saved: {OUTPUT}")
plt.close()

# ── Also print summary table ──
print(f"\n{'='*70}")
print(f"Final Scores Summary")
print(f"{'='*70}")
print(f"{'Algo':10s} | {'Dataset':25s} | {'Latest':>8s} | {'Best':>8s} | {'Seeds':>5s}")
print("-" * 65)
for algo in sorted(data.keys()):
    for ds in DATASETS:
        if ds not in data[algo]:
            continue
        steps, means, stds = aggregate(data[algo][ds])
        if len(steps) == 0:
            continue
        latest = means[-1]
        best = means.max()
        n_seeds = len(data[algo][ds][steps[-1]])
        print(f"{algo:10s} | {ds:25s} | {latest:>8.2f} | {best:>8.2f} | {n_seeds:>5d}")
