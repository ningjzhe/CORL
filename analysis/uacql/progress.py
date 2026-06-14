#!/usr/bin/env python3
"""Scan all checkpoints in /code/results/ and report training progress."""
import os, re, json, time
from glob import glob
from collections import defaultdict

RESULTS = "/code/results"
TOTAL_STEPS = 1_000_000


def scan_progress():
    runs = []
    for rundir in sorted(glob(f"{RESULTS}/*/")):
        name = os.path.basename(rundir.rstrip("/"))
        # Parse: {ALGO}-{dataset}-s{seed}-{uuid}
        m = re.match(r"(.+?)-(halfcheetah|hopper|walker2d)-medium-v2-s(\d)-[a-f0-9]+", name)
        if not m:
            continue
        algo, ds_short, seed = m.group(1), m.group(2), int(m.group(3))
        dataset = f"{ds_short}-medium-v2"

        # Find latest checkpoint
        ckpts = sorted(glob(f"{rundir}/checkpoint_*.pt"),
                       key=lambda x: int(re.search(r"checkpoint_(\d+)\.pt", x).group(1)))
        latest_step = int(re.search(r"checkpoint_(\d+)\.pt", ckpts[-1]).group(1)) if ckpts else 0
        n_ckpts = len(ckpts)

        # Check eval results
        eval_file = f"{rundir}/eval_results.json"
        best_d4rl = None
        if os.path.exists(eval_file):
            with open(eval_file) as f:
                evals = json.load(f)
            scores = [e["norm_score"] for e in evals if e.get("status") == "ok"]
            if scores:
                best_d4rl = max(scores)

        runs.append({
            "algo": algo, "dataset": dataset, "seed": seed,
            "step": latest_step, "pct": 100 * latest_step / TOTAL_STEPS,
            "n_ckpts": n_ckpts, "best_d4rl": best_d4rl,
        })

    return runs


def print_table(runs):
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"=== Training Progress @ {now} ===\n")

    # By algo+dataset
    groups = defaultdict(list)
    for r in runs:
        groups[(r["algo"], r["dataset"])].append(r)

    print(f"{'Algo':10s} | {'Dataset':25s} | {'Seeds':30s} | {'Avg Step':>9s} | {'Avg%':>6s} | {'Best D4RL':>10s}")
    print("-" * 105)

    for (algo, ds) in sorted(groups.keys()):
        items = groups[(algo, ds)]
        seed_info = []
        steps = []
        scores = []
        for r in sorted(items, key=lambda x: x["seed"]):
            s_str = f"s={r['seed']}:{r['step']/1000:.0f}k"
            if r["best_d4rl"] is not None:
                s_str += f"({r['best_d4rl']:.1f})"
            seed_info.append(s_str)
            steps.append(r["step"])
            scores.append(r["best_d4rl"])

        avg_step = sum(steps) / len(steps)
        avg_pct = avg_step / TOTAL_STEPS * 100
        valid_scores = [s for s in scores if s is not None]
        best_str = f"{max(valid_scores):.2f}" if valid_scores else "-"

        print(f"{algo:10s} | {ds:25s} | {', '.join(seed_info):30s} | {avg_step/1000:>6.1f}k | {avg_pct:>5.1f}% | {best_str:>10s}")

    # Overall summary
    all_steps = [r["step"] for r in runs]
    completed = sum(1 for r in runs if r["step"] >= TOTAL_STEPS)
    print(f"\n{'='*60}")
    print(f"  Total runs:         {len(runs)}")
    print(f"  Completed (1M):     {completed}/{len(runs)}")
    print(f"  Avg progress:       {sum(all_steps)/len(all_steps)/1000:.0f}k ({sum(all_steps)/len(all_steps)/TOTAL_STEPS*100:.1f}%)")
    print(f"  Min step:           {min(all_steps)/1000:.0f}k")
    print(f"  Max step:           {max(all_steps)/1000:.0f}k")
    evaled = sum(1 for r in runs if r["best_d4rl"] is not None)
    print(f"  Eval coverage:      {evaled}/{len(runs)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    runs = scan_progress()
    print_table(runs)
