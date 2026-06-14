#!/usr/bin/env python3
"""View offline RL benchmark results."""
import json, os, sys
from glob import glob
from collections import defaultdict

RESULTS_DIR = "/code/results"

def collect_results():
    """Collect all summary results."""
    results = []
    for f in sorted(glob(f"{RESULTS_DIR}/*_summary.json")):
        try:
            with open(f) as fh:
                d = json.load(fh)
            results.append(d)
        except:
            pass
    return results

def print_table(results):
    """Print results as a formatted table."""
    if not results:
        print("No results found yet. Experiments are still running.")
        return

    # Group by algo, dataset
    grouped = defaultdict(lambda: defaultdict(list))
    for r in results:
        grouped[r["algo"]][r["dataset"]].append(r["best_norm_score"])

    datasets = sorted(set(r["dataset"] for r in results))
    algos = sorted(set(r["algo"] for r in results))

    # Header
    header = f"{'Algo':10s}"
    for ds in datasets:
        ds_short = ds.replace("-medium-v2", "")
        header += f" | {ds_short:>14s}"
    header += f" | {'Avg':>8s}"
    print(header)
    print("-" * len(header))

    # Rows
    for algo in algos:
        row = f"{algo:10s}"
        scores = []
        for ds in datasets:
            vals = grouped[algo].get(ds, [])
            if vals:
                mean = sum(vals) / len(vals)
                std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
                row += f" | {mean:>7.2f}±{std:.0f}"
                scores.extend(vals)
            else:
                row += f" | {'-':>14s}"
        avg = sum(scores) / len(scores) if scores else 0
        row += f" | {avg:>8.2f}"
        print(row)

    print(f"\nDetails:")
    for r in sorted(results, key=lambda x: (x["algo"], x["dataset"], x["seed"])):
        t = r.get("total_time", 0) / 3600
        print(f"  {r['algo']:10s} | {r['dataset']:25s} | s={r['seed']} | D4RL={r['best_norm_score']:.2f} | {t:.1f}h")

def print_progress():
    """Print overall progress."""
    progress_file = f"{RESULTS_DIR}/progress.json"
    if os.path.exists(progress_file):
        with open(progress_file) as f:
            p = json.load(f)
        print(f"Progress: {p['completed']}/{p['total']} | "
              f"OK: {p['success']} | Fail: {p['failed']} | "
              f"Elapsed: {p['elapsed_h']:.1f}h | ETA: {p['eta_h']:.1f}h")
    else:
        # Count completed from launcher log
        log_file = f"{RESULTS_DIR}/launcher.log"
        if os.path.exists(log_file):
            with open(log_file) as f:
                content = f.read()
            ok = content.count("[✓]")
            fail = content.count("[✗]")
            print(f"From log: {ok} completed, {fail} failed (running...)")
        else:
            print("No progress data yet.")

if __name__ == "__main__":
    results = collect_results()
    print_table(results)
    print()
    print_progress()
