#!/usr/bin/env python3
"""
Multi-GPU experiment launcher with per-GPU concurrency.
Distributes 45 experiments across 4 A800 GPUs, running multiple
experiments concurrently on each GPU for maximum utilization.

Usage:
    python launch.py                          # Run all experiments
    python launch.py --dry-run                # List experiments without running
    python launch.py --workers-per-gpu 4      # 4 concurrent experiments per GPU
    python launch.py --algos CQL,IQL          # Run specific algorithms
"""
import argparse
import os
import sys
import subprocess
import time
import json
from pathlib import Path
from typing import List, Dict, Tuple
import threading
from datetime import datetime
from collections import Counter

# ══════════════════════════════════════════════════════════════════════
# Experiment definitions
# ══════════════════════════════════════════════════════════════════════

BASELINE_ALGOS = ["CQL", "IQL", "TD3_BC"]
VARIANT_ALGOS = ["UA_CQL", "UGH_CQL"]
ALL_ALGOS = BASELINE_ALGOS + VARIANT_ALGOS

ALL_DATASETS = [
    "halfcheetah-medium-v2",
    "hopper-medium-v2",
    "walker2d-medium-v2",
]
ALL_SEEDS = [0, 1, 2]

# Estimated hours per 1M steps (from benchmarking on A800)
SPEED_ESTIMATES = {
    "CQL": 11.5,
    "IQL": 4.1,
    "TD3_BC": 2.3,
    "UA_CQL": 7.2,
    "UGH_CQL": 9.7,
}


def generate_experiments(
    algos: List[str] = None,
    datasets: List[str] = None,
    seeds: List[int] = None,
) -> List[Dict]:
    if algos is None:
        algos = ALL_ALGOS
    if datasets is None:
        datasets = ALL_DATASETS
    if seeds is None:
        seeds = ALL_SEEDS
    experiments = []
    for algo in algos:
        for dataset in datasets:
            for seed in seeds:
                experiments.append({"algo": algo, "dataset": dataset, "seed": seed})
    return experiments


# ══════════════════════════════════════════════════════════════════════
# GPU-aware concurrent launcher
# ══════════════════════════════════════════════════════════════════════

def launch_all(
    experiments: List[Dict],
    gpus: List[int],
    output_dir: str = "/code/results",
    workers_per_gpu: int = 3,
    dry_run: bool = False,
):
    """
    Launch experiments across GPUs with per-GPU concurrency.

    Each GPU runs up to `workers_per_gpu` experiments concurrently.
    CUDA time-sharing + CPU/GPU overlap enables 2-3x wall-clock speedup.
    """
    n_gpus = len(gpus)
    n_exps = len(experiments)
    total_slots = n_gpus * workers_per_gpu

    # ── Print summary ──
    print(f"{'='*70}")
    print(f"Offline RL Benchmark — Multi-GPU Concurrent Launcher")
    print(f"{'='*70}")
    print(f"  Experiments:    {n_exps}")
    print(f"  GPUs:           {gpus}")
    print(f"  Workers/GPU:    {workers_per_gpu}")
    print(f"  Total slots:    {total_slots}")
    print(f"  Output:         {output_dir}")
    print(f"{'='*70}")

    algo_counts = Counter(e["algo"] for e in experiments)
    total_est_hours = sum(SPEED_ESTIMATES.get(e["algo"], 10) for e in experiments)
    for algo, count in sorted(algo_counts.items()):
        est = SPEED_ESTIMATES.get(algo, 10)
        print(f"  {algo:10s}: {count} runs × ~{est}h = ~{count * est:.0f}h")
    print(f"  {'Total':10s}  ~{total_est_hours:.0f}h sequential")
    # With concurrency, wall time is roughly: total / (n_gpus * workers_per_gpu) * slowdown_factor
    # But concurrency has overhead, so assume ~70% efficiency
    est_wall = total_est_hours / (n_gpus * workers_per_gpu) / 0.7
    print(f"  Est. wall time:  ~{est_wall:.0f}h ({est_wall/24:.1f} days) with "
          f"{n_gpus} GPUs × {workers_per_gpu} workers")
    print(f"{'='*70}")

    # ── Greedy distribution: assign each experiment to least-loaded GPU ──
    experiments_sorted = sorted(
        experiments, key=lambda e: SPEED_ESTIMATES.get(e["algo"], 10), reverse=True
    )
    gpu_queues = {gpu: [] for gpu in gpus}
    gpu_loads = {gpu: 0.0 for gpu in gpus}

    for exp in experiments_sorted:
        gpu = min(gpu_loads, key=gpu_loads.get)
        gpu_queues[gpu].append(exp)
        gpu_loads[gpu] += SPEED_ESTIMATES.get(exp["algo"], 10)

    for gpu in gpus:
        print(f"  GPU {gpu}: {len(gpu_queues[gpu])} experiments, "
              f"est. {gpu_loads[gpu]:.0f}h sequential "
              f"→ ~{gpu_loads[gpu]/workers_per_gpu:.0f}h with {workers_per_gpu}× concurrency")

    if dry_run:
        print(f"\n[Dry run] Experiments that would be launched:")
        idx = 0
        for gpu in sorted(gpus):
            print(f"\n  ── GPU {gpu} ──")
            for i, exp in enumerate(gpu_queues[gpu]):
                idx += 1
                est = SPEED_ESTIMATES.get(exp["algo"], "?")
                print(f"  [{idx:2d}] {exp['algo']:8s} | {exp['dataset']:25s} | "
                      f"seed={exp['seed']} | ~{est}h")
        return

    # ── Run: per-GPU manager threads, each spawning up to N concurrent subprocesses ──
    results = {"success": [], "failed": [], "started_at": datetime.now().isoformat()}
    results_lock = threading.Lock()

    job_list = []
    for gpu in gpus:
        for exp in gpu_queues[gpu]:
            job_list.append((gpu, exp))

    print(f"\n{'='*70}")
    print(f"Starting {len(job_list)} experiments on {n_gpus} GPUs "
          f"({workers_per_gpu} concurrent per GPU)...")
    print(f"{'='*70}\n")

    start_time = time.time()
    completed = [0]  # mutable counter for thread-safe increment
    total = len(job_list)

    def gpu_manager(gpu: int, exps: List[Dict]):
        """Manage experiments for one GPU: keep up to workers_per_gpu running at all times."""
        queue = list(exps)  # copy
        running: Dict[subprocess.Popen, Dict] = {}  # active subprocesses

        while queue or running:
            # Start new processes while we have capacity and queued experiments
            while len(running) < workers_per_gpu and queue:
                exp = queue.pop(0)
                cmd = [
                    sys.executable, "/code/train_all.py",
                    "--algo", exp["algo"],
                    "--dataset", exp["dataset"],
                    "--seed", str(exp["seed"]),
                    "--gpu", str(gpu),
                    "--output_dir", output_dir,
                ]
                tag = f"{exp['algo']:8s} | {exp['dataset']:25s} | seed={exp['seed']} | GPU={gpu}"
                print(f"  [START] {tag}  (GPU{gpu}: {len(running)+1}/{workers_per_gpu} active)")
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                running[p] = {"exp": exp, "tag": tag, "start": time.time()}

            if not running:
                break

            # Wait for ANY running process to finish, then collect all finished ones
            time.sleep(5)  # poll interval
            finished = []
            for p in list(running.keys()):
                ret = p.poll()
                if ret is not None:
                    finished.append(p)

            for p in finished:
                info = running.pop(p)
                exp = info["exp"]
                tag = info["tag"]
                elapsed = time.time() - info["start"]
                success = p.returncode == 0
                stdout = p.stdout.read() if p.stdout else ""
                stderr = p.stderr.read() if p.stderr else ""

                # Extract score
                final_score = None
                for line in stdout.split("\n"):
                    if "Best D4RL score:" in line:
                        try:
                            final_score = float(line.split("Best D4RL score:")[1].split("|")[0].strip())
                        except:
                            pass

                score_str = f" D4RL={final_score:.2f}" if final_score is not None else ""
                status = "✓" if success else "✗"
                print(f"  [{status}] {tag} | {elapsed/3600:.1f}h{score_str}")

                with results_lock:
                    if success:
                        results["success"].append(exp)
                    else:
                        results["failed"].append(exp)
                        fail_log = os.path.join(
                            output_dir,
                            f"FAILED_{exp['algo']}_{exp['dataset']}_s{exp['seed']}.log"
                        )
                        with open(fail_log, "w") as f:
                            f.write(stdout + "\n" + stderr)
                    completed[0] += 1

        print(f"  [GPU {gpu}] All {len(exps)} experiments finished.")

    # Launch one manager thread per GPU
    manager_threads = []
    for gpu in gpus:
        t = threading.Thread(target=gpu_manager, args=(gpu, gpu_queues[gpu]), daemon=True)
        t.start()
        manager_threads.append(t)

    # Monitor overall progress while managers run
    while any(t.is_alive() for t in manager_threads):
        time.sleep(30)
        elapsed = time.time() - start_time
        c = completed[0]
        if c > 0 and elapsed > 0:
            rate = c / (elapsed / 3600)
            eta = (total - c) / rate if rate > 0 else float('inf')
            print(f"[Progress] {c}/{total} done | "
                  f"{len(results['success'])} ok, {len(results['failed'])} fail | "
                  f"{elapsed/3600:.1f}h elapsed | ETA {eta:.1f}h")
            # Save progress
            with results_lock:
                progress_path = os.path.join(output_dir, "progress.json")
                with open(progress_path, "w") as f:
                    json.dump({
                        "completed": c, "total": total,
                        "success": len(results["success"]),
                        "failed": len(results["failed"]),
                        "elapsed_h": elapsed / 3600,
                        "eta_h": eta,
                        "updated_at": datetime.now().isoformat(),
                    }, f, indent=2)

    for t in manager_threads:
        t.join()

    total_time = time.time() - start_time
    results["total_time"] = total_time
    results["completed_at"] = datetime.now().isoformat()

    # ── Final summary ──
    print(f"\n{'='*70}")
    print(f"ALL EXPERIMENTS COMPLETE")
    print(f"{'='*70}")
    print(f"  Success: {len(results['success'])}/{n_exps}")
    print(f"  Failed:  {len(results['failed'])}/{n_exps}")
    print(f"  Time:    {total_time/3600:.2f} hours ({total_time/86400:.2f} days)")

    if results["failed"]:
        print(f"\n  Failed experiments:")
        for exp in results["failed"]:
            print(f"    - {exp['algo']} | {exp['dataset']} | seed={exp['seed']}")

    # Save full results
    summary_path = os.path.join(output_dir, "final_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSummary: {summary_path}")

    # ── Collect scores table ──
    print(f"\n{'='*70}")
    print(f"RESULTS TABLE")
    print(f"{'='*70}")
    print(f"{'Algo':10s} | {'Dataset':25s} | {'Seed':5s} | {'Best D4RL':>10s} | {'Status':8s}")
    print("-" * 72)

    all_exps = generate_experiments(
        algos=[e["algo"] for e in experiments],
        datasets=list(set(e["dataset"] for e in experiments)),
        seeds=list(set(e["seed"] for e in experiments)),
    )
    # Deduplicate
    seen = set()
    unique_exps = []
    for e in experiments:
        key = (e["algo"], e["dataset"], e["seed"])
        if key not in seen:
            seen.add(key)
            unique_exps.append(e)

    for exp in unique_exps:
        summary_file = os.path.join(
            output_dir,
            f"{exp['algo']}_{exp['dataset']}_s{exp['seed']}_summary.json"
        )
        if os.path.exists(summary_file):
            with open(summary_file) as f:
                data = json.load(f)
            score = data.get("best_norm_score", float("nan"))
            status = "OK"
        else:
            score = float("nan")
            status = "MISSING"
        print(f"{exp['algo']:10s} | {exp['dataset']:25s} | {exp['seed']:5d} | {score:10.2f} | {status:8s}")

    print(f"\n{'='*70}")


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Offline RL Experiment Launcher")
    parser.add_argument("--algos", type=str, default=None,
                        help="Comma-separated algorithms (default: all)")
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated datasets (default: all)")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds (default: 0,1,2)")
    parser.add_argument("--gpus", type=str, default="0,1,2,3",
                        help="Comma-separated GPU IDs (default: 0,1,2,3)")
    parser.add_argument("--workers-per-gpu", type=int, default=8,
                        help="Concurrent experiments per GPU (default: 8)")
    parser.add_argument("--output_dir", type=str, default="/code/results")
    parser.add_argument("--dry-run", action="store_true",
                        help="List experiments without running")
    args = parser.parse_args()

    algos = args.algos.split(",") if args.algos else None
    datasets = args.datasets.split(",") if args.datasets else None
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    gpus = [int(g) for g in args.gpus.split(",")]

    experiments = generate_experiments(algos, datasets, seeds)

    print(f"\n{'='*70}")
    print(f"Offline RL Benchmark — Multi-GPU Concurrent Launcher")
    print(f"{'='*70}")
    print(f"  Algorithms: {algos or ALL_ALGOS}")
    print(f"  Datasets:   {datasets or ALL_DATASETS}")
    print(f"  Seeds:      {seeds or ALL_SEEDS}")
    print(f"  GPUs:       {gpus}")
    print(f"  Workers/GPU: {args.workers_per_gpu}")
    print(f"  Total:      {len(experiments)} experiments")
    print(f"{'='*70}\n")

    launch_all(experiments, gpus, args.output_dir, args.workers_per_gpu, args.dry_run)


if __name__ == "__main__":
    main()
