#!/usr/bin/env python3
"""Unit tests and speed tests for train_frozen_iad_actor_fast.py."""
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/code/CORL")
sys.path.insert(0, "/code/analysis/iad_iql/scripts")

from run_pipeline_tests import run_zero_disagreement_test, check_frozen_qv  # noqa: E402

ROOT = Path("/code/analysis/iad_iql")
SCRIPTS = ROOT / "scripts"
RESULTS = ROOT / "results"
FAST = SCRIPTS / "train_frozen_iad_actor_fast.py"

ENV_PREFIX = (
    "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 "
    "NUMEXPR_NUM_THREADS=4 TORCH_NUM_THREADS=4 TORCH_NUM_INTEROP_THREADS=1 "
    "MUJOCO_GL=osmesa D4RL_SUPPRESS_IMPORT_ERROR=1"
)


def run_unit_tests(device="cuda"):
    zd = run_zero_disagreement_test(device)
    tol = 1e-5
    zero_ok = (
        zd["max_abs_A1_minus_A2"] < tol
        and zd["max_weight_diff_standard_vs_ensemble"] < tol
        and zd["max_weight_diff_ensemble_vs_iad05"] < tol
        and zd["max_weight_diff_ensemble_vs_iad10"] < tol
        and zd["max_weight_diff_ensemble_vs_shuffled"] < tol
        and zd["max_loss_diff"] < tol
    )

    run_name = "fast_freeze_smoke_1000"
    log_csv = ROOT / "logs" / f"{run_name}.csv"
    ckpt_dir = ROOT / "checkpoints" / run_name
    if log_csv.exists():
        log_csv.unlink()
    subprocess.check_call([
        "bash", "-lc",
        f"source /opt/conda/etc/profile.d/conda.sh && conda activate iql && "
        f"{ENV_PREFIX} && python {FAST} "
        f"--variant standard_iql --run-name {run_name} "
        f"--smoke-test --device {device} --use-same-checkpoint"
    ])

    meta = json.loads((ckpt_dir / "meta.json").read_text())
    freeze_ok = (
        meta["end_freeze_check"]["max_q_delta"] == 0.0
        and meta["end_freeze_check"]["max_v_delta"] == 0.0
        and meta["actor_changed"]
    )

    return {
        "zero_disagreement": zd,
        "zero_disagreement_passed": zero_ok,
        "freeze_smoke_run": run_name,
        "freeze_smoke_meta": {
            "max_q_delta": meta["end_freeze_check"]["max_q_delta"],
            "max_v_delta": meta["end_freeze_check"]["max_v_delta"],
            "actor_changed": meta["actor_changed"],
            "max_actor_delta": meta["max_actor_delta"],
            "avg_steps_per_sec": meta["avg_steps_per_sec"],
        },
        "freeze_smoke_passed": freeze_ok,
        "all_unit_tests_passed": zero_ok and freeze_ok,
    }


def run_speed_case(variant, gpu, run_name, device="cuda"):
    log_csv = ROOT / "logs" / f"{run_name}.csv"
    stdout_log = ROOT / "logs" / f"{run_name}.stdout.log"
    ckpt_dir = ROOT / "checkpoints" / run_name
    for p in (log_csv, stdout_log, ckpt_dir / "meta.json"):
        if p.exists():
            p.unlink()
    if ckpt_dir.exists():
        for f in ckpt_dir.iterdir():
            f.unlink()

    cmd = (
        f"source /opt/conda/etc/profile.d/conda.sh && conda activate iql && "
        f"{ENV_PREFIX} && CUDA_VISIBLE_DEVICES={gpu} python {FAST} "
        f"--variant {variant} --run-name {run_name} "
        f"--max-updates 5000 --eval-freq 5000 --n-eval-episodes 3 "
        f"--device cuda --freeze-check-interval 0"
    )
    t0 = time.time()
    with open(stdout_log, "w") as out:
        subprocess.check_call(["bash", "-lc", cmd], stdout=out, stderr=subprocess.STDOUT)
    wall = time.time() - t0
    meta = json.loads((ckpt_dir / "meta.json").read_text())

    import csv
    with open(log_csv) as f:
        rows = list(csv.DictReader(f))
    last = rows[-1]

    return {
        "variant": variant,
        "run_name": run_name,
        "gpu": gpu,
        "wall_seconds": wall,
        "total_seconds": meta["total_seconds"],
        "train_seconds": meta["train_seconds"],
        "eval_seconds": meta["eval_seconds"],
        "updates_per_sec": meta["avg_steps_per_sec"],
        "final_d4rl_score": float(last["D4RL_normalized_score"]),
        "ess_over_batch_size": float(last["ESS_over_batch_size"]),
        "clamp_ratio": float(last["clamp_ratio"]),
        "end_freeze_check": meta["end_freeze_check"],
    }


def main():
    device = "cuda"
    unit = run_unit_tests(device)
    (RESULTS / "fast_actor_unit_tests.json").write_text(json.dumps(unit, indent=2))
    if not unit["all_unit_tests_passed"]:
        print("UNIT TESTS FAILED")
        sys.exit(1)

    speed = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "target_updates_per_sec": 10.0,
        "cases": [],
    }
    speed["cases"].append(run_speed_case("ensemble_mean", 0, "speedtest_ensemble_mean_5k"))
    speed["cases"].append(run_speed_case("iad_lambda_0.5", 1, "speedtest_iad_0_5_5k"))

    min_rate = min(c["updates_per_sec"] for c in speed["cases"])
    speed["min_updates_per_sec"] = min_rate
    speed["passed"] = min_rate >= 10.0
    (RESULTS / "fast_actor_speedtest.json").write_text(json.dumps(speed, indent=2))

    print(json.dumps({"unit": unit["all_unit_tests_passed"], "speed_passed": speed["passed"], "min_rate": min_rate}, indent=2))
    sys.exit(0 if speed["passed"] else 1)


if __name__ == "__main__":
    main()
