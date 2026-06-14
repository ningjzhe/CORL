#!/usr/bin/env python3
"""Run smoke tests for best-checkpoint actor protocol."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/code/analysis/iad_iql/best_checkpoint_protocol")
SCRIPT = "/code/analysis/iad_iql/scripts/train_frozen_iad_actor_best.py"
OUT = ROOT / "results" / "best_protocol_smoke_tests.json"

CONFIGS = [
    {
        "name": "halfcheetah_ensemble_repA_5k",
        "env": "halfcheetah-medium-v2",
        "run_name": "smoke_hc_ensemble_repA",
        "project": ROOT / "halfcheetah_medium_v2",
        "value_ckpt_dir": "/code/analysis/iad_iql/checkpoints",
        "checkpoint_a": "/code/analysis/iad_iql/checkpoints/value_seed0/checkpoint_999999.pt",
        "checkpoint_b": "/code/analysis/iad_iql/checkpoints/value_seed1/checkpoint_999999.pt",
        "norm_stats": ROOT / "halfcheetah_medium_v2/checkpoints/norm_stats.npz",
        "init": ROOT / "halfcheetah_medium_v2/checkpoints/shared_actor_init_seed12345.pt",
        "batches": ROOT / "halfcheetah_medium_v2/checkpoints/actor_training_batch_indices_seed42.npy",
        "actor_seed": 12345,
        "batch_seed": 42,
    },
    {
        "name": "hopper_ensemble_repA_5k",
        "env": "hopper-medium-v2",
        "run_name": "smoke_hopper_ensemble_repA",
        "project": ROOT / "hopper_medium_v2",
        "value_ckpt_dir": "/code/analysis/iad_iql/hopper_medium_v2/checkpoints",
        "checkpoint_a": "/code/analysis/iad_iql/hopper_medium_v2/checkpoints/value_seed0/checkpoint_999999.pt",
        "checkpoint_b": "/code/analysis/iad_iql/hopper_medium_v2/checkpoints/value_seed1/checkpoint_999999.pt",
        "norm_stats": ROOT / "hopper_medium_v2/checkpoints/norm_stats.npz",
        "init": ROOT / "hopper_medium_v2/checkpoints/shared_actor_init_seed12345.pt",
        "batches": ROOT / "hopper_medium_v2/checkpoints/actor_training_batch_indices_seed42.npy",
        "actor_seed": 12345,
        "batch_seed": 42,
    },
]


def run_one(cfg, device="cuda"):
    ckpt_out = cfg["project"] / "checkpoints"
    log_dir = cfg["project"] / "logs"
    results_dir = cfg["project"] / "results"
    cmd = [
        sys.executable, SCRIPT,
        "--variant", "ensemble_mean",
        "--env", cfg["env"],
        "--run-name", cfg["run_name"],
        "--replicate", "A",
        "--seed", "0",
        "--actor-seed", str(cfg["actor_seed"]),
        "--batch-seed", str(cfg["batch_seed"]),
        "--max-updates", "5000",
        "--eval-freq", "5000",
        "--n-eval-episodes", "10",
        "--best-eval-episodes", "50",
        "--final-eval-episodes", "50",
        "--checkpoint-a", cfg["checkpoint_a"],
        "--checkpoint-b", cfg["checkpoint_b"],
        "--checkpoints-dir", cfg["value_ckpt_dir"],
        "--norm-stats", str(cfg["norm_stats"]),
        "--shared-actor-init", str(cfg["init"]),
        "--batch-indices", str(cfg["batches"]),
        "--output-dir", str(ckpt_out),
        "--log-dir", str(log_dir),
        "--results-dir", str(results_dir),
        "--device", device,
        "--smoke-test",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    run_dir = ckpt_out / cfg["run_name"]
    summary_path = results_dir / f"{cfg['run_name']}_summary.json"
    checks = {
        "returncode": proc.returncode,
        "actor_step_005000": (run_dir / "actor_step_005000.pt").exists(),
        "best_actor_pt": (run_dir / "best_actor.pt").exists(),
        "actor_final_pt": (run_dir / "actor_final.pt").exists(),
        "summary_json": summary_path.exists(),
    }
    if summary_path.exists():
        s = json.loads(summary_path.read_text())
        checks.update({
            "best_50ep_score_mean": s.get("best_50ep_score_mean"),
            "final_50ep_score_mean": s.get("final_50ep_score_mean"),
            "qv_max_q_delta": s.get("qv_start_end_max_delta", {}).get("max_q_delta"),
            "qv_max_v_delta": s.get("qv_start_end_max_delta", {}).get("max_v_delta"),
            "qv_frozen": (
                s.get("qv_start_end_max_delta", {}).get("max_q_delta", 1) == 0
                and s.get("qv_start_end_max_delta", {}).get("max_v_delta", 1) == 0
            ),
        })
    checks["passed"] = (
        proc.returncode == 0
        and all(checks[k] for k in (
            "actor_step_005000", "best_actor_pt", "actor_final_pt", "summary_json"
        ))
        and checks.get("qv_frozen", False)
        and checks.get("best_50ep_score_mean") is not None
    )
    if proc.returncode != 0:
        checks["stderr_tail"] = proc.stderr[-2000:]
        checks["stdout_tail"] = proc.stdout[-2000:]
    return checks


def main():
    results = {}
    all_pass = True
    for cfg in CONFIGS:
        print(f"Running smoke test: {cfg['name']}...")
        r = run_one(cfg)
        results[cfg["name"]] = r
        all_pass = all_pass and r["passed"]
        print(json.dumps(r, indent=2))
    results["all_passed"] = all_pass
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"Wrote {OUT}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
