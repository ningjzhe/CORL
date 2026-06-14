#!/usr/bin/env python3
"""30-minute health check for concurrent seed2 value training."""
import argparse
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path


def parse_metrics(log_path: Path, pid: int):
    alive = bool(subprocess.run(["ps", "-p", str(pid)], capture_output=True).returncode == 0)
    step = 0
    d4rl = float("nan")
    errors = []
    if log_path.exists():
        text = log_path.read_text(errors="ignore")
        for line in text.splitlines():
            m = re.match(r"Time steps:\s*(\d+)", line.strip())
            if m:
                step = int(m.group(1))
            if "D4RL score:" in line:
                try:
                    d4rl = float(line.split("D4RL score:")[-1].strip())
                except ValueError:
                    pass
        for pat in ("Traceback", "NaN", "Inf", "CUDA out of memory", "OOM"):
            if pat in text and "UserWarning" not in text:
                errors.append(pat)
    ckpt_dir = log_path.parent.parent / "checkpoints" / log_path.stem.replace(".log", "")
    ckpts = list(ckpt_dir.glob("checkpoint_*.pt")) if ckpt_dir.exists() else []
    etimes = 0
    if alive:
        r = subprocess.run(["ps", "-p", str(pid), "-o", "etimes="], capture_output=True, text=True)
        etimes = int(r.stdout.strip() or 0)
    sps = step / etimes if etimes > 0 else 0.0
    eta_h = (1_000_000 - step) / sps / 3600 if sps > 0 else float("nan")
    return {
        "pid": pid,
        "alive": alive,
        "step": step,
        "d4rl": d4rl,
        "ckpt_count": len(ckpts),
        "elapsed_s": etimes,
        "steps_per_sec": round(sps, 2),
        "eta_h": round(eta_h, 2) if eta_h == eta_h else None,
        "errors": errors,
        "log": str(log_path),
        "ckpt_dir": str(ckpt_dir),
    }


def gpu_info():
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    procs = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
         "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    return {"gpus": r.stdout.strip(), "processes": procs.stdout.strip()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jobs-json", required=True, help='JSON list of {name,gpu,pid,log}')
    p.add_argument("--baseline-json", help="Metrics at t=0 for speed comparison")
    p.add_argument("--output", required=True)
    p.add_argument("--min-steps-per-sec", type=float, default=35.0)
    args = p.parse_args()

    jobs = json.loads(Path(args.jobs_json).read_text())
    baseline = json.loads(Path(args.baseline_json).read_text()) if args.baseline_json else {}

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "jobs": [],
        "gpu": gpu_info(),
        "disk": subprocess.run(["df", "-h", "/code"], capture_output=True, text=True).stdout.splitlines()[-1],
        "cpu_load": Path("/proc/loadavg").read_text().split()[:3] if Path("/proc/loadavg").exists() else [],
        "recommendation": "continue",
        "fallback_actions": [],
    }

    all_ok = True
    for job in jobs:
        m = parse_metrics(Path(job["log"]), int(job["pid"]))
        m["name"] = job["name"]
        m["gpu"] = job["gpu"]
        if job["name"] in baseline:
            b = baseline[job["name"]]
            delta_step = m["step"] - b.get("step", 0)
            delta_t = m["elapsed_s"] - b.get("elapsed_s", 0)
            m["interval_steps_per_sec"] = round(delta_step / delta_t, 2) if delta_t > 0 else 0
        slow = m.get("interval_steps_per_sec", m["steps_per_sec"]) < args.min_steps_per_sec and m["step"] < 50000
        if m["errors"] or not m["alive"] or slow:
            all_ok = False
        report["jobs"].append(m)

    # Compare shared GPU jobs
    by_gpu = {}
    for j in report["jobs"]:
        by_gpu.setdefault(j["gpu"], []).append(j)
    for gpu, js in by_gpu.items():
        if len(js) == 2:
            sps = [j.get("interval_steps_per_sec", j["steps_per_sec"]) for j in js]
            report[f"gpu{gpu}_shared_speed"] = {"job_a": js[0]["name"], "sps_a": sps[0],
                                                  "job_b": js[1]["name"], "sps_b": sps[1]}

    if not all_ok:
        report["recommendation"] = "fallback"
        # GPU1: stop later-started (walker2d) if both slow
        gpu1 = by_gpu.get(1, [])
        if len(gpu1) == 2:
            slower = min(gpu1, key=lambda x: x.get("interval_steps_per_sec", x["steps_per_sec"]))
            if slower.get("interval_steps_per_sec", slower["steps_per_sec"]) < 30:
                report["fallback_actions"].append({
                    "action": "stop",
                    "name": slower["name"],
                    "pid": slower["pid"],
                    "reason": "shared GPU steps/sec < 30",
                })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    return 0 if report["recommendation"] == "continue" else 2


if __name__ == "__main__":
    raise SystemExit(main())
