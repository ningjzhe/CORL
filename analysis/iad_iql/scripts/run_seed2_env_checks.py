#!/usr/bin/env python3
"""Run seed2 value quality + pair + triple checks for one environment."""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/code/analysis/iad_iql/scripts")
from seed2_env_config import ENVS, ckpt_path, SUPP  # noqa: E402

SCRIPT_DIR = Path("/code/analysis/iad_iql/scripts")


def parse_log_d4rl(log_path: Path):
    scores, returns = [], []
    if not log_path.exists():
        return float("nan"), float("nan")
    for line in log_path.read_text(errors="ignore").splitlines():
        if "D4RL score:" in line:
            try:
                scores.append(float(line.split("D4RL score:")[-1].strip()))
            except ValueError:
                pass
        if "Evaluation over" in line and "D4RL score:" in line:
            m = re.search(r"Evaluation over .*?:\s*([\d.]+)", line)
            if m:
                returns.append(float(m.group(1)))
    return (float(scores[-1]) if scores else float("nan"),
            float(returns[-1]) if returns else float("nan"))


def validate_seed2(env_key: str) -> dict:
    cfg = ENVS[env_key]
    ckpt = Path(ckpt_path(env_key, 2))
    log = Path(cfg["log_dir"]) / "value_seed2.log"
    status = "completed"
    if not ckpt.exists():
        return {"env": cfg["env_name"], "status": "failed", "reason": "missing checkpoint_999999.pt"}

    sd = torch.load(ckpt, map_location="cpu")
    for k in ("qf", "vf", "actor"):
        if k not in sd:
            return {"env": cfg["env_name"], "status": "failed", "reason": f"missing {k}"}
        for tk, tv in sd[k].items():
            t = tv if hasattr(tv, "isnan") else torch.tensor(tv)
            if torch.isnan(t).any() or torch.isinf(t).any():
                return {"env": cfg["env_name"], "status": "failed", "reason": f"NaN/Inf in {k}"}

    if log.exists() and "free(): invalid pointer" in log.read_text(errors="ignore"):
        status = "completed_with_cleanup_error"

    last_step = 0
    for line in log.read_text(errors="ignore").splitlines():
        m = re.match(r"Time steps:\s*(\d+)", line.strip())
        if m:
            last_step = int(m.group(1))
    if last_step != 1000000:
        return {"env": cfg["env_name"], "status": "failed", "reason": f"last_step={last_step}"}

    d4rl, ret = parse_log_d4rl(log)
    return {
        "env": cfg["env_name"],
        "seed2_final_D4RL": d4rl,
        "seed2_final_return": ret,
        "checkpoint_path": str(ckpt),
        "status": status,
        "cleanup_error": status == "completed_with_cleanup_error",
        "nan_inf": False,
    }


def run_cmd(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
    return r.returncode == 0, r.stdout


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env-key", required=True, choices=list(ENVS.keys()))
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    cfg = ENVS[args.env_key]
    out_dir = SUPP / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    quality = validate_seed2(args.env_key)
    quality_path = out_dir / f"{args.env_key}_seed2_value_quality.json"
    with open(quality_path, "w") as f:
        json.dump(quality, f, indent=2)
    print(f"Wrote {quality_path}")
    if quality["status"].startswith("failed"):
        sys.exit(1)

    pairs = [
        ("pair01", 0, 1),
        ("pair02", 0, 2),
        ("pair12", 1, 2),
    ]
    pair_results = {}
    for name, si, sj in pairs:
        out = out_dir / f"{args.env_key}_{name}_compat.json"
        ok, _ = run_cmd([
            sys.executable, str(SCRIPT_DIR / "seed2_pair_compatibility.py"),
            "--env", cfg["env_name"],
            "--checkpoint-i", ckpt_path(args.env_key, si),
            "--checkpoint-j", ckpt_path(args.env_key, sj),
            "--seed-i", str(si), "--seed-j", str(sj),
            "--norm-stats", cfg["norm_stats"],
            "--batch-indices", cfg["batch42"],
            "--output", str(out),
            "--device", args.device,
        ])
        pair_results[name] = json.loads(out.read_text()) if out.exists() else {"compatible": False}

    combined_pairs = {
        "env": cfg["env_name"],
        "pair01": pair_results["pair01"],
        "pair02": pair_results["pair02"],
        "pair12": pair_results["pair12"],
    }
    pair_path = out_dir / f"{args.env_key}_pair_compatibility.json"
    with open(pair_path, "w") as f:
        json.dump(combined_pairs, f, indent=2)
    print(f"Wrote {pair_path}")

    triple_out = out_dir / f"{args.env_key}_triple_value_statistics.json"
    triple_ok, _ = run_cmd([
        sys.executable, str(SCRIPT_DIR / "seed2_triple_statistics.py"),
        "--env", cfg["env_name"],
        "--checkpoint-a", ckpt_path(args.env_key, 0),
        "--checkpoint-b", ckpt_path(args.env_key, 1),
        "--checkpoint-c", ckpt_path(args.env_key, 2),
        "--norm-stats", cfg["norm_stats"],
        "--batch-indices", cfg["batch42"],
        "--output", str(triple_out),
        "--device", args.device,
    ])
    triple = json.loads(triple_out.read_text()) if triple_out.exists() else {}

    gate = {
        "env": cfg["env_name"],
        "seed2_ok": quality["status"] in ("completed", "completed_with_cleanup_error"),
        "pair02_compatible": pair_results["pair02"].get("compatible", False),
        "pair12_compatible": pair_results["pair12"].get("compatible", False),
        "triple_forward_ok": triple.get("forward_pass_ok", False),
        "allow_3q_actor": (
            quality["status"] in ("completed", "completed_with_cleanup_error")
            and pair_results["pair02"].get("compatible", False)
            and pair_results["pair12"].get("compatible", False)
            and triple.get("forward_pass_ok", False)
        ),
    }
    gate_path = out_dir / f"{args.env_key}_3q_gate.json"
    with open(gate_path, "w") as f:
        json.dump(gate, f, indent=2)
    print(json.dumps(gate, indent=2))
    sys.exit(0)


if __name__ == "__main__":
    main()
