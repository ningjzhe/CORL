#!/usr/bin/env python3
"""Update overnight_status.json fields."""
import argparse
import json
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--status-file", default="/code/analysis/iad_iql/results/overnight_status.json")
    parser.add_argument("--set", action="append", default=[], help="key=value")
    parser.add_argument("--append-error", default="")
    args = parser.parse_args()

    path = Path(args.status_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with open(path) as f:
            data = json.load(f)
    else:
        data = {
            "pipeline_start_time": datetime.now().isoformat(timespec="seconds"),
            "pipeline_end_time": None,
            "standard_status": "pending",
            "value_seed1_status": "pending",
            "compatibility_status": "pending",
            "ensemble_mean_status": "pending",
            "iad_0_5_status": "pending",
            "iad_1_0_status": "pending",
            "shuffled_status": "pending",
            "summary_status": "pending",
            "errors": [],
        }

    for item in args.set:
        key, _, val = item.partition("=")
        if val.lower() == "true":
            data[key] = True
        elif val.lower() == "false":
            data[key] = False
        elif val.startswith("[") or val.startswith("{"):
            data[key] = json.loads(val)
        else:
            data[key] = val

    if args.append_error:
        data.setdefault("errors", []).append(args.append_error)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()
