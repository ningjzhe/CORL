#!/bin/bash
# =============================================================================
# HPC Launch Script — Offline RL Benchmark (4× A800-80GB)
# 45 experiments: 5 algorithms × 3 datasets × 3 seeds
#
# Each GPU runs 6 experiments concurrently (configurable via env or --workers-per-gpu N).
# Estimated wall time: ~8-12 hours for all 45 experiments.
#
# Usage:
#   ./run_all.sh                         # All experiments (6 workers/GPU)
#   WORKERS_PER_GPU=4 ./run_all.sh       # 4 concurrent per GPU
#   ./run_all.sh --dry-run               # Preview only
#   ./run_all.sh --baseline-only         # 27 baseline experiments
#   ./run_all.sh --variants-only         # 18 UA/UGH experiments
#   ./run_all.sh --algos CQL,IQL         # Specific algorithms
# =============================================================================

set -e

# ── Configuration ──
GPUS=${GPUS:-"0,1,2,3"}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-8}
OUTPUT_DIR=${OUTPUT_DIR:-"/code/results"}

# CUDA optimizations
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TF32_ENABLE=1
export CUDA_LAUNCH_BLOCKING=0

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "Offline RL Benchmark — 4× A800-80GB"
echo "============================================================"
echo "GPUs:              $GPUS"
echo "Workers per GPU:   $WORKERS_PER_GPU"
echo "Output:            $OUTPUT_DIR"
echo "PyTorch:           $(python3 -c 'import torch; print(torch.__version__)')"
echo "GPU count:         $(python3 -c 'import torch; print(torch.cuda.device_count())')"
echo "GPU 0 mem:         $(python3 -c 'import torch; print(f\"{torch.cuda.get_device_properties(0).total_mem/1024**3:.0f}GB\")' 2>/dev/null || echo 'N/A')"
echo "============================================================"
echo ""

# ── Build python command ──
PY_CMD="python3 launch.py --gpus $GPUS --output_dir $OUTPUT_DIR --workers-per-gpu $WORKERS_PER_GPU"

# Parse special flags; pass unknown ones through to Python
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            PY_CMD="$PY_CMD --dry-run"
            shift ;;
        --baseline-only)
            PY_CMD="$PY_CMD --algos CQL,IQL,DT"
            shift ;;
        --variants-only)
            PY_CMD="$PY_CMD --algos UA_CQL,UGH_CQL"
            shift ;;
        --algos)
            PY_CMD="$PY_CMD --algos $2"
            shift 2 ;;
        --datasets)
            PY_CMD="$PY_CMD --datasets $2"
            shift 2 ;;
        --seeds)
            PY_CMD="$PY_CMD --seeds $2"
            shift 2 ;;
        --workers-per-gpu)
            PY_CMD="$PY_CMD --workers-per-gpu $2"
            shift 2 ;;
        --gpus)
            PY_CMD="$PY_CMD --gpus $2"
            shift 2 ;;
        *)
            echo "Unknown flag: $1"
            shift ;;
    esac
done

# ── Launch ──
echo "Running: $PY_CMD"
echo ""
$PY_CMD

echo ""
echo "Done. Results in: $OUTPUT_DIR"
echo "Monitor: watch -n 30 'cat $OUTPUT_DIR/progress.json 2>/dev/null || echo waiting...'"
