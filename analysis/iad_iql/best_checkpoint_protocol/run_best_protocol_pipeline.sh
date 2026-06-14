#!/usr/bin/env bash
# Best-checkpoint protocol: 16 actor-only runs (Hopper then HalfCheetah)
set -euo pipefail

ROOT="/code/analysis/iad_iql/best_checkpoint_protocol"
SCRIPT="/code/analysis/iad_iql/scripts/train_frozen_iad_actor_best.py"
LOG="${ROOT}/logs/best_protocol_pipeline.log"

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4 TORCH_NUM_THREADS=4 TORCH_NUM_INTEROP_THREADS=1
export MUJOCO_GL=osmesa D4RL_SUPPRESS_IMPORT_ERROR=1 WANDB_MODE=offline

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate iql

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG}"; }

run_one() {
  local gpu=$1 env_key=$2 run_name=$3 variant=$4 rep=$5 actor_seed=$6 batch_seed=$7
  local init=$8 batches=$9
  local project="${ROOT}/${env_key}"
  local ckpt_a ckpt_b value_dir norm

  if [[ "${env_key}" == "hopper_medium_v2" ]]; then
    value_dir="/code/analysis/iad_iql/hopper_medium_v2/checkpoints"
    ckpt_a="${value_dir}/value_seed0/checkpoint_999999.pt"
    ckpt_b="${value_dir}/value_seed1/checkpoint_999999.pt"
    norm="${project}/checkpoints/norm_stats.npz"
    env_name="hopper-medium-v2"
  else
    value_dir="/code/analysis/iad_iql/checkpoints"
    ckpt_a="${value_dir}/value_seed0/checkpoint_999999.pt"
    ckpt_b="${value_dir}/value_seed1/checkpoint_999999.pt"
    norm="${project}/checkpoints/norm_stats.npz"
    env_name="halfcheetah-medium-v2"
  fi

  if [[ -f "${project}/results/${run_name}_summary.json" ]]; then
    log "SKIP ${run_name} (summary exists)"
    return 0
  fi

  log "START ${run_name} GPU=${gpu} variant=${variant} rep=${rep}"
  CUDA_VISIBLE_DEVICES="${gpu}" python "${SCRIPT}" \
    --variant "${variant}" \
    --env "${env_name}" \
    --run-name "${run_name}" \
    --replicate "${rep}" \
    --seed 0 \
    --actor-seed "${actor_seed}" \
    --batch-seed "${batch_seed}" \
    --max-updates 100000 \
    --eval-freq 5000 \
    --batch-size 256 \
    --n-eval-episodes 10 \
    --best-eval-episodes 50 \
    --final-eval-episodes 50 \
    --checkpoint-a "${ckpt_a}" \
    --checkpoint-b "${ckpt_b}" \
    --checkpoints-dir "${value_dir}" \
    --norm-stats "${norm}" \
    --shared-actor-init "${init}" \
    --batch-indices "${batches}" \
    --output-dir "${project}/checkpoints" \
    --log-dir "${project}/logs" \
    --results-dir "${project}/results" \
    --device cuda \
    2>&1 | tee -a "${project}/logs/${run_name}.stdout.log"
  log "DONE ${run_name}"
}

run_round() {
  local env_key=$1 gpu0_run=$2 gpu0_var=$3 gpu0_rep=$4 gpu0_as=$5 gpu0_bs=$6 gpu0_init=$7 gpu0_batch=$8
  local gpu1_run=$9 gpu1_var=${10} gpu1_rep=${11} gpu1_as=${12} gpu1_bs=${13} gpu1_init=${14} gpu1_batch=${15}
  run_one 0 "${env_key}" "${gpu0_run}" "${gpu0_var}" "${gpu0_rep}" "${gpu0_as}" "${gpu0_bs}" "${gpu0_init}" "${gpu0_batch}" &
  local p0=$!
  sleep 90
  run_one 1 "${env_key}" "${gpu1_run}" "${gpu1_var}" "${gpu1_rep}" "${gpu1_as}" "${gpu1_bs}" "${gpu1_init}" "${gpu1_batch}" &
  local p1=$!
  wait "${p0}"
  wait "${p1}"
}

HOPPER="${ROOT}/hopper_medium_v2/checkpoints"
HC="${ROOT}/halfcheetah_medium_v2/checkpoints"

log "=== Best-checkpoint protocol pipeline start ==="

# Hopper Rep A/B (8 runs)
run_round hopper_medium_v2 hopper_repA_standard_iql standard_iql A 12345 42 \
  "${HOPPER}/shared_actor_init_seed12345.pt" "${HOPPER}/actor_training_batch_indices_seed42.npy" \
  hopper_repA_ensemble_mean ensemble_mean A 12345 42 \
  "${HOPPER}/shared_actor_init_seed12345.pt" "${HOPPER}/actor_training_batch_indices_seed42.npy"

run_round hopper_medium_v2 hopper_repA_iad_lambda_1.0 iad_lambda_1.0 A 12345 42 \
  "${HOPPER}/shared_actor_init_seed12345.pt" "${HOPPER}/actor_training_batch_indices_seed42.npy" \
  hopper_repA_shuffled_iad_lambda_1.0 shuffled_iad_lambda_1.0 A 12345 42 \
  "${HOPPER}/shared_actor_init_seed12345.pt" "${HOPPER}/actor_training_batch_indices_seed42.npy"

run_round hopper_medium_v2 hopper_repB_standard_iql standard_iql B 54321 43 \
  "${HOPPER}/shared_actor_init_seed54321.pt" "${HOPPER}/actor_training_batch_indices_seed43.npy" \
  hopper_repB_ensemble_mean ensemble_mean B 54321 43 \
  "${HOPPER}/shared_actor_init_seed54321.pt" "${HOPPER}/actor_training_batch_indices_seed43.npy"

run_round hopper_medium_v2 hopper_repB_iad_lambda_1.0 iad_lambda_1.0 B 54321 43 \
  "${HOPPER}/shared_actor_init_seed54321.pt" "${HOPPER}/actor_training_batch_indices_seed43.npy" \
  hopper_repB_shuffled_iad_lambda_1.0 shuffled_iad_lambda_1.0 B 54321 43 \
  "${HOPPER}/shared_actor_init_seed54321.pt" "${HOPPER}/actor_training_batch_indices_seed43.npy"

# HalfCheetah Rep A/B (8 runs)
run_round halfcheetah_medium_v2 hc_repA_standard_iql standard_iql A 12345 42 \
  "${HC}/shared_actor_init_seed12345.pt" "${HC}/actor_training_batch_indices_seed42.npy" \
  hc_repA_ensemble_mean ensemble_mean A 12345 42 \
  "${HC}/shared_actor_init_seed12345.pt" "${HC}/actor_training_batch_indices_seed42.npy"

run_round halfcheetah_medium_v2 hc_repA_iad_lambda_1.0 iad_lambda_1.0 A 12345 42 \
  "${HC}/shared_actor_init_seed12345.pt" "${HC}/actor_training_batch_indices_seed42.npy" \
  hc_repA_shuffled_iad_lambda_1.0 shuffled_iad_lambda_1.0 A 12345 42 \
  "${HC}/shared_actor_init_seed12345.pt" "${HC}/actor_training_batch_indices_seed42.npy"

run_round halfcheetah_medium_v2 hc_repB_standard_iql standard_iql B 54321 43 \
  "${HC}/shared_actor_init_seed54321.pt" "${HC}/actor_training_batch_indices_seed43.npy" \
  hc_repB_ensemble_mean ensemble_mean B 54321 43 \
  "${HC}/shared_actor_init_seed54321.pt" "${HC}/actor_training_batch_indices_seed43.npy"

run_round halfcheetah_medium_v2 hc_repB_iad_lambda_1.0 iad_lambda_1.0 B 54321 43 \
  "${HC}/shared_actor_init_seed54321.pt" "${HC}/actor_training_batch_indices_seed43.npy" \
  hc_repB_shuffled_iad_lambda_1.0 shuffled_iad_lambda_1.0 B 54321 43 \
  "${HC}/shared_actor_init_seed54321.pt" "${HC}/actor_training_batch_indices_seed43.npy"

log "=== All 16 runs complete; running summarize ==="
python /code/analysis/iad_iql/scripts/summarize_best_checkpoint_protocol.py
python /code/analysis/iad_iql/scripts/generate_best_protocol_figures.py
python /code/analysis/iad_iql/scripts/generate_best_protocol_report.py
log "=== Pipeline FINISHED ==="
