#!/usr/bin/env bash
set -euo pipefail

BASE="${MYVLA_SERVER_BASE:-/root/workspace/qianyupeng}"
WAIT_PID="${1:?usage: queue_ll_pi05_train_after_pid.sh <wait_pid> <manifest> <output_dir> [gpu_id] [poll_interval_s]}"
MANIFEST_PATH="${2:?usage: queue_ll_pi05_train_after_pid.sh <wait_pid> <manifest> <output_dir> [gpu_id] [poll_interval_s]}"
OUTPUT_DIR="${3:?usage: queue_ll_pi05_train_after_pid.sh <wait_pid> <manifest> <output_dir> [gpu_id] [poll_interval_s]}"
GPU_ID="${4:-2}"
POLL_INTERVAL_S="${5:-60}"

ISAAC_PY="${BASE}/isaac-sim-standalone@4.5.0/python.sh"
TRAIN_PY="${BASE}/myVLA/scripts/train/train_fold_tops_ll_pi05_sft.py"
CHECKPOINT_DIR="${BASE}/myVLA/pi05_droid_pytorch"
LOG_DIR="${BASE}/logs"
QUEUE_LOG="${LOG_DIR}/fold_tops_ll_pi05_train_queue.out"
TRAIN_LOG="${LOG_DIR}/fold_tops_ll_pi05_train.out"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${QUEUE_LOG}"
}

while kill -0 "${WAIT_PID}" 2>/dev/null; do
  log "waiting for low-level expert collection pid=${WAIT_PID}"
  sleep "${POLL_INTERVAL_S}"
done

if [[ ! -f "${MANIFEST_PATH}" ]]; then
  log "manifest missing: ${MANIFEST_PATH}"
  exit 1
fi

count="$(find "$(dirname "${MANIFEST_PATH}")" -maxdepth 2 -name 'episode_ll_expert.npz' | wc -l)"
log "expert collection finished; saved_successes=${count}"
if (( count < 1 )); then
  log "no low-level expert episodes available; skipping training"
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
exec "${ISAAC_PY}" "${TRAIN_PY}" \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --manifest "${MANIFEST_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --epochs 1 \
  --train_batch_size 2 \
  --eval_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --logging_steps 20 \
  --eval_steps 100 \
  --save_steps 100 \
  --train_scope expert \
  --gradient_checkpointing \
  --bf16 \
  >> "${TRAIN_LOG}" 2>&1
