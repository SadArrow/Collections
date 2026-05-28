#!/usr/bin/env bash
set -euo pipefail

BASE="${MYVLA_SERVER_BASE:-/root/workspace/qianyupeng}"
TARGET_MIN_COUNT="${1:-10}"
DEX_ROOT="${2:-${BASE}/DexGarmentLab-main}"
OUTPUT_DIR="${3:-${BASE}/eval_results/fold_tops_hl_sft_dataset}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
SEQUENCE_LEN="${SEQUENCE_LEN:-4}"
POLL_INTERVAL_S="${POLL_INTERVAL_S:-60}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

PREPARE_PY="${BASE}/myVLA/scripts/data/prepare_fold_tops_hl_sft_dataset.py"
TRAIN_DATA_DIR="${DEX_ROOT}/Data/Fold_Tops/train_data"
LOG_DIR="${BASE}/logs"
LOG_PATH="${LOG_DIR}/fold_tops_hl_sft_waiter.out"

mkdir -p "${TRAIN_DATA_DIR}" "${OUTPUT_DIR}" "${LOG_DIR}"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${LOG_PATH}"
}

count_demos() {
  find "${TRAIN_DATA_DIR}" -maxdepth 1 -type f -name 'data_*.npz' | wc -l
}

log "waiting for Fold_Tops native train_data count >= ${TARGET_MIN_COUNT}"

while true; do
  current_count="$(count_demos)"
  log "current_train_data_count=${current_count}"
  if (( current_count >= TARGET_MIN_COUNT )); then
    log "enough native data found; preparing high-level SFT dataset into ${OUTPUT_DIR}"
    exec "${PYTHON_BIN}" "${PREPARE_PY}" \
      --dex_root "${DEX_ROOT}" \
      --task_name Fold_Tops \
      --output_dir "${OUTPUT_DIR}" \
      --sequence_len "${SEQUENCE_LEN}" \
      --image_size "${IMAGE_SIZE}" \
      --limit 0 \
      >> "${LOG_PATH}" 2>&1
  fi
  sleep "${POLL_INTERVAL_S}"
done
