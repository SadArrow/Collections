#!/usr/bin/env bash
set -euo pipefail

BASE="${MYVLA_SERVER_BASE:-/root/workspace/qianyupeng}"
GPU_ID="${1:-1}"
RUN_NAME="${2:-fold_tops_ll_dexshadow_smoke}"
TARGET_SUCCESSES="${3:-2}"
MAX_ATTEMPTS="${4:-10}"
SEED="${5:-0}"
PROMPT_STYLE="${PROMPT_STYLE:-goal_only}"

ISAAC_PY="${BASE}/isaac-sim-standalone@4.5.0/python.sh"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SUPERVISOR_PY="${BASE}/myVLA/scripts/server/run_fold_tops_ll_dexshadow_collection_supervisor.py"
DEX_ROOT="${BASE}/DexGarmentLab-main"
OUT_DIR="${BASE}/artifacts/${RUN_NAME}"
LOG_PATH="${BASE}/artifacts/${RUN_NAME}.log"

mkdir -p "${OUT_DIR}"

echo "[launch] base=${BASE}" | tee "${LOG_PATH}"
echo "[launch] gpu=${GPU_ID} run_name=${RUN_NAME}" | tee -a "${LOG_PATH}"
echo "[launch] out_dir=${OUT_DIR}" | tee -a "${LOG_PATH}"
echo "[launch] prompt_style=${PROMPT_STYLE}" | tee -a "${LOG_PATH}"

exec "${PYTHON_BIN}" "${SUPERVISOR_PY}" \
  --gpu "${GPU_ID}" \
  --dex_root "${DEX_ROOT}" \
  --isaac_python "${ISAAC_PY}" \
  --output_dir "${OUT_DIR}" \
  --target_successes "${TARGET_SUCCESSES}" \
  --max_attempts "${MAX_ATTEMPTS}" \
  --seed "${SEED}" \
  --prompt_style "${PROMPT_STYLE}" \
  --record_video_flag \
  >> "${LOG_PATH}" 2>&1
