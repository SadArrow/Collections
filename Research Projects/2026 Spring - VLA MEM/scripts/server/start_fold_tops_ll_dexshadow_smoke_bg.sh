#!/usr/bin/env bash
set -euo pipefail

BASE="${MYVLA_SERVER_BASE:-/root/workspace/qianyupeng}"
GPU_ID="${1:-1}"
RUN_NAME="${2:-fold_tops_ll_dexshadow_smoke}"
TARGET_SUCCESSES="${3:-2}"
MAX_ATTEMPTS="${4:-10}"
SEED="${5:-0}"

mkdir -p "${BASE}/artifacts"
pkill -f collect_fold_tops_ll_dexshadow_dataset.py || true

nohup bash "${BASE}/myVLA/scripts/server/launch_fold_tops_ll_dexshadow_smoke.sh" \
  "${GPU_ID}" \
  "${RUN_NAME}" \
  "${TARGET_SUCCESSES}" \
  "${MAX_ATTEMPTS}" \
  "${SEED}" \
  > "${BASE}/artifacts/${RUN_NAME}.launch.log" 2>&1 < /dev/null &

echo "started:${RUN_NAME}:gpu=${GPU_ID}"
