#!/usr/bin/env bash
set -euo pipefail

BASE="${MYVLA_SERVER_BASE:-/root/workspace/qianyupeng}"
WAIT_PIDS_RAW="${1:-}"
RUN_DURATION="${2:-3h}"
GPU_LIST_RAW="${3:-0,2,3}"
INCREMENTAL_DEMOS="${4:-999}"
DEX_ROOT="${5:-${BASE}/DexGarmentLab-main}"
MYVLA_ROOT="${6:-${BASE}/myVLA}"
ISAAC_PY="${7:-${BASE}/isaac-sim-standalone@4.5.0/python.sh}"
POLL_INTERVAL_S="${POLL_INTERVAL_S:-60}"

COLLECT_SH="${BASE}/myVLA/scripts/server/wait_for_clean_gpu_and_collect_fold_tops_native_data.sh"
LOG_DIR="${BASE}/logs"
QUEUE_LOG="${LOG_DIR}/fold_collect_queued_after_pids.out"

mkdir -p "${LOG_DIR}"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${QUEUE_LOG}"
}

normalize_gpu_items() {
  local raw="$1"
  local item
  IFS=, read -ra gpu_items <<< "${raw}"
  for item in "${gpu_items[@]}"; do
    item="${item// /}"
    [[ -n "${item}" ]] || continue
    printf '%s\n' "${item}"
  done
}

wait_for_pids() {
  local pid
  local alive_count
  if [[ -z "${WAIT_PIDS_RAW// /}" ]]; then
    log "no wait pids provided; launch will begin immediately"
    return 0
  fi

  log "waiting for prior pids to exit: ${WAIT_PIDS_RAW}"
  while true; do
    alive_count=0
    for pid in ${WAIT_PIDS_RAW}; do
      if kill -0 "${pid}" 2>/dev/null; then
        alive_count=$(( alive_count + 1 ))
      fi
    done
    if (( alive_count == 0 )); then
      log "all prior pids finished; starting queued collection jobs"
      return 0
    fi
    log "still waiting; alive_count=${alive_count}"
    sleep "${POLL_INTERVAL_S}"
  done
}

launch_jobs() {
  local gpu
  local launch_log
  local run_log
  local child_pid
  while IFS= read -r gpu; do
    [[ -n "${gpu}" ]] || continue
    launch_log="${LOG_DIR}/fold_collect_${RUN_DURATION}_gpu${gpu}.launch"
    run_log="${LOG_DIR}/fold_collect_${RUN_DURATION}_gpu${gpu}.out"
    nohup env \
      LOG_PATH="${run_log}" \
      timeout -k 120 "${RUN_DURATION}" \
      bash "${COLLECT_SH}" "${gpu}" "${INCREMENTAL_DEMOS}" "${DEX_ROOT}" "${MYVLA_ROOT}" "${ISAAC_PY}" \
      >"${launch_log}" 2>&1 &
    child_pid=$!
    log "launched queued collection on gpu=${gpu} pid=${child_pid} duration=${RUN_DURATION} target_increment=${INCREMENTAL_DEMOS}"
  done < <(normalize_gpu_items "${GPU_LIST_RAW}")
}

wait_for_pids
launch_jobs
