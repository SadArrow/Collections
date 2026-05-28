#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-${MYVLA_SERVER_BASE:-/home/nvme04/qianyupeng}}"
MYVLA_ROOT="${MYVLA_ROOT:-${BASE}/myVLA}"
AUTOWAIT_SCRIPT="${AUTOWAIT_SCRIPT:-${MYVLA_ROOT}/scripts/server/run_world_model_diffusion_vla_square_fold_isaac45_autowait.sh}"
RUNTIME_DIR="${RUNTIME_DIR:-${MYVLA_ROOT}/WorldModelDiffusionVlaRuntime}"
LOG_DIR="${LOG_DIR:-${RUNTIME_DIR}/logs}"
WATCHDOG_LOG="${WATCHDOG_LOG:-${LOG_DIR}/WorldModelDiffusionVlaSquareFold_watchdog.log}"
WATCHDOG_SLEEP_S="${WATCHDOG_SLEEP_S:-30}"

mkdir -p "${LOG_DIR}"

log() {
  echo "[$(date '+%F %T')] $1" | tee -a "${WATCHDOG_LOG}"
}

if [[ ! -x "${AUTOWAIT_SCRIPT}" ]]; then
  log "missing autowait script: ${AUTOWAIT_SCRIPT}"
  exit 2
fi

log "watchdog started pid=$$ autowait=${AUTOWAIT_SCRIPT}"

while true; do
  existing_pid="$(pgrep -f "${AUTOWAIT_SCRIPT}" | grep -v "^$$\$" | head -n 1 || true)"
  if [[ -n "${existing_pid}" ]]; then
    log "autowait already running pid=${existing_pid}; sleeping ${WATCHDOG_SLEEP_S}s"
    sleep "${WATCHDOG_SLEEP_S}"
    continue
  fi

  log "starting autowait"
  set +e
  "${AUTOWAIT_SCRIPT}" >> "${WATCHDOG_LOG}" 2>&1
  status=$?
  set -e
  log "autowait exited status=${status}; sleeping ${WATCHDOG_SLEEP_S}s before retry"
  sleep "${WATCHDOG_SLEEP_S}"
done
