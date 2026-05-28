#!/usr/bin/env bash
set -euo pipefail

BASE="${MYVLA_SERVER_BASE:-/home/nvme04/qianyupeng}"
TARGET_GPU="${1:-5}"
LABEL="${2:-halo14_official_smoke}"
TASKS="${3:-all}"
EPISODES_PER_TASK="${4:-1}"
RESULTS_ROOT="${5:-${BASE}/eval_results/halo14_official}"

FREE_MIN_MIB="${FREE_MIN_MIB:-30000}"
MAX_UTIL_PCT="${MAX_UTIL_PCT:-10}"
STABLE_POLLS="${STABLE_POLLS:-2}"
POLL_INTERVAL_S="${POLL_INTERVAL_S:-60}"

RUN_PY="${BASE}/myVLA/scripts/server/run_halo14_official_eval.py"
LOG_PATH="${RESULTS_ROOT}/${LABEL}_waiter.out"

mkdir -p "${RESULTS_ROOT}"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${LOG_PATH}"
}

read_gpu_row() {
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits \
    | awk -F, -v target="${TARGET_GPU}" '
        {
          gsub(/ /, "", $1)
          gsub(/ /, "", $2)
          gsub(/ /, "", $3)
          gsub(/ /, "", $4)
          if ($1 == target) {
            print $2 "," $3 "," $4
          }
        }
      '
}

stable_count=0
log "waiting for gpu=${TARGET_GPU} free>=${FREE_MIN_MIB}MiB util<=${MAX_UTIL_PCT}% stable_polls=${STABLE_POLLS}"

while true; do
  row="$(read_gpu_row)"
  if [[ -z "${row}" ]]; then
    stable_count=0
    log "gpu=${TARGET_GPU} not found in nvidia-smi output"
    sleep "${POLL_INTERVAL_S}"
    continue
  fi

  IFS=, read -r used_mib total_mib util_pct <<< "${row}"
  free_mib=$(( total_mib - used_mib ))

  if (( free_mib >= FREE_MIN_MIB && util_pct <= MAX_UTIL_PCT )); then
    stable_count=$(( stable_count + 1 ))
  else
    stable_count=0
  fi

  log "gpu=${TARGET_GPU} used=${used_mib}MiB free=${free_mib}MiB util=${util_pct}% stable=${stable_count}/${STABLE_POLLS}"

  if (( stable_count >= STABLE_POLLS )); then
    log "gpu=${TARGET_GPU} looks clean enough; launching official halo14 eval label=${LABEL} tasks=${TASKS}"
    exec env \
      CUDA_VISIBLE_DEVICES="${TARGET_GPU}" \
      python3 "${RUN_PY}" \
        --label "${LABEL}" \
        --tasks "${TASKS}" \
        --episodes_per_task "${EPISODES_PER_TASK}" \
        --results_root "${RESULTS_ROOT}" \
        >> "${LOG_PATH}" 2>&1
  fi

  sleep "${POLL_INTERVAL_S}"
done
