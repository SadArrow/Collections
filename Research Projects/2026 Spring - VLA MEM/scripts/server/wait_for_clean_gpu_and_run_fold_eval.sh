#!/usr/bin/env bash
set -euo pipefail

BASE="${MYVLA_SERVER_BASE:-/home/nvme04/qianyupeng}"
TARGET_GPU="${1:-5}"
MODEL_LABEL="${2:-oldarchs70guard_wait_v1}"
MYVLA_ROOT="${3:-${BASE}/myVLA_arch_backup_20260401_202638}"
RESULTS_ROOT="${4:-${BASE}/eval_results/fold_tops_halo}"

FREE_MIN_MIB="${FREE_MIN_MIB:-30000}"
MAX_UTIL_PCT="${MAX_UTIL_PCT:-10}"
STABLE_POLLS="${STABLE_POLLS:-2}"
POLL_INTERVAL_S="${POLL_INTERVAL_S:-60}"

EVAL_PY="${BASE}/myVLA/scripts/server/run_fold_tops_halo_eval.py"
LOG_PATH="${RESULTS_ROOT}/${MODEL_LABEL}_waiter.out"

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
    log "gpu=${TARGET_GPU} looks clean enough; launching fold eval model_label=${MODEL_LABEL}"
    exec env \
      MIN_FREE_MEM_MIB="${FREE_MIN_MIB}" \
      MAX_UTIL_PCT="${MAX_UTIL_PCT}" \
      FORCE_RESTART_RPC=1 \
      python3 "${EVAL_PY}" \
        --model_label "${MODEL_LABEL}" \
        --episodes 1 \
        --seed 0 \
        --myvla_root "${MYVLA_ROOT}" \
        --preferred_gpus "${TARGET_GPU}" \
        --wait_interval_s 5 \
        --idle_confirm_polls 1 \
        --keep_videos \
        --keep_step_artifacts \
        >> "${LOG_PATH}" 2>&1
  fi

  sleep "${POLL_INTERVAL_S}"
done
