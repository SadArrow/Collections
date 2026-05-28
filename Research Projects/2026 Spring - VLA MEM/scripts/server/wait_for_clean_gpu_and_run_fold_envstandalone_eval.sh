#!/usr/bin/env bash
set -euo pipefail

BASE="${MYVLA_SERVER_BASE:-/root/workspace/qianyupeng}"
GPU_LIST="${1:-0,1,2,3}"
MODEL_LABEL="${2:-newarch_envstandalone_queue_v1}"
MYVLA_ROOT="${3:-${BASE}/myVLA}"
RESULTS_ROOT="${4:-${BASE}/eval_results/fold_tops_envstandalone_eval}"
EPISODES="${5:-2}"
SEED="${6:-0}"
RPC_CODE_ROOT="${7:-${MYVLA_ROOT}}"

FREE_MIN_MIB="${FREE_MIN_MIB:-32000}"
MAX_UTIL_PCT="${MAX_UTIL_PCT:-10}"
STABLE_POLLS="${STABLE_POLLS:-2}"
POLL_INTERVAL_S="${POLL_INTERVAL_S:-60}"

EVAL_PY="${BASE}/myVLA/scripts/server/run_fold_tops_envstandalone_eval.py"
LOG_PATH="${RESULTS_ROOT}/${MODEL_LABEL}_waiter.out"

mkdir -p "${RESULTS_ROOT}"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${LOG_PATH}"
}

query_rows() {
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits \
    | awk -F, '
        {
          gsub(/ /, "", $1)
          gsub(/ /, "", $2)
          gsub(/ /, "", $3)
          gsub(/ /, "", $4)
          print $1 "," $2 "," $3 "," $4
        }
      '
}

pick_clean_gpu() {
  local rows="$1"
  local gpu_id
  local row
  local used_mib
  local total_mib
  local util_pct
  local free_mib

  IFS=, read -ra gpu_items <<< "${GPU_LIST}"
  for gpu_id in "${gpu_items[@]}"; do
    gpu_id="${gpu_id// /}"
    [[ -n "${gpu_id}" ]] || continue
    row="$(printf '%s\n' "${rows}" | awk -F, -v target="${gpu_id}" '$1 == target { print $0; exit }')"
    if [[ -z "${row}" ]]; then
      continue
    fi
    IFS=, read -r _idx used_mib total_mib util_pct <<< "${row}"
    free_mib=$(( total_mib - used_mib ))
    log "gpu=${gpu_id} used=${used_mib}MiB free=${free_mib}MiB util=${util_pct}%"
    if (( free_mib >= FREE_MIN_MIB && util_pct <= MAX_UTIL_PCT )); then
      printf '%s' "${gpu_id}"
      return 0
    fi
  done
  return 1
}

candidate_gpu=""
stable_count=0
log "waiting for any gpu in [${GPU_LIST}] free>=${FREE_MIN_MIB}MiB util<=${MAX_UTIL_PCT}% stable_polls=${STABLE_POLLS}"

while true; do
  rows="$(query_rows)"
  current_gpu=""
  if current_gpu="$(pick_clean_gpu "${rows}")"; then
    if [[ "${current_gpu}" == "${candidate_gpu}" ]]; then
      stable_count=$(( stable_count + 1 ))
    else
      candidate_gpu="${current_gpu}"
      stable_count=1
    fi
    log "candidate_gpu=${candidate_gpu} stable=${stable_count}/${STABLE_POLLS}"
    if (( stable_count >= STABLE_POLLS )); then
      log "launching envstandalone eval on gpu=${candidate_gpu} model_label=${MODEL_LABEL} episodes=${EPISODES} seed=${SEED}"
      exec env \
        MYVLA_SERVER_BASE="${BASE}" \
        python3 "${EVAL_PY}" \
          --model_label "${MODEL_LABEL}" \
          --episodes "${EPISODES}" \
          --seed "${SEED}" \
          --gpu "${candidate_gpu}" \
          --results_root "${RESULTS_ROOT}" \
          --myvla_root "${MYVLA_ROOT}" \
          --rpc_code_root "${RPC_CODE_ROOT}" \
          --keep_videos \
          --keep_step_artifacts \
          >> "${LOG_PATH}" 2>&1
    fi
  else
    candidate_gpu=""
    stable_count=0
    log "no clean gpu found in [${GPU_LIST}]"
  fi
  sleep "${POLL_INTERVAL_S}"
done
