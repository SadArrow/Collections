#!/usr/bin/env bash
set -euo pipefail

BASE="${MYVLA_SERVER_BASE:-/root/workspace/qianyupeng}"
GPU_LIST="${1:-0,2,3}"
TARGET_SUCCESSES="${2:-10}"
MAX_ATTEMPTS="${3:-60}"
OUTPUT_ROOT="${4:-${BASE}/artifacts/fold_tops_ll_expert_queued_$(date +%Y%m%d_%H%M%S)}"
SEED="${5:-0}"
DEX_ROOT="${6:-${BASE}/DexGarmentLab-main}"
MYVLA_ROOT="${7:-${BASE}/myVLA}"
ISAAC_PY="${8:-${BASE}/isaac-sim-standalone@4.5.0/python.sh}"

FREE_MIN_MIB="${FREE_MIN_MIB:-32000}"
MAX_UTIL_PCT="${MAX_UTIL_PCT:-10}"
STABLE_POLLS="${STABLE_POLLS:-2}"
POLL_INTERVAL_S="${POLL_INTERVAL_S:-60}"
ATTEMPT_TIMEOUT_S="${ATTEMPT_TIMEOUT_S:-7200}"
RECORD_VIDEO_FLAG="${RECORD_VIDEO_FLAG:-1}"

SUPERVISOR_PY="${BASE}/myVLA/scripts/server/run_fold_tops_ll_expert_collection_supervisor.py"
LOG_DIR="${BASE}/logs"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/fold_tops_ll_expert_waiter.out}"

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${LOG_PATH}" >&2
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

log "queued low-level expert collection; gpu_list=[${GPU_LIST}] target_successes=${TARGET_SUCCESSES} max_attempts=${MAX_ATTEMPTS}"

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
      run_dir="${OUTPUT_ROOT}/gpu${candidate_gpu}"
      run_log="${LOG_DIR}/fold_tops_ll_expert_gpu${candidate_gpu}_queued.out"
      cmd=(
        python3 "${SUPERVISOR_PY}"
        --gpu "${candidate_gpu}"
        --output_dir "${run_dir}"
        --target_successes "${TARGET_SUCCESSES}"
        --max_attempts "${MAX_ATTEMPTS}"
        --seed "$(( SEED + candidate_gpu * 1000 ))"
        --dex_root "${DEX_ROOT}"
        --myvla_root "${MYVLA_ROOT}"
        --isaac_python "${ISAAC_PY}"
        --attempt_timeout_s "${ATTEMPT_TIMEOUT_S}"
      )
      if [[ "${RECORD_VIDEO_FLAG}" == "1" ]]; then
        cmd+=(--record_video_flag)
      fi
      log "launching supervisor on gpu=${candidate_gpu} output_dir=${run_dir}"
      nohup "${cmd[@]}" >"${run_log}" 2>&1 &
      child_pid=$!
      log "supervisor pid=${child_pid} run_log=${run_log}"
      exit 0
    fi
  else
    candidate_gpu=""
    stable_count=0
    log "no clean gpu found in [${GPU_LIST}]"
  fi

  sleep "${POLL_INTERVAL_S}"
done
