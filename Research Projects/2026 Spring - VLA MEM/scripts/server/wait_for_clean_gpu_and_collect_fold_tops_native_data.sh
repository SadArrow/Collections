#!/usr/bin/env bash
set -euo pipefail

BASE="${MYVLA_SERVER_BASE:-/root/workspace/qianyupeng}"
GPU_LIST="${1:-0,1,2,3}"
INCREMENTAL_DEMOS="${2:-10}"
DEX_ROOT="${3:-${BASE}/DexGarmentLab-main}"
MYVLA_ROOT="${4:-${BASE}/myVLA}"
ISAAC_PY="${5:-${BASE}/isaac-sim-standalone@4.5.0/python.sh}"

FREE_MIN_MIB="${FREE_MIN_MIB:-32000}"
MAX_UTIL_PCT="${MAX_UTIL_PCT:-10}"
STABLE_POLLS="${STABLE_POLLS:-2}"
POLL_INTERVAL_S="${POLL_INTERVAL_S:-60}"

WRAPPER_PY="${BASE}/myVLA/scripts/server/run_myvla_envstandalone_wrapped.py"
TRAIN_DATA_DIR="${DEX_ROOT}/Data/Fold_Tops/train_data"
LOG_DIR="${BASE}/logs"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/fold_tops_native_collect_waiter.out}"

mkdir -p "${TRAIN_DATA_DIR}" "${LOG_DIR}"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${LOG_PATH}" >&2
}

count_demos() {
  find "${TRAIN_DATA_DIR}" -maxdepth 1 -type f -name 'data_*.npz' | wc -l
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

start_count="$(count_demos)"
target_count=$(( start_count + INCREMENTAL_DEMOS ))
candidate_gpu=""
stable_count=0

log "native Fold_Tops collection queued; start_count=${start_count} target_count=${target_count} gpu_list=[${GPU_LIST}]"

while true; do
  current_count="$(count_demos)"
  if (( current_count >= target_count )); then
    log "collection finished current_count=${current_count} target_count=${target_count}"
    exit 0
  fi

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
      log "launching one native Fold_Tops data-collection episode on gpu=${candidate_gpu}"
      if env \
        MYVLA_SERVER_BASE="${BASE}" \
        python3 "${WRAPPER_PY}" \
          --gpu "${candidate_gpu}" \
          --dex_root "${DEX_ROOT}" \
          --isaac_python "${ISAAC_PY}" \
          --runtime_dir "${MYVLA_ROOT}/WorldModelDiffusionVlaRuntime" \
          --script_rel "Env_StandAlone/Fold_Tops_Env.py" \
          --run_as_main \
          -- \
          --garment_random_flag True \
          --data_collection_flag True \
          --record_video_flag True \
          >> "${LOG_PATH}" 2>&1; then
        log "one collection episode finished successfully"
      else
        log "collection episode exited nonzero; will continue waiting and retry"
      fi
      candidate_gpu=""
      stable_count=0
    fi
  else
    candidate_gpu=""
    stable_count=0
    log "no clean gpu found in [${GPU_LIST}]"
  fi

  sleep "${POLL_INTERVAL_S}"
done
