#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_BASE="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BASE="${BASE:-${MYVLA_SERVER_BASE:-${DEFAULT_BASE}}}"
ISAAC_ROOT="${ISAAC_ROOT:-${BASE}/isaac-sim-standalone@4.5.0}"
ISAAC_PY="${ISAAC_PY:-${ISAAC_ROOT}/python.sh}"
MYVLA_ROOT="${MYVLA_ROOT:-${BASE}/myVLA}"
DEX_ROOT="${DEX_ROOT:-${BASE}/DexGarmentLab-main}"
RUNTIME_DIR="${RUNTIME_DIR:-${MYVLA_ROOT}/WorldModelDiffusionVlaRuntime}"
LOG_DIR="${LOG_DIR:-${RUNTIME_DIR}/logs}"
VIZ_DIR="${VIZ_DIR:-${DEX_ROOT}/server_viz}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${MYVLA_ROOT}/pi05_droid_pytorch}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-${MYVLA_ROOT}/assets/paligemma_tokenizer.model}"
PYDEPS_DIR="${PYDEPS_DIR:-${RUNTIME_DIR}/pydeps}"
LOCAL_SYSROOT="${LOCAL_SYSROOT:-${RUNTIME_DIR}/syslibs}"
LOCAL_SYSLIB_DIR="${LOCAL_SYSROOT}/usr/lib/x86_64-linux-gnu"
ROLLOUT_EXTRA_SYSROOT="${ROLLOUT_EXTRA_SYSROOT:-${RUNTIME_DIR}/rollout_syslibs}"
ROLLOUT_EXTRA_SYSLIB_DIR="${ROLLOUT_EXTRA_SYSROOT}/usr/lib/x86_64-linux-gnu"
VK_ICD_JSON="${VK_ICD_JSON:-${BASE}/vulkan_test/nvidia_abs_egl.json}"
WORKSPACE_NVIDIA_GL_ROOT="${WORKSPACE_NVIDIA_GL_ROOT:-${BASE}/downloads/nvidia570/libnvidia-gl-570_570.86.10-0ubuntu1_amd64}"
WORKSPACE_NVIDIA_GL_LIB_DIR="${WORKSPACE_NVIDIA_GL_ROOT}/usr/lib/x86_64-linux-gnu"
WORKSPACE_NVIDIA_EGL_VENDOR_DIR="${WORKSPACE_NVIDIA_GL_ROOT}/usr/share/glvnd/egl_vendor.d"
WORKSPACE_NVIDIA_EGL_PLATFORM_DIR="${WORKSPACE_NVIDIA_GL_ROOT}/usr/share/egl/egl_external_platform.d"
WORKSPACE_NVIDIA_VK_ICD_JSON="${WORKSPACE_NVIDIA_GL_ROOT}/usr/share/vulkan/icd.d/nvidia_icd.json"
WORKSPACE_NVIDIA_VK_LAYER_DIR="${WORKSPACE_NVIDIA_GL_ROOT}/usr/share/vulkan/implicit_layer.d"
WORKSPACE_NVIDIA_VK_EGL_JSON="${WORKSPACE_NVIDIA_VK_EGL_JSON:-${BASE}/downloads/nvidia570/nvidia_egl_icd_570.86.10.json}"
WORKSPACE_X11_RUNTIME_ROOT="${WORKSPACE_X11_RUNTIME_ROOT:-${BASE}/downloads/x11_runtime_libs}"
WORKSPACE_X11_RUNTIME_LIB_DIR="${WORKSPACE_X11_RUNTIME_ROOT}/usr/lib/x86_64-linux-gnu"
WORKSPACE_LIBXT_ROOT="${WORKSPACE_LIBXT_ROOT:-${BASE}/downloads/libxt6}"
WORKSPACE_LIBXT_LIB_DIR="${WORKSPACE_LIBXT_ROOT}/usr/lib/x86_64-linux-gnu"
RUN_PREFIX="${RUN_PREFIX:-WorldModelDiffusionVlaSquareFoldSimulation}"
RUN_SUFFIX="${RUN_SUFFIX:-Isaac45SingleGpuWarm}"
WAIT_INTERVAL_S="${WAIT_INTERVAL_S:-60}"
IDLE_CONFIRM_POLLS="${IDLE_CONFIRM_POLLS:-3}"
MIN_FREE_MEM_MIB="${MIN_FREE_MEM_MIB:-0}"
MAX_UTIL_PCT="${MAX_UTIL_PCT:-5}"
PREFERRED_GPUS="${PREFERRED_GPUS:-}"
OUTER_STEPS="${OUTER_STEPS:-0}"
MIN_OUTER_STEPS="${MIN_OUTER_STEPS:-50}"
NUM_STEPS="${NUM_STEPS:-4}"
VIDEO_WINDOW="${VIDEO_WINDOW:-4}"
ROLLOUT_POLICY_MODE="${ROLLOUT_POLICY_MODE:-rpc}"
HL_VLM_DIR="${HL_VLM_DIR-${MYVLA_ROOT}/pretrained_vlm/google_paligemma-3b-mix-224-bfloat16}"
HL_DEVICE="${HL_DEVICE:-cuda:0}"
HL_DTYPE="${HL_DTYPE:-bfloat16}"
HL_MAX_NEW_TOKENS="${HL_MAX_NEW_TOKENS:-32}"
HL_TEMPERATURE="${HL_TEMPERATURE:-0.0}"
GOAL_TEXT="${GOAL_TEXT:-Fold the shirt into a compact square block by folding both sleeves inward and then folding the lower hem upward using two robot arms.}"
CAPTURE_SAMPLES="${CAPTURE_SAMPLES:-3}"
CAPTURE_RT_SUBFRAMES="${CAPTURE_RT_SUBFRAMES:-16}"
CAPTURE_MEDIAN_FILTER="${CAPTURE_MEDIAN_FILTER:-1}"
CAPTURE_NLM_STRENGTH="${CAPTURE_NLM_STRENGTH:-2.8}"
CAPTURE_SHARPEN_AMOUNT="${CAPTURE_SHARPEN_AMOUNT:-0.14}"
VIDEO_FPS="${VIDEO_FPS:-6}"
CLEAR_MEDIAN_FILTER="${CLEAR_MEDIAN_FILTER:-3}"
CLEAR_NLM_STRENGTH="${CLEAR_NLM_STRENGTH:-10.5}"
CLEAR_SHARPEN_AMOUNT="${CLEAR_SHARPEN_AMOUNT:-0.22}"
CLEAR_TEMPORAL_MIX="${CLEAR_TEMPORAL_MIX:-0.28}"
MOTION_SETTLE_STEPS="${MOTION_SETTLE_STEPS:-28}"
COMPLETION_PATIENCE="${COMPLETION_PATIENCE:-4}"
COMPLETION_DELTA_THRESHOLD="${COMPLETION_DELTA_THRESHOLD:-6.0}"
ENABLE_VIDEO_STABLE_FALLBACK="${ENABLE_VIDEO_STABLE_FALLBACK:-0}"
TARGET_JUMP_WARN_THRESHOLD="${TARGET_JUMP_WARN_THRESHOLD:-0.08}"
EXTRA_SETTLE_STEPS_ON_LARGE_JUMP="${EXTRA_SETTLE_STEPS_ON_LARGE_JUMP:-10}"
DEXGARMENTLAB_DENSE_SAMPLE_SCALE_FREE="${DEXGARMENTLAB_DENSE_SAMPLE_SCALE_FREE:-0.005}"
DEXGARMENTLAB_DENSE_SAMPLE_SCALE_ATTACH="${DEXGARMENTLAB_DENSE_SAMPLE_SCALE_ATTACH:-0.0035}"
DEXGARMENTLAB_ARM_MAX_JOINT_STEP="${DEXGARMENTLAB_ARM_MAX_JOINT_STEP:-0.08}"
DEXGARMENTLAB_HAND_MAX_JOINT_STEP="${DEXGARMENTLAB_HAND_MAX_JOINT_STEP:-0.12}"
DEXGARMENTLAB_ARM_MAX_JOINT_ACCEL_STEP="${DEXGARMENTLAB_ARM_MAX_JOINT_ACCEL_STEP:-0.06}"
DEXGARMENTLAB_HAND_MAX_JOINT_ACCEL_STEP="${DEXGARMENTLAB_HAND_MAX_JOINT_ACCEL_STEP:-0.12}"
DEXGARMENTLAB_JOINT_WORLD_STEPS="${DEXGARMENTLAB_JOINT_WORLD_STEPS:-2}"
DEXGARMENTLAB_HAND_SETTLE_STEPS="${DEXGARMENTLAB_HAND_SETTLE_STEPS:-10}"
DEXGARMENTLAB_MAX_JOINT_SMOOTH_SUBSTEPS="${DEXGARMENTLAB_MAX_JOINT_SMOOTH_SUBSTEPS:-24}"
DEXGARMENTLAB_NOOP_TASKSPACE_EPS="${DEXGARMENTLAB_NOOP_TASKSPACE_EPS:-0.002}"
DEXGARMENTLAB_NOOP_ORI_EPS_RAD="${DEXGARMENTLAB_NOOP_ORI_EPS_RAD:-0.015}"
DEXGARMENTLAB_MAX_ABS_JOINT_POSITION="${DEXGARMENTLAB_MAX_ABS_JOINT_POSITION:-20.0}"
DEXGARMENTLAB_IMPLAUSIBLE_JOINT_DELTA_FACTOR="${DEXGARMENTLAB_IMPLAUSIBLE_JOINT_DELTA_FACTOR:-10.0}"
ROLLOUT_HEADLESS_EXCLUDED_EXTENSIONS="${ROLLOUT_HEADLESS_EXCLUDED_EXTENSIONS:-}"
ROLLOUT_ENABLE_CLEAR_VIDEOS="${ROLLOUT_ENABLE_CLEAR_VIDEOS:-1}"
ROLLOUT_EXTRA_ARGS_FILE="${ROLLOUT_EXTRA_ARGS_FILE:-}"
FORCE_RESTART_RPC="${FORCE_RESTART_RPC:-1}"
ROLLOUT_TIMEOUT_S="${ROLLOUT_TIMEOUT_S:-21600}"

mkdir -p "${LOG_DIR}" "${VIZ_DIR}" "${RUNTIME_DIR}"
mkdir -p "${LOCAL_SYSLIB_DIR}"
mkdir -p "${ROLLOUT_EXTRA_SYSLIB_DIR}"

if [[ ! -e "${LOCAL_SYSLIB_DIR}/libcuda.so" ]]; then
  for candidate in \
    /usr/lib/x86_64-linux-gnu/libcuda.so \
    /usr/lib/x86_64-linux-gnu/libcuda.so.1 \
    /usr/lib/x86_64-linux-gnu/libcuda.so.*; do
    if [[ -e "${candidate}" ]] && [[ "${candidate}" != *"/stubs/"* ]]; then
      ln -sf "${candidate}" "${LOCAL_SYSLIB_DIR}/libcuda.so"
      break
    fi
  done
fi

if [[ ! -e "${LOCAL_SYSLIB_DIR}/libcuda.so.1" ]]; then
  if [[ -e /usr/lib/x86_64-linux-gnu/libcuda.so.1 ]]; then
    ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 "${LOCAL_SYSLIB_DIR}/libcuda.so.1"
  elif [[ -e "${LOCAL_SYSLIB_DIR}/libcuda.so" ]]; then
    ln -sf "${LOCAL_SYSLIB_DIR}/libcuda.so" "${LOCAL_SYSLIB_DIR}/libcuda.so.1"
  fi
fi

if [[ ! -x "${ISAAC_PY}" ]]; then
  echo "[launcher] missing Isaac python launcher: ${ISAAC_PY}" >&2
  exit 2
fi

build_isaac_extcache_library_path() {
  local extscache_root="${ISAAC_ROOT}/extscache"
  if [[ ! -d "${extscache_root}" ]]; then
    return 0
  fi
  python3 - <<'PY' "${extscache_root}"
from pathlib import Path
import sys

root = Path(sys.argv[1])
items = []
seen = set()
for pattern in ("*/bin", "*/lib", "*/bin/deps", "*/lib/deps"):
    for path in sorted(root.glob(pattern)):
        if not path.is_dir():
            continue
        text = str(path)
        if text in seen:
            continue
        seen.add(text)
        items.append(text)
print(":".join(items))
PY
}

ISAAC_EXTCACHE_LIBRARY_PATHS="$(build_isaac_extcache_library_path)"
BASE_LD_LIBRARY_PATH="${LOCAL_SYSLIB_DIR}:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
if [[ -d "${WORKSPACE_X11_RUNTIME_LIB_DIR}" ]]; then
  BASE_LD_LIBRARY_PATH="${WORKSPACE_X11_RUNTIME_LIB_DIR}:${BASE_LD_LIBRARY_PATH}"
fi
if [[ -d "${WORKSPACE_LIBXT_LIB_DIR}" ]]; then
  BASE_LD_LIBRARY_PATH="${WORKSPACE_LIBXT_LIB_DIR}:${BASE_LD_LIBRARY_PATH}"
fi
if [[ -d "${WORKSPACE_NVIDIA_GL_LIB_DIR}" ]]; then
  BASE_LD_LIBRARY_PATH="${WORKSPACE_NVIDIA_GL_LIB_DIR}:${BASE_LD_LIBRARY_PATH}"
  mkdir -p "$(dirname "${WORKSPACE_NVIDIA_VK_EGL_JSON}")"
  cat > "${WORKSPACE_NVIDIA_VK_EGL_JSON}" <<JSON
{
  "file_format_version": "1.0.1",
  "ICD": {
    "library_path": "${WORKSPACE_NVIDIA_GL_LIB_DIR}/libEGL_nvidia.so.0",
    "api_version": "1.4.303"
  }
}
JSON
  if [[ "${VK_ICD_JSON}" == "${BASE}/vulkan_test/nvidia_abs_egl.json" ]] && [[ -f "${WORKSPACE_NVIDIA_VK_EGL_JSON}" ]]; then
    VK_ICD_JSON="${WORKSPACE_NVIDIA_VK_EGL_JSON}"
  fi
fi
ROLLOUT_LD_LIBRARY_PATH="${ROLLOUT_EXTRA_SYSLIB_DIR}:${BASE_LD_LIBRARY_PATH}"
if [[ -n "${ISAAC_EXTCACHE_LIBRARY_PATHS}" ]]; then
  ROLLOUT_LD_LIBRARY_PATH="${ROLLOUT_LD_LIBRARY_PATH}:${ISAAC_EXTCACHE_LIBRARY_PATHS}"
fi

export PYTHONPATH="${PYDEPS_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export VK_ICD_FILENAMES="${VK_ICD_JSON}"
if [[ -d "${WORKSPACE_NVIDIA_EGL_VENDOR_DIR}" ]]; then
  export __EGL_VENDOR_LIBRARY_DIRS="${WORKSPACE_NVIDIA_EGL_VENDOR_DIR}${__EGL_VENDOR_LIBRARY_DIRS:+:${__EGL_VENDOR_LIBRARY_DIRS}}"
fi
if [[ -d "${WORKSPACE_NVIDIA_EGL_PLATFORM_DIR}" ]]; then
  export __EGL_EXTERNAL_PLATFORM_CONFIG_DIRS="${WORKSPACE_NVIDIA_EGL_PLATFORM_DIR}${__EGL_EXTERNAL_PLATFORM_CONFIG_DIRS:+:${__EGL_EXTERNAL_PLATFORM_CONFIG_DIRS}}"
fi
if [[ -d "${WORKSPACE_NVIDIA_VK_LAYER_DIR}" ]]; then
  export VK_LAYER_PATH="${WORKSPACE_NVIDIA_VK_LAYER_DIR}${VK_LAYER_PATH:+:${VK_LAYER_PATH}}"
fi
export LD_LIBRARY_PATH="${BASE_LD_LIBRARY_PATH}"
export DEXGARMENTLAB_DENSE_SAMPLE_SCALE_FREE
export DEXGARMENTLAB_DENSE_SAMPLE_SCALE_ATTACH
export DEXGARMENTLAB_ARM_MAX_JOINT_STEP
export DEXGARMENTLAB_HAND_MAX_JOINT_STEP
export DEXGARMENTLAB_ARM_MAX_JOINT_ACCEL_STEP
export DEXGARMENTLAB_HAND_MAX_JOINT_ACCEL_STEP
export DEXGARMENTLAB_JOINT_WORLD_STEPS
export DEXGARMENTLAB_HAND_SETTLE_STEPS
export DEXGARMENTLAB_MAX_JOINT_SMOOTH_SUBSTEPS
export DEXGARMENTLAB_NOOP_TASKSPACE_EPS
export DEXGARMENTLAB_NOOP_ORI_EPS_RAD
export DEXGARMENTLAB_MAX_ABS_JOINT_POSITION
export DEXGARMENTLAB_IMPLAUSIBLE_JOINT_DELTA_FACTOR

timestamp="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="${RUN_PREFIX}_${timestamp}_${RUN_SUFFIX}"
RUN_DIR="${VIZ_DIR}/${RUN_TAG}"
LAUNCH_LOG="${LOG_DIR}/${RUN_TAG}_launcher.log"
ROLLOUT_LOG="${LOG_DIR}/${RUN_TAG}_rollout.log"

log() {
  local message="$1"
  echo "[$(date '+%F %T')] ${message}" | tee -a "${LAUNCH_LOG}"
}

cleanup_launcher_lock() {
  flock -u 9 2>/dev/null || true
  exec 9>&- 2>/dev/null || true
}

LOCK_FILE="${RUNTIME_DIR}/square_fold_launcher.lock"
if ! command -v flock >/dev/null 2>&1; then
  log "missing required command: flock"
  exit 2
fi
exec 9>"${LOCK_FILE}"
if flock -n 9; then
  log "acquired square-fold launcher lock: ${LOCK_FILE}"
else
  log "waiting for square-fold launcher lock: ${LOCK_FILE}"
  flock 9
  log "acquired square-fold launcher lock after wait: ${LOCK_FILE}"
fi

gpu_wait_requirement_label() {
  if [[ "${MIN_FREE_MEM_MIB}" =~ ^[0-9]+$ ]] && (( MIN_FREE_MEM_MIB > 0 )); then
    printf '%s MiB' "${MIN_FREE_MEM_MIB}"
  else
    printf 'auto(>=85%% free, min 4 GiB)'
  fi
}

select_idle_gpu() {
  python3 - <<PY
import csv
import os
import subprocess
import sys

max_util = int(${MAX_UTIL_PCT})
min_free = int(${MIN_FREE_MEM_MIB})
preferred_raw = os.environ.get("PREFERRED_GPUS", "").strip()
preferred = {
    int(token.strip())
    for token in preferred_raw.split(",")
    if token.strip()
}
raw = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ],
    text=True,
)
candidates = []
for row in csv.reader(line for line in raw.strip().splitlines() if line.strip()):
    idx = int(row[0].strip())
    if preferred and idx not in preferred:
        continue
    used = int(row[1].strip())
    total = int(row[2].strip())
    util = int(row[3].strip())
    free = total - used
    required_free = min_free if min_free > 0 else max(4096, int(total * 0.85))
    if util <= max_util and free >= required_free:
        candidates.append((used, util, -free, idx))
if not candidates:
    sys.exit(1)
candidates.sort()
print(candidates[0][3])
PY
}

cleanup_old_square_fold_processes() {
  local keep_state_file="$1"
  local pattern="/isaac_sim/policy_rpc_server.py"
  ps -eo pid=,args= | while read -r pid args; do
    if [[ "${args}" == *"${pattern}"* ]] && [[ "${args}" == *"WorldModelDiffusionVlaSquareFold"* ]]; then
      if [[ -n "${keep_state_file}" ]] && [[ "${args}" == *"${keep_state_file}"* ]]; then
        continue
      fi
      log "stopping stale rpc pid=${pid}"
      kill "${pid}" >/dev/null 2>&1 || true
    fi
  done
  local rollout_pattern="${BASE}/DexGarmentLab-main/tools/run_myvla_fold_tops_demo.py"
  ps -eo pid=,args= | while read -r pid args; do
    if [[ "${args}" == *"${rollout_pattern}"* ]] && [[ "${args}" == *"WorldModelDiffusionVlaSquareFoldSimulation_"* ]]; then
      log "stopping stale rollout pid=${pid}"
      kill "${pid}" >/dev/null 2>&1 || true
    fi
  done
}

trap 'cleanup_launcher_lock; log "interrupted"; exit 130' INT TERM
trap cleanup_launcher_lock EXIT

log "run_dir=${RUN_DIR}"
log "rollout_log=${ROLLOUT_LOG}"

if [[ "${FORCE_RESTART_RPC}" == "1" ]]; then
  log "pre-wait cleanup of stale square-fold rpc/rollout processes"
  cleanup_old_square_fold_processes ""
fi

log "waiting for a single idle GPU (util<=${MAX_UTIL_PCT}, free_mem>=${GPU_WAIT_REQUIREMENT_LABEL:-$(gpu_wait_requirement_label)})"
candidate_gpu=""
candidate_count=0
while true; do
  if current_gpu="$(select_idle_gpu)"; then
    if [[ "${current_gpu}" == "${candidate_gpu}" ]]; then
      candidate_count=$((candidate_count + 1))
    else
      candidate_gpu="${current_gpu}"
      candidate_count=1
    fi
    log "idle_gpu_candidate=${candidate_gpu} stable_count=${candidate_count}/${IDLE_CONFIRM_POLLS}"
    if (( candidate_count >= IDLE_CONFIRM_POLLS )); then
      GPU="${candidate_gpu}"
      break
    fi
  else
    candidate_gpu=""
    candidate_count=0
    log "no eligible GPU yet"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | tee -a "${LAUNCH_LOG}"
  fi
  sleep "${WAIT_INTERVAL_S}"
done

RPC_STATE_FILE="${RUNTIME_DIR}/WorldModelDiffusionVlaSquareFoldWarmRpcIsaac45_gpu${GPU}.json"
RPC_LOG="${LOG_DIR}/WorldModelDiffusionVlaSquareFoldWarmRpcIsaac45_gpu${GPU}.log"
RPC_PORT=""
RPC_REUSED="false"

if [[ "${ROLLOUT_POLICY_MODE}" == "rpc" ]]; then
  cleanup_old_square_fold_processes "${RPC_STATE_FILE}"
  sleep 3
fi

START_WARM_EXTRA_ARGS=()
if [[ "${ROLLOUT_POLICY_MODE}" == "rpc" ]] && [[ "${FORCE_RESTART_RPC}" == "1" ]]; then
  cleanup_old_square_fold_processes ""
  START_WARM_EXTRA_ARGS+=(--force_restart)
fi

log "selected GPU=${GPU}"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | tee -a "${LAUNCH_LOG}"
if [[ "${ROLLOUT_LD_LIBRARY_PATH}" == *"${WORKSPACE_NVIDIA_GL_LIB_DIR}"* ]]; then
  log "rollout_env vk_icd=${VK_ICD_JSON} workspace_nvidia_ld=present"
else
  log "rollout_env vk_icd=${VK_ICD_JSON} workspace_nvidia_ld=missing"
fi

if [[ "${ROLLOUT_POLICY_MODE}" == "rpc" ]]; then
  START_WARM_CMD=(
    "${ISAAC_PY}" "${MYVLA_ROOT}/isaac_sim/start_warm_rpc_server.py"
    --host 127.0.0.1
    --port 0
    --timeout_s 10
    --start_timeout_s 900
    --python_exe "${ISAAC_PY}"
    --state_file "${RPC_STATE_FILE}"
    --checkpoint_dir "${CHECKPOINT_DIR}"
    --tokenizer_model "${TOKENIZER_MODEL}"
    --device cuda:0
    --video_window "${VIDEO_WINDOW}"
    --viz_dir "${MYVLA_ROOT}/isaac_sim_viz"
    --viz_name "${RUN_TAG}_rpc_gpu${GPU}"
  )
  if [[ -n "${HL_VLM_DIR}" ]]; then
    START_WARM_CMD+=(
      --hl_vlm_dir "${HL_VLM_DIR}"
      --hl_device "${HL_DEVICE}"
      --hl_dtype "${HL_DTYPE}"
      --hl_max_new_tokens "${HL_MAX_NEW_TOKENS}"
      --hl_temperature "${HL_TEMPERATURE}"
    )
  fi
  if (( ${#START_WARM_EXTRA_ARGS[@]} > 0 )); then
    START_WARM_CMD+=("${START_WARM_EXTRA_ARGS[@]}")
  fi
  START_WARM_CMD+=(--log_path "${RPC_LOG}")

  # Keep the rollout overlay pydeps away from the warm RPC server. The runtime
  # pydeps bundle currently carries a newer huggingface_hub that conflicts with
  # the Transformers version expected by myVLA/pi0.5. We intentionally sanitize
  # Python import env here instead of modifying the Isaac Sim installation.
  RPC_JSON="$(
    env -u PYTHONPATH \
      PYTHONNOUSERSITE=1 \
      CUDA_VISIBLE_DEVICES="${GPU}" \
      "${START_WARM_CMD[@]}"
  )"
  log "rpc_start_result=${RPC_JSON}"

  RPC_PORT="$(
    python3 - <<'PY' "${RPC_JSON}"
import json
import sys
payload = json.loads(sys.argv[1])
print(int(payload["port"]))
PY
  )"
  RPC_REUSED="$(
    python3 - <<'PY' "${RPC_JSON}"
import json
import sys
payload = json.loads(sys.argv[1])
print(str(bool(payload.get("reused", False))).lower())
PY
  )"
  log "rpc_ready port=${RPC_PORT} reused=${RPC_REUSED}"
else
  log "rpc_skipped policy_mode=${ROLLOUT_POLICY_MODE}"
fi

set +e
ROLLOUT_ACTIVE_GPU="${GPU}"
ROLLOUT_PHYSICS_GPU="${GPU}"
log "rollout_gpu_mapping visible=all active=${ROLLOUT_ACTIVE_GPU} physics=${ROLLOUT_PHYSICS_GPU}"
ROLLOUT_CMD=(
  timeout "${ROLLOUT_TIMEOUT_S}" "${ISAAC_PY}" "${DEX_ROOT}/tools/run_myvla_fold_tops_demo.py"
  --repo_root "${DEX_ROOT}"
  --headless
  --policy_mode "${ROLLOUT_POLICY_MODE}"
  --goal "${GOAL_TEXT}"
  --outer_steps "${OUTER_STEPS}"
  --min_outer_steps "${MIN_OUTER_STEPS}"
  --num_steps "${NUM_STEPS}"
  --action_scale_xy 0.018
  --action_scale_z 0.012
  --active_gpu "${ROLLOUT_ACTIVE_GPU}"
  --physics_gpu "${ROLLOUT_PHYSICS_GPU}"
  --vk_icd_json "${VK_ICD_JSON}"
  --camera_width 1024
  --camera_height 768
  --capture_samples "${CAPTURE_SAMPLES}"
  --capture_rt_subframes "${CAPTURE_RT_SUBFRAMES}"
  --capture_sample_settle_steps 0
  --capture_median_filter "${CAPTURE_MEDIAN_FILTER}"
  --capture_nlm_strength "${CAPTURE_NLM_STRENGTH}"
  --capture_sharpen_amount "${CAPTURE_SHARPEN_AMOUNT}"
  --video_fps "${VIDEO_FPS}"
  --clear_median_filter "${CLEAR_MEDIAN_FILTER}"
  --clear_nlm_strength "${CLEAR_NLM_STRENGTH}"
  --clear_sharpen_amount "${CLEAR_SHARPEN_AMOUNT}"
  --clear_temporal_mix "${CLEAR_TEMPORAL_MIX}"
  --motion_settle_steps "${MOTION_SETTLE_STEPS}"
  --completion_patience "${COMPLETION_PATIENCE}"
  --completion_delta_threshold "${COMPLETION_DELTA_THRESHOLD}"
  --target_jump_warn_threshold "${TARGET_JUMP_WARN_THRESHOLD}"
  --extra_settle_steps_on_large_jump "${EXTRA_SETTLE_STEPS_ON_LARGE_JUMP}"
  --viz_dir "${VIZ_DIR}"
  --viz_name "${RUN_TAG}"
)
if [[ "${ROLLOUT_POLICY_MODE}" == "rpc" ]]; then
  ROLLOUT_CMD+=(
    --rpc_host 127.0.0.1
    --rpc_port "${RPC_PORT}"
    --rpc_timeout_s 900
  )
fi
if [[ "${ENABLE_VIDEO_STABLE_FALLBACK}" == "1" ]]; then
  ROLLOUT_CMD+=(--enable_video_stable_fallback)
fi
if [[ "${ROLLOUT_ENABLE_CLEAR_VIDEOS}" == "1" ]]; then
  ROLLOUT_CMD+=(--export_clear_videos)
fi
if [[ -n "${ROLLOUT_EXTRA_ARGS_FILE}" ]]; then
  if [[ ! -f "${ROLLOUT_EXTRA_ARGS_FILE}" ]]; then
    echo "[launcher] missing rollout extra args file: ${ROLLOUT_EXTRA_ARGS_FILE}" >&2
    exit 3
  fi
  while IFS= read -r extra_arg || [[ -n "${extra_arg}" ]]; do
    if [[ -z "${extra_arg}" ]]; then
      continue
    fi
    ROLLOUT_CMD+=("${extra_arg}")
  done < "${ROLLOUT_EXTRA_ARGS_FILE}"
fi
# Keep the Isaac Sim rollout on the stock Isaac Python environment as well.
# The runtime pydeps overlay is useful for some auxiliary tools, but its NumPy
# bundle conflicts with Isaac Sim extensions (replicator / syntheticdata).
env -u CUDA_VISIBLE_DEVICES -u PYTHONPATH \
  PYTHONNOUSERSITE=1 \
  DEXGARMENTLAB_HEADLESS_EXCLUDED_EXTENSIONS="${ROLLOUT_HEADLESS_EXCLUDED_EXTENSIONS}" \
  LD_LIBRARY_PATH="${ROLLOUT_LD_LIBRARY_PATH}" \
  "${ROLLOUT_CMD[@]}" > "${ROLLOUT_LOG}" 2>&1
STATUS=$?
set -e

log "rollout_status=${STATUS}"
log "run_dir=${RUN_DIR}"
if [[ -f "${RUN_DIR}/summary.json" ]]; then
  echo "SUMMARY_JSON_BEGIN" | tee -a "${LAUNCH_LOG}"
  cat "${RUN_DIR}/summary.json" | tee -a "${LAUNCH_LOG}"
  echo "SUMMARY_JSON_END" | tee -a "${LAUNCH_LOG}"
fi
if [[ -f "${RUN_DIR}/run_status.json" ]]; then
  echo "RUN_STATUS_JSON_BEGIN" | tee -a "${LAUNCH_LOG}"
  cat "${RUN_DIR}/run_status.json" | tee -a "${LAUNCH_LOG}"
  echo "RUN_STATUS_JSON_END" | tee -a "${LAUNCH_LOG}"
fi

exit "${STATUS}"
