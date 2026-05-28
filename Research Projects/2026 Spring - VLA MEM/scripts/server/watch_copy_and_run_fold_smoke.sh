#!/usr/bin/env bash
set -euo pipefail

BASE="${MYVLA_SERVER_BASE:-/root/workspace/qianyupeng}"
MODEL_LABEL="${1:-newarch_a6000_smoke}"
MYVLA_ROOT="${2:-${BASE}/myVLA}"
GPU_ID="${3:-0}"
RUNTIME_DIR="${4:-${BASE}/myVLA/WorldModelDiffusionVlaRuntime_smoke}"
CHECKPOINT_DIR="${5:-${BASE}/myVLA/pi05_droid_pytorch}"
TOKENIZER_MODEL="${6:-${BASE}/myVLA/assets/paligemma_tokenizer.model}"
HL_VLM_DIR="${7:-${BASE}/myVLA/pretrained_vlm/google_paligemma-3b-mix-224-bfloat16}"
LAUNCHER="${8:-${BASE}/myVLA/scripts/server/run_world_model_diffusion_vla_square_fold_isaac45_autowait.sh}"
RESULTS_ROOT="${9:-${BASE}/eval_results/fold_tops_halo}"

LOG_DIR="${BASE}/logs"
LOG_PATH="${LOG_DIR}/${MODEL_LABEL}_copy_waiter.log"
EVAL_PY="${BASE}/myVLA/scripts/server/run_fold_tops_halo_eval.py"
DEX_ROOT="${BASE}/DexGarmentLab-main"
ISAAC_PY="${BASE}/isaac-sim-standalone@4.5.0/python.sh"

mkdir -p "${LOG_DIR}" "${RESULTS_ROOT}"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${LOG_PATH}"
}

wait_for_required_files() {
  while true; do
    local missing=0
    [[ -x "${ISAAC_PY}" ]] || { log "waiting: missing ${ISAAC_PY}"; missing=1; }
    [[ -f "${DEX_ROOT}/tools/run_myvla_fold_tops_demo.py" ]] || { log "waiting: missing ${DEX_ROOT}/tools/run_myvla_fold_tops_demo.py"; missing=1; }
    [[ -f "${CHECKPOINT_DIR}/model.safetensors" ]] || { log "waiting: missing ${CHECKPOINT_DIR}/model.safetensors"; missing=1; }
    [[ -f "${TOKENIZER_MODEL}" ]] || { log "waiting: missing ${TOKENIZER_MODEL}"; missing=1; }
    [[ -f "${HL_VLM_DIR}/model-00001-of-00002.safetensors" ]] || { log "waiting: missing ${HL_VLM_DIR}/model-00001-of-00002.safetensors"; missing=1; }
    [[ -f "${HL_VLM_DIR}/model-00002-of-00002.safetensors" ]] || { log "waiting: missing ${HL_VLM_DIR}/model-00002-of-00002.safetensors"; missing=1; }
    [[ -f "${BASE}/vulkan_test/nvidia_abs_egl.json" ]] || { log "waiting: missing ${BASE}/vulkan_test/nvidia_abs_egl.json"; missing=1; }
    if [[ "${missing}" == "0" ]]; then
      return 0
    fi
    sleep 300
  done
}

log "watcher starting model_label=${MODEL_LABEL} gpu=${GPU_ID} myvla_root=${MYVLA_ROOT}"
wait_for_required_files
log "required files detected; launching Fold Tops smoke eval"

exec env MYVLA_SERVER_BASE="${BASE}" \
  python3 "${EVAL_PY}" \
    --model_label "${MODEL_LABEL}" \
    --episodes 1 \
    --seed 0 \
    --myvla_root "${MYVLA_ROOT}" \
    --launcher "${LAUNCHER}" \
    --runtime_dir "${RUNTIME_DIR}" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --tokenizer_model "${TOKENIZER_MODEL}" \
    --hl_vlm_dir "${HL_VLM_DIR}" \
    --preferred_gpus "${GPU_ID}" \
    --wait_interval_s 5 \
    --idle_confirm_polls 1 \
    --keep_videos \
    --keep_step_artifacts \
    --results_root "${RESULTS_ROOT}" \
    >> "${LOG_PATH}" 2>&1
