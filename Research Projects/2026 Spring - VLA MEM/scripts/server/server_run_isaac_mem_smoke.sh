#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-/home/nvme04/qianyupeng/isaac-sim-standalone@5.1.0}"
HL_VLM_DIR="${HL_VLM_DIR:-${ROOT_DIR}/pretrained_vlm/google_paligemma-3b-mix-224-bfloat16}"
STATE_FILE="${ROOT_DIR}/isaac_sim_runtime/rpc_server_state_server.json"
LOG_DIR="${ROOT_DIR}/isaac_sim_runtime/server_logs"
VK_ICD_DIR="${ROOT_DIR}/isaac_sim_runtime/vulkan_icd"
mkdir -p "${LOG_DIR}" "${ROOT_DIR}/isaac_sim_runtime"
mkdir -p "${VK_ICD_DIR}"

cat > "${VK_ICD_DIR}/nvidia_egl_icd.json" <<'JSON'
{
  "file_format_version": "1.0.0",
  "ICD": {
    "library_path": "libEGL_nvidia.so.0",
    "api_version": "1.3.204"
  }
}
JSON

GPU_LIST="${MYVLA_GPU_LIST:-$(python3 "${ROOT_DIR}/scripts/server/select_free_gpus.py" --count 1)}"
if [[ -z "${GPU_LIST}" ]]; then
  echo "[run] no free GPUs detected"
  exit 1
fi
IFS=',' read -r SIM_GPU INFER_GPU <<< "${GPU_LIST}"
SIM_GPU="${SIM_GPU:-0}"
INFER_GPU="${INFER_GPU:-${SIM_GPU}}"

if [[ "${SIM_GPU}" == "${INFER_GPU}" ]]; then
  echo "[run] single-GPU mode: sim+policy+memory all on GPU ${SIM_GPU}"
else
  echo "[run] dual-GPU mode: sim on GPU ${SIM_GPU}, policy+memory on GPU ${INFER_GPU}"
fi

export MYVLA_DISABLE_TORCH_COMPILE=1
export VK_ICD_FILENAMES="${VK_ICD_DIR}/nvidia_egl_icd.json"

echo "[run] leaving CUDA_VISIBLE_DEVICES unset to avoid Isaac Sim Vulkan/GPU enumeration issues"
echo "[run] using VK_ICD_FILENAMES=${VK_ICD_FILENAMES}"

bash "${ISAAC_SIM_ROOT}/python.sh" "${ROOT_DIR}/isaac_sim/fold_shirt_dual_arm_mem_pi05.py" \
  --headless \
  --policy_mode rpc \
  --rpc_auto_start \
  --rpc_state_file "${STATE_FILE}" \
  --rpc_log_path "${LOG_DIR}/policy_rpc_server.log" \
  --rpc_python_exe "${ISAAC_SIM_ROOT}/python.sh" \
  --rpc_policy_device "cuda:${INFER_GPU}" \
  --rpc_hl_device "cuda:${INFER_GPU}" \
  --sim_device cuda \
  --gpu_rank "${SIM_GPU}" \
  --active_gpu "${SIM_GPU}" \
  --physics_gpu "${SIM_GPU}" \
  --world_size 1 \
  --mem_steps 2 \
  --execute_horizon 2 \
  --sim_steps_per_action 4 \
  --camera_warmup_steps 2 \
  --camera_read_attempts 6 \
  --task_adapter_residual_scale 0.10 \
  --checkpoint_dir "${ROOT_DIR}/pi05_droid_pytorch" \
  --hl_vlm_dir "${HL_VLM_DIR}" \
  --viz_name "server_smoke_$(date +%Y%m%d_%H%M%S)"
