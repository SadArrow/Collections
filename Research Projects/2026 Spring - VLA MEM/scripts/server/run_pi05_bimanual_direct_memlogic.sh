#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_DIR="${MYVLA_SERVER_BASE:-/root/workspace/qianyupeng}"
DEX_ROOT="${DEX_ROOT:-${BASE_DIR}/DexGarmentLab-main}"
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-${BASE_DIR}/isaac-sim-standalone@4.5.0}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ROOT_DIR}/pi05_droid_pytorch}"
RESULTS_DIR_REL="${RESULTS_DIR_REL:-../artifacts/pi05_bimanual_direct_memlogic}"
TAG="${TAG:-smoke60}"
ACTIVE_GPU="${ACTIVE_GPU:-0}"
DEVICE="${DEVICE:-cuda:${ACTIVE_GPU}}"
MAX_STEPS="${MAX_STEPS:-40}"
EVAL_EVERY="${EVAL_EVERY:-10}"
SCRIPT_REL="${SCRIPT_REL:-DexGarmentLab-main/tools/run_myvla_fold_tops_pi05_direct.py}"
LAUNCHER_PY="${ROOT_DIR}/scripts/server/run_myvla_envstandalone_wrapped.py"
LOG_DIR="${ROOT_DIR}/isaac_sim_runtime/server_logs"

mkdir -p "${LOG_DIR}" "${ROOT_DIR}/isaac_sim_runtime"

export MYVLA_DISABLE_TORCH_COMPILE=1

echo "[run] direct pi0.5 bimanual smoke via proven wrapped launcher"
echo "[run] DEX_ROOT=${DEX_ROOT}"
echo "[run] ISAAC_SIM_ROOT=${ISAAC_SIM_ROOT}"
echo "[run] CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "[run] RESULTS_DIR_REL=${RESULTS_DIR_REL}"
echo "[run] SCRIPT_REL=${SCRIPT_REL}"
echo "[run] LAUNCHER_PY=${LAUNCHER_PY}"

cd "${DEX_ROOT}"
bash "${ISAAC_SIM_ROOT}/python.sh" "${LAUNCHER_PY}" \
  --gpu "${ACTIVE_GPU}" \
  --gpu_binding_mode omniverse \
  --dex_root "${DEX_ROOT}" \
  --script_rel "${SCRIPT_REL}" \
  -- \
  --policy_kind dex_bimanual \
  --active_action_dim 60 \
  --model_action_dim 64 \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --results_dir_rel "${RESULTS_DIR_REL}" \
  --tag "${TAG}" \
  --device "${DEVICE}" \
  --active_gpu "${ACTIVE_GPU}" \
  --physics_gpu "${ACTIVE_GPU}" \
  --max_steps "${MAX_STEPS}" \
  --eval_every "${EVAL_EVERY}" \
  --validation_flag true
