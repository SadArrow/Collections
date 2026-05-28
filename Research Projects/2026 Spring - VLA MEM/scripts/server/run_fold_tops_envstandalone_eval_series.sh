#!/usr/bin/env bash
set -euo pipefail

WAIT_PID="${1:-}"
CHECKPOINT_DIR="${2:?usage: run_fold_tops_envstandalone_eval_series.sh [wait_pid|-] <checkpoint_dir> [gpu_id] [model_label] [results_root]}"
GPU_ID="${3:-0}"
MODEL_LABEL="${4:-pi05_bimanual60_halo50}"
RESULTS_ROOT="${5:-}"

BASE_DIR="${MYVLA_SERVER_BASE:-/root/workspace/qianyupeng}"
DEX_ROOT="${BASE_DIR}/DexGarmentLab-main"
MYVLA_ROOT="${BASE_DIR}/myVLA"
RESULTS_ROOT="${RESULTS_ROOT:-${BASE_DIR}/eval_results/fold_tops_envstandalone_eval}"
EPISODES="${EPISODES:-50}"
SEEDS="${SEEDS:-0 1 2}"
LAUNCHER_GPU_BINDING_MODE="${LAUNCHER_GPU_BINDING_MODE:-omniverse}"
GOAL_TEXT="${MYVLA_FOLD_TOPS_GOAL_TEXT:-Use two robot arms to fold the shirt into a neat compact square. Start by visually aligning and flattening the garment on the table. Fold the left sleeve inward toward the center of the shirt, then fold the right sleeve inward toward the center, while keeping the cloth low and controlled. Next, grasp the lower hem, lift it only as much as needed, and fold the lower part of the shirt upward toward the center or upper body so the shirt becomes a compact rectangular or square block. Finish by gently pressing and aligning the folded shirt so the edges look tidy, symmetric, and stable. Use the current visual observation to decide the next local motion. If the shirt already appears neatly folded with the sleeves tucked in and the lower hem folded up, stop making large manipulation motions and only keep the folded shirt stable.}"

if [[ -n "${WAIT_PID}" && "${WAIT_PID}" != "-" ]]; then
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    sleep 30
  done
fi

for SEED in ${SEEDS}; do
  python "${MYVLA_ROOT}/scripts/server/run_fold_tops_envstandalone_eval.py" \
    --model_label "${MODEL_LABEL}" \
    --episodes "${EPISODES}" \
    --seed "${SEED}" \
    --gpu "${GPU_ID}" \
    --launcher_gpu_binding_mode "${LAUNCHER_GPU_BINDING_MODE}" \
    --dex_root "${DEX_ROOT}" \
    --myvla_root "${MYVLA_ROOT}" \
    --rpc_code_root "${MYVLA_ROOT}" \
    --entry_script_rel "tools/myvla_fold_tops_envstandalone_entry_ablation.py" \
    --rpc_server_rel "isaac_sim/policy_rpc_server_ablation.py" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --hl_vlm_dir "" \
    --goal_text "${GOAL_TEXT}" \
    --results_root "${RESULTS_ROOT}" \
    --keep_videos \
    --eval_name_suffix "seed${SEED}_gpu${GPU_ID}_series"
done
