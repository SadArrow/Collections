#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:?gpu id required}"
MODEL_LABEL="${2:?model label required}"
SEED="${3:?seed required}"
EPISODES="${4:?episodes required}"
EPISODE_INDICES="${5:-}"
EVAL_NAME_SUFFIX="${6:-}"
DEX_ROOT_ARG="${7:-}"
CHECKPOINT_DIR_ARG="${8:-}"
LAUNCHER_GPU_BINDING_MODE="${9:-${MYVLA_LAUNCHER_GPU_BINDING_MODE:-cuda_visible_devices}}"

if [[ "$EPISODE_INDICES" == "all" || "$EPISODE_INDICES" == "__all__" || "$EPISODE_INDICES" == "-" ]]; then
  EPISODE_INDICES=""
fi

BASE_DIR="${MYVLA_SERVER_BASE:-/root/workspace/qianyupeng}"
DEX_ROOT="${DEX_ROOT_ARG:-$BASE_DIR/DexGarmentLab-main}"
CHECKPOINT_DIR="${CHECKPOINT_DIR_ARG:-$BASE_DIR/artifacts/fold_tops_ll_pi05_train_full_20260415_v1/final}"
GOAL_TEXT="${MYVLA_FOLD_TOPS_GOAL_TEXT:-Use two robot arms to fold the shirt into a neat compact square. Start by visually aligning and flattening the garment on the table. Fold the left sleeve inward toward the center of the shirt, then fold the right sleeve inward toward the center, while keeping the cloth low and controlled. Next, grasp the lower hem, lift it only as much as needed, and fold the lower part of the shirt upward toward the center or upper body so the shirt becomes a compact rectangular or square block. Finish by gently pressing and aligning the folded shirt so the edges look tidy, symmetric, and stable. Use the current visual observation to decide the next local motion. If the shirt already appears neatly folded with the sleeves tucked in and the lower hem folded up, stop making large manipulation motions and only keep the folded shirt stable.}"
LOG_DIR="$BASE_DIR/logs"
mkdir -p "$LOG_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="${MODEL_LABEL}_gpu${GPU_ID}_seed${SEED}_${STAMP}"
LOG_PATH="$LOG_DIR/${RUN_TAG}.log"
PID_PATH="$LOG_DIR/${RUN_TAG}.pid"

CMD=(
  python
  "$BASE_DIR/myVLA/scripts/server/run_fold_tops_envstandalone_eval.py"
  --model_label "$MODEL_LABEL"
  --episodes "$EPISODES"
  --seed "$SEED"
  --gpu "$GPU_ID"
  --launcher_gpu_binding_mode "$LAUNCHER_GPU_BINDING_MODE"
  --dex_root "$DEX_ROOT"
  --myvla_root "$BASE_DIR/myVLA"
  --rpc_code_root "$BASE_DIR/myVLA"
  --entry_script_rel "tools/myvla_fold_tops_envstandalone_entry_ablation.py"
  --rpc_server_rel "isaac_sim/policy_rpc_server_ablation.py"
  --checkpoint_dir "$CHECKPOINT_DIR"
  --hl_vlm_dir ""
  --goal_text "$GOAL_TEXT"
  --results_root "$BASE_DIR/eval_results/fold_tops_envstandalone_eval"
  --keep_videos
)

if [[ -n "$EPISODE_INDICES" ]]; then
  CMD+=(--episode_indices "$EPISODE_INDICES")
fi

if [[ -n "$EVAL_NAME_SUFFIX" ]]; then
  CMD+=(--eval_name_suffix "$EVAL_NAME_SUFFIX")
fi

nohup "${CMD[@]}" >"$LOG_PATH" 2>&1 </dev/null &
PID="$!"
echo "$PID" >"$PID_PATH"

printf '{\n'
printf '  "pid": %s,\n' "$PID"
printf '  "gpu": %s,\n' "$GPU_ID"
printf '  "model_label": "%s",\n' "$MODEL_LABEL"
printf '  "seed": %s,\n' "$SEED"
printf '  "episodes": %s,\n' "$EPISODES"
printf '  "episode_indices": "%s",\n' "$EPISODE_INDICES"
printf '  "eval_name_suffix": "%s",\n' "$EVAL_NAME_SUFFIX"
printf '  "dex_root": "%s",\n' "$DEX_ROOT"
printf '  "checkpoint_dir": "%s",\n' "$CHECKPOINT_DIR"
printf '  "launcher_gpu_binding_mode": "%s",\n' "$LAUNCHER_GPU_BINDING_MODE"
printf '  "log_path": "%s",\n' "$LOG_PATH"
printf '  "pid_path": "%s"\n' "$PID_PATH"
printf '}\n'
