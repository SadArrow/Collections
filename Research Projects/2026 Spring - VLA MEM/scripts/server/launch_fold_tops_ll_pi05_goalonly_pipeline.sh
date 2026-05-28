#!/usr/bin/env bash
set -euo pipefail

BASE="${MYVLA_SERVER_BASE:-/root/workspace/qianyupeng}"
GPU_ID="${1:-0}"
RUN_NAME="${2:-pi05_goalonly_retrain_$(date +%Y%m%d_%H%M%S)}"
TARGET_SUCCESSES="${3:-100}"
MAX_ATTEMPTS="${4:-400}"
SEED="${5:-2200}"
EXPERT_EPOCHS="${6:-2}"
FULL_EPOCHS="${7:-1}"

ISAAC_PY="${BASE}/isaac-sim-standalone@4.5.0/python.sh"
HOST_PYTHON_BIN="${PYTHON_BIN:-python3}"
QUEUE_PYTHON_BIN="${QUEUE_PYTHON_BIN:-${ISAAC_PY}}"
COLLECT_SUPERVISOR="${BASE}/myVLA/scripts/server/run_fold_tops_nativeexpert_collection_supervisor.py"
PREPARE_PY="${BASE}/myVLA/scripts/data/prepare_fold_tops_ll_pi05_from_native_dex.py"
TRAIN_PY="${BASE}/myVLA/scripts/train/train_fold_tops_ll_pi05_dexshadow_sft.py"
CHECKPOINT_DIR="${BASE}/myVLA/pi05_droid_pytorch"
DEX_ROOT="${BASE}/DexGarmentLab-main"
MYVLA_ROOT="${BASE}/myVLA"
RUN_ROOT="${BASE}/artifacts/${RUN_NAME}"
COLLECT_DIR="${RUN_ROOT}/collect"
CONVERT_DIR="${RUN_ROOT}/converted"
TRAIN_STAGE1_DIR="${RUN_ROOT}/train_stage1_expert"
TRAIN_STAGE2_DIR="${RUN_ROOT}/train_stage2_full"
LOG_DIR="${RUN_ROOT}/logs"
PIPELINE_STATE="${RUN_ROOT}/pipeline_state.json"
QUEUE_SCRIPT="${RUN_ROOT}/run_train_queue.sh"
COLLECT_LOG="${LOG_DIR}/collect.log"
QUEUE_LOG="${LOG_DIR}/train_queue.log"
LAUNCH_LOG="${LOG_DIR}/launch.log"

mkdir -p "${COLLECT_DIR}" "${CONVERT_DIR}" "${TRAIN_STAGE1_DIR}" "${TRAIN_STAGE2_DIR}" "${LOG_DIR}"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${LAUNCH_LOG}"
}

write_state() {
  cat > "${PIPELINE_STATE}" <<EOF
{
  "run_name": "${RUN_NAME}",
  "gpu": ${GPU_ID},
  "target_successes": ${TARGET_SUCCESSES},
  "max_attempts": ${MAX_ATTEMPTS},
  "seed": ${SEED},
  "expert_epochs": ${EXPERT_EPOCHS},
  "full_epochs": ${FULL_EPOCHS},
  "collection_mode": "nativeexpert",
  "collect_dir": "${COLLECT_DIR}",
  "convert_dir": "${CONVERT_DIR}",
  "train_stage1_dir": "${TRAIN_STAGE1_DIR}",
  "train_stage2_dir": "${TRAIN_STAGE2_DIR}",
  "collect_log": "${COLLECT_LOG}",
  "queue_log": "${QUEUE_LOG}",
  "collect_pid": ${COLLECT_PID:-0},
  "queue_pid": ${QUEUE_PID:-0}
}
EOF
}

log "base=${BASE}"
log "run_name=${RUN_NAME}"
log "gpu=${GPU_ID}"
log "collect_dir=${COLLECT_DIR}"
log "convert_dir=${CONVERT_DIR}"
log "train_stage1_dir=${TRAIN_STAGE1_DIR}"
log "train_stage2_dir=${TRAIN_STAGE2_DIR}"

nohup "${HOST_PYTHON_BIN}" "${COLLECT_SUPERVISOR}" \
  --gpu "${GPU_ID}" \
  --dex_root "${DEX_ROOT}" \
  --myvla_root "${MYVLA_ROOT}" \
  --isaac_python "${ISAAC_PY}" \
  --output_dir "${COLLECT_DIR}" \
  --target_successes "${TARGET_SUCCESSES}" \
  --max_attempts "${MAX_ATTEMPTS}" \
  --record_video_flag \
  > "${COLLECT_LOG}" 2>&1 &
COLLECT_PID=$!
log "started collect pid=${COLLECT_PID}"

cat > "${QUEUE_SCRIPT}" <<EOF
#!/usr/bin/env bash
set -euo pipefail

COLLECT_PID="${COLLECT_PID}"
COLLECT_DIR="${COLLECT_DIR}"
MANIFEST_PATH="${COLLECT_DIR}/manifest.jsonl"
CONVERT_DIR="${CONVERT_DIR}"
PREPARE_PY="${PREPARE_PY}"
PYTHON_BIN="${QUEUE_PYTHON_BIN}"
BASE="${BASE}"
TRAIN_PY="${TRAIN_PY}"
CHECKPOINT_DIR="${CHECKPOINT_DIR}"
TRAIN_STAGE1_DIR="${TRAIN_STAGE1_DIR}"
TRAIN_STAGE2_DIR="${TRAIN_STAGE2_DIR}"
EXPERT_EPOCHS="${EXPERT_EPOCHS}"
FULL_EPOCHS="${FULL_EPOCHS}"

while kill -0 "\${COLLECT_PID}" 2>/dev/null; do
  sleep 60
done

if [[ ! -f "\${MANIFEST_PATH}" ]]; then
  echo "[queue] manifest missing: \${MANIFEST_PATH}"
  exit 1
fi

SUCCESS_COUNT=\$("\${PYTHON_BIN}" - "\${MANIFEST_PATH}" <<'PY'
import json
import sys
count = 0
with open(sys.argv[1], "r", encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        copied_dataset = str(item.get("copied_dataset", "")).strip()
        copied_train_files = [str(path).strip() for path in item.get("copied_train_files", []) if str(path).strip()]
        dataset_path = str(item.get("dataset_path", "")).strip()
        has_dataset = bool(dataset_path or copied_dataset or copied_train_files)
        if bool(item.get("success", False)) and has_dataset:
            count += 1
print(count)
PY
)
echo "[queue] collected_successes=\${SUCCESS_COUNT}"
if [[ "\${SUCCESS_COUNT}" -lt 1 ]]; then
  echo "[queue] no successful episodes found; aborting training"
  exit 1
fi

cd "\${BASE}/myVLA"
export PYTHONPATH="\${BASE}/myVLA"

"\${PYTHON_BIN}" "\${PREPARE_PY}" \
  --artifact_dirs "\${COLLECT_DIR}" \
  --output_dir "\${CONVERT_DIR}" \
  --prompt_style goal_only

CONVERT_MANIFEST="\${CONVERT_DIR}/manifest.jsonl"
if [[ ! -f "\${CONVERT_MANIFEST}" ]]; then
  echo "[queue] converted manifest missing: \${CONVERT_MANIFEST}"
  exit 1
fi

"\${PYTHON_BIN}" "\${TRAIN_PY}" \
  --checkpoint_dir "\${CHECKPOINT_DIR}" \
  --manifest "\${CONVERT_MANIFEST}" \
  --output_dir "\${TRAIN_STAGE1_DIR}" \
  --epochs "\${EXPERT_EPOCHS}" \
  --train_batch_size 2 \
  --eval_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-5 \
  --train_scope expert \
  --active_action_dim 60 \
  --model_action_dim 64 \
  --target_mode delta \
  --split_unit episode \
  --bf16 \
  --gradient_checkpointing

STAGE1_FINAL="\${TRAIN_STAGE1_DIR}/final"
if [[ ! -f "\${STAGE1_FINAL}/model.safetensors" ]]; then
  echo "[queue] stage1 final checkpoint missing: \${STAGE1_FINAL}"
  exit 1
fi

"\${PYTHON_BIN}" "\${TRAIN_PY}" \
  --checkpoint_dir "\${STAGE1_FINAL}" \
  --manifest "\${CONVERT_MANIFEST}" \
  --output_dir "\${TRAIN_STAGE2_DIR}" \
  --epochs "\${FULL_EPOCHS}" \
  --train_batch_size 2 \
  --eval_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-5 \
  --train_scope full \
  --active_action_dim 60 \
  --model_action_dim 64 \
  --target_mode delta \
  --split_unit episode \
  --bf16 \
  --gradient_checkpointing
EOF

chmod +x "${QUEUE_SCRIPT}"

nohup bash "${QUEUE_SCRIPT}" > "${QUEUE_LOG}" 2>&1 &
QUEUE_PID=$!
log "started queue pid=${QUEUE_PID}"

write_state

cat <<EOF
{
  "ok": true,
  "run_name": "${RUN_NAME}",
  "gpu": ${GPU_ID},
  "collect_pid": ${COLLECT_PID},
  "queue_pid": ${QUEUE_PID},
  "collect_dir": "${COLLECT_DIR}",
  "train_stage1_dir": "${TRAIN_STAGE1_DIR}",
  "train_stage2_dir": "${TRAIN_STAGE2_DIR}",
  "collect_log": "${COLLECT_LOG}",
  "queue_log": "${QUEUE_LOG}",
  "pipeline_state": "${PIPELINE_STATE}"
}
EOF
