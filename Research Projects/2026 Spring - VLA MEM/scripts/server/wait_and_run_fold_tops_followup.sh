#!/usr/bin/env bash
set -euo pipefail

BASE="${MYVLA_SERVER_BASE:-/home/nvme04/qianyupeng}"
OLD_EVAL_DIR="${1:?usage: wait_and_run_fold_tops_followup.sh <old_eval_dir> <new_model_label> <new_myvla_root> [results_root] [poll_interval_s] }"
NEW_MODEL_LABEL="${2:?usage: wait_and_run_fold_tops_followup.sh <old_eval_dir> <new_model_label> <new_myvla_root> [results_root] [poll_interval_s] }"
NEW_MYVLA_ROOT="${3:?usage: wait_and_run_fold_tops_followup.sh <old_eval_dir> <new_model_label> <new_myvla_root> [results_root] [poll_interval_s] }"
RESULTS_ROOT="${4:-${BASE}/eval_results/fold_tops_halo}"
POLL_INTERVAL_S="${5:-60}"

EVAL_SCRIPT="${BASE}/myVLA/scripts/server/run_fold_tops_halo_eval.py"
COMPARE_SCRIPT="${BASE}/myVLA/scripts/server/compare_fold_tops_halo_eval.py"
FOLLOWUP_LOG="${RESULTS_ROOT}/${NEW_MODEL_LABEL}_followup.out"
COMPARE_OUT_DIR="${RESULTS_ROOT}/compare_$(basename "${OLD_EVAL_DIR}")_vs_${NEW_MODEL_LABEL}"

OLD_SUMMARY="${OLD_EVAL_DIR}/summary.json"
PLAN_PATH="${OLD_EVAL_DIR}/episode_plan.json"

mkdir -p "${RESULTS_ROOT}"

{
  echo "[followup] old_eval_dir=${OLD_EVAL_DIR}"
  echo "[followup] new_model_label=${NEW_MODEL_LABEL}"
  echo "[followup] new_myvla_root=${NEW_MYVLA_ROOT}"
  echo "[followup] waiting_for=${OLD_SUMMARY}"
} >> "${FOLLOWUP_LOG}"

while [[ ! -f "${OLD_SUMMARY}" ]]; do
  echo "[followup] $(date '+%F %T') waiting for old summary" >> "${FOLLOWUP_LOG}"
  sleep "${POLL_INTERVAL_S}"
done

python3 "${EVAL_SCRIPT}" \
  --model_label "${NEW_MODEL_LABEL}" \
  --episodes 1 \
  --seed 0 \
  --plan_path "${PLAN_PATH}" \
  --results_root "${RESULTS_ROOT}" \
  --myvla_root "${NEW_MYVLA_ROOT}" \
  --keep_videos \
  --keep_step_artifacts \
  >> "${FOLLOWUP_LOG}" 2>&1

NEW_EVAL_DIR="$(
  python3 - <<'PY' "${RESULTS_ROOT}" "${NEW_MODEL_LABEL}"
from pathlib import Path
import sys

root = Path(sys.argv[1])
label = sys.argv[2]
candidates = sorted(path for path in root.glob(f"{label}_*") if path.is_dir())
print(candidates[-1] if candidates else "")
PY
)"

if [[ -z "${NEW_EVAL_DIR}" || ! -f "${NEW_EVAL_DIR}/summary.json" ]]; then
  echo "[followup] failed to resolve new eval dir for ${NEW_MODEL_LABEL}" >> "${FOLLOWUP_LOG}"
  exit 2
fi

python3 "${COMPARE_SCRIPT}" \
  --baseline_summary "${OLD_SUMMARY}" \
  --candidate_summary "${NEW_EVAL_DIR}/summary.json" \
  --output_dir "${COMPARE_OUT_DIR}" \
  >> "${FOLLOWUP_LOG}" 2>&1

echo "[followup] completed new_eval_dir=${NEW_EVAL_DIR}" >> "${FOLLOWUP_LOG}"
