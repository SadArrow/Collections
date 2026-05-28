#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_BASE = Path(os.environ.get("MYVLA_SERVER_BASE", "/home/nvme04/qianyupeng"))
DEFAULT_RESULTS_ROOT = DEFAULT_BASE / "eval_results" / "fold_tops_halo"
DEFAULT_MYVLA_ROOT = DEFAULT_BASE / "myVLA"
DEFAULT_RUNTIME_DIR = DEFAULT_MYVLA_ROOT / "WorldModelDiffusionVlaRuntime"
DEFAULT_CHECKPOINT_DIR = DEFAULT_MYVLA_ROOT / "pi05_droid_pytorch"
DEFAULT_TOKENIZER_MODEL = DEFAULT_MYVLA_ROOT / "assets" / "paligemma_tokenizer.model"
DEFAULT_HL_VLM_DIR = DEFAULT_MYVLA_ROOT / "pretrained_vlm" / "google_paligemma-3b-mix-224-bfloat16"
DEFAULT_EVAL_SCRIPT = DEFAULT_MYVLA_ROOT / "scripts" / "server" / "run_fold_tops_halo_eval.py"
DEFAULT_COMPARE_SCRIPT = DEFAULT_MYVLA_ROOT / "scripts" / "server" / "compare_fold_tops_halo_eval.py"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wait for an old-model Fold Tops eval, then run the new model and compare.")
    parser.add_argument("--old_eval_dir", required=True)
    parser.add_argument("--plan_path", required=True)
    parser.add_argument("--results_root", default=os.fspath(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--new_model_label", default="pcmbdebug")
    parser.add_argument("--myvla_root", default=os.fspath(DEFAULT_MYVLA_ROOT))
    parser.add_argument("--runtime_dir", default=os.fspath(DEFAULT_RUNTIME_DIR))
    parser.add_argument("--checkpoint_dir", default=os.fspath(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--tokenizer_model", default=os.fspath(DEFAULT_TOKENIZER_MODEL))
    parser.add_argument("--hl_vlm_dir", default=os.fspath(DEFAULT_HL_VLM_DIR))
    parser.add_argument("--eval_script", default=os.fspath(DEFAULT_EVAL_SCRIPT))
    parser.add_argument("--compare_script", default=os.fspath(DEFAULT_COMPARE_SCRIPT))
    parser.add_argument("--poll_interval_s", type=int, default=60)
    parser.add_argument("--capture_samples", type=int, default=2)
    parser.add_argument("--capture_rt_subframes", type=int, default=8)
    parser.add_argument("--fail_fast", action="store_true")
    return parser.parse_args()


def _extract_keyed_value(text: str, key: str) -> str:
    prefix = f"{key}="
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if prefix in line:
            return line.split(prefix, 1)[1].strip()
    return ""


def main() -> int:
    args = _parse_args()
    old_eval_dir = Path(args.old_eval_dir).expanduser().resolve()
    old_summary = old_eval_dir / "summary.json"
    while not old_summary.is_file():
        print(f"[followup] waiting for {old_summary}", flush=True)
        time.sleep(max(5, int(args.poll_interval_s)))

    results_root = Path(args.results_root).expanduser().resolve()
    eval_cmd = [
        "python3",
        os.fspath(Path(args.eval_script).expanduser().resolve()),
        "--model_label",
        str(args.new_model_label),
        "--episodes",
        "1",
        "--seed",
        "0",
        "--plan_path",
        os.fspath(Path(args.plan_path).expanduser().resolve()),
        "--results_root",
        os.fspath(results_root),
        "--myvla_root",
        os.fspath(Path(args.myvla_root).expanduser().resolve()),
        "--runtime_dir",
        os.fspath(Path(args.runtime_dir).expanduser().resolve()),
        "--checkpoint_dir",
        os.fspath(Path(args.checkpoint_dir).expanduser().resolve()),
        "--tokenizer_model",
        os.fspath(Path(args.tokenizer_model).expanduser().resolve()),
        "--hl_vlm_dir",
        os.fspath(Path(args.hl_vlm_dir).expanduser().resolve()),
        "--capture_samples",
        str(int(args.capture_samples)),
        "--capture_rt_subframes",
        str(int(args.capture_rt_subframes)),
    ]
    if bool(args.fail_fast):
        eval_cmd.append("--fail_fast")
    print(f"[followup] launching new-model eval: {' '.join(eval_cmd)}", flush=True)
    proc = subprocess.run(eval_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    sys.stdout.write(proc.stdout)
    sys.stdout.flush()
    if proc.returncode != 0:
        print(f"[followup] new-model eval failed with returncode={proc.returncode}", flush=True)
        return int(proc.returncode)

    new_eval_dir_text = _extract_keyed_value(proc.stdout, "eval_dir")
    if new_eval_dir_text:
        new_eval_dir = Path(new_eval_dir_text).expanduser().resolve()
    else:
        candidates = sorted(results_root.glob(f"{args.new_model_label}_*"))
        if not candidates:
            raise RuntimeError("Could not resolve the new-model eval directory.")
        new_eval_dir = candidates[-1].resolve()

    compare_out_dir = results_root / f"compare_{old_eval_dir.name}_vs_{new_eval_dir.name}"
    compare_cmd = [
        "python3",
        os.fspath(Path(args.compare_script).expanduser().resolve()),
        "--baseline_summary",
        os.fspath(old_summary),
        "--candidate_summary",
        os.fspath(new_eval_dir / "summary.json"),
        "--output_dir",
        os.fspath(compare_out_dir),
    ]
    print(f"[followup] running comparison: {' '.join(compare_cmd)}", flush=True)
    compare_proc = subprocess.run(compare_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    sys.stdout.write(compare_proc.stdout)
    sys.stdout.flush()
    return int(compare_proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
