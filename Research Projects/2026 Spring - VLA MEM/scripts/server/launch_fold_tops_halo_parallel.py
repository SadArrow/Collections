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
DEFAULT_RESULTS_ROOT = DEFAULT_BASE / "eval_results" / "fold_tops_halo_parallel"
DEFAULT_RUNNER = DEFAULT_BASE / "myVLA" / "scripts" / "server" / "run_fold_tops_halo_eval.py"
DEFAULT_MYVLA_ROOT = DEFAULT_BASE / "myVLA"
DEFAULT_DEX_ROOT = DEFAULT_BASE / "DexGarmentLab-main"
DEFAULT_LAUNCHER = DEFAULT_BASE / "myVLA" / "scripts" / "server" / "run_world_model_diffusion_vla_square_fold_isaac45_autowait.sh"
DEFAULT_RUNTIME_DIR = DEFAULT_BASE / "myVLA" / "WorldModelDiffusionVlaRuntime"
DEFAULT_CHECKPOINT_DIR = DEFAULT_BASE / "myVLA" / "pi05_droid_pytorch"
DEFAULT_TOKENIZER_MODEL = DEFAULT_BASE / "myVLA" / "assets" / "paligemma_tokenizer.model"
DEFAULT_HL_VLM_DIR = DEFAULT_BASE / "myVLA" / "pretrained_vlm" / "google_paligemma-3b-mix-224-bfloat16"
DEFAULT_VIZ_DIR = DEFAULT_BASE / "DexGarmentLab-main" / "server_viz"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Fold Tops HALO-style eval shards across multiple GPUs.")
    parser.add_argument("--model_label", required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpus", default="0,1,2,3", help="Comma-separated physical GPU ids.")
    parser.add_argument("--results_root", default=os.fspath(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--runner", default=os.fspath(DEFAULT_RUNNER))
    parser.add_argument("--myvla_root", default=os.fspath(DEFAULT_MYVLA_ROOT))
    parser.add_argument("--dex_root", default=os.fspath(DEFAULT_DEX_ROOT))
    parser.add_argument("--launcher", default=os.fspath(DEFAULT_LAUNCHER))
    parser.add_argument("--runtime_dir", default=os.fspath(DEFAULT_RUNTIME_DIR))
    parser.add_argument("--checkpoint_dir", default=os.fspath(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--tokenizer_model", default=os.fspath(DEFAULT_TOKENIZER_MODEL))
    parser.add_argument("--hl_vlm_dir", default=os.fspath(DEFAULT_HL_VLM_DIR))
    parser.add_argument("--viz_dir", default=os.fspath(DEFAULT_VIZ_DIR))
    parser.add_argument("--goal_text", default="Fold the shirt into a compact square block by folding both sleeves inward and then folding the lower hem upward using two robot arms.")
    parser.add_argument("--outer_steps", type=int, default=0)
    parser.add_argument("--min_outer_steps", type=int, default=50)
    parser.add_argument("--num_steps", type=int, default=4)
    parser.add_argument("--validation_threshold", type=float, default=0.12)
    parser.add_argument("--wait_interval_s", type=int, default=5)
    parser.add_argument("--idle_confirm_polls", type=int, default=1)
    parser.add_argument("--rollout_timeout_s", type=int, default=21600)
    parser.add_argument("--max_abs_joint_position", type=float, default=20.0)
    parser.add_argument("--implausible_joint_delta_factor", type=float, default=10.0)
    parser.add_argument("--capture_samples", type=int, default=3)
    parser.add_argument("--capture_rt_subframes", type=int, default=16)
    parser.add_argument("--keep_videos", action="store_true")
    parser.add_argument("--keep_step_artifacts", action="store_true")
    parser.add_argument("--fail_fast", action="store_true")
    return parser.parse_args()


def _parse_gpu_list(text: str) -> list[str]:
    items = [token.strip() for token in str(text).split(",") if token.strip()]
    if not items:
        raise ValueError("At least one GPU id is required.")
    return items


def _episode_splits(episodes: int, gpu_ids: list[str]) -> dict[str, list[int]]:
    result = {gpu_id: [] for gpu_id in gpu_ids}
    for episode_index in range(int(episodes)):
        gpu_id = gpu_ids[episode_index % len(gpu_ids)]
        result[gpu_id].append(int(episode_index))
    return result


def _indices_text(indices: list[int]) -> str:
    return ",".join(str(int(item)) for item in indices)


def main() -> int:
    args = _parse_args()
    gpu_ids = _parse_gpu_list(args.gpus)
    splits = _episode_splits(int(args.episodes), gpu_ids)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    launch_dir = Path(args.results_root).expanduser().resolve() / f"{args.model_label}_seed{int(args.seed)}_{timestamp}"
    launch_dir.mkdir(parents=True, exist_ok=False)
    plan_path = launch_dir / "shared_episode_plan.json"

    launch_manifest: dict[str, object] = {
        "model_label": str(args.model_label),
        "seed": int(args.seed),
        "episodes": int(args.episodes),
        "gpus": gpu_ids,
        "launch_dir": os.fspath(launch_dir),
        "plan_path": os.fspath(plan_path),
        "workers": [],
    }

    for worker_rank, gpu_id in enumerate(gpu_ids):
        episode_indices = splits[gpu_id]
        if not episode_indices:
            continue
        worker_label = f"{args.model_label}_seed{int(args.seed)}_gpu{gpu_id}"
        log_path = launch_dir / f"{worker_label}.out"
        cmd = [
            sys.executable,
            os.fspath(Path(args.runner).expanduser().resolve()),
            "--model_label",
            str(args.model_label),
            "--episodes",
            str(int(args.episodes)),
            "--seed",
            str(int(args.seed)),
            "--plan_path",
            os.fspath(plan_path),
            "--episode_indices",
            _indices_text(episode_indices),
            "--eval_name_suffix",
            f"seed{int(args.seed)}_gpu{gpu_id}",
            "--results_root",
            os.fspath(launch_dir / "workers"),
            "--myvla_root",
            os.fspath(Path(args.myvla_root).expanduser().resolve()),
            "--dex_root",
            os.fspath(Path(args.dex_root).expanduser().resolve()),
            "--launcher",
            os.fspath(Path(args.launcher).expanduser().resolve()),
            "--runtime_dir",
            os.fspath(Path(args.runtime_dir).expanduser().resolve()),
            "--checkpoint_dir",
            os.fspath(Path(args.checkpoint_dir).expanduser().resolve()),
            "--tokenizer_model",
            os.fspath(Path(args.tokenizer_model).expanduser().resolve()),
            "--hl_vlm_dir",
            os.fspath(Path(args.hl_vlm_dir).expanduser().resolve()),
            "--viz_dir",
            os.fspath(Path(args.viz_dir).expanduser().resolve()),
            "--goal_text",
            str(args.goal_text),
            "--outer_steps",
            str(int(args.outer_steps)),
            "--min_outer_steps",
            str(int(args.min_outer_steps)),
            "--num_steps",
            str(int(args.num_steps)),
            "--validation_threshold",
            str(float(args.validation_threshold)),
            "--wait_interval_s",
            str(int(args.wait_interval_s)),
            "--idle_confirm_polls",
            str(int(args.idle_confirm_polls)),
            "--preferred_gpus",
            str(gpu_id),
            "--rollout_timeout_s",
            str(int(args.rollout_timeout_s)),
            "--max_abs_joint_position",
            str(float(args.max_abs_joint_position)),
            "--implausible_joint_delta_factor",
            str(float(args.implausible_joint_delta_factor)),
            "--capture_samples",
            str(int(args.capture_samples)),
            "--capture_rt_subframes",
            str(int(args.capture_rt_subframes)),
        ]
        if bool(args.keep_videos):
            cmd.append("--keep_videos")
        if bool(args.keep_step_artifacts):
            cmd.append("--keep_step_artifacts")
        if bool(args.fail_fast):
            cmd.append("--fail_fast")

        log_fh = log_path.open("a", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env=dict(os.environ),
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )
        launch_manifest["workers"].append(
            {
                "worker_rank": int(worker_rank),
                "gpu_id": str(gpu_id),
                "pid": int(proc.pid),
                "episode_indices": episode_indices,
                "log_path": os.fspath(log_path),
            }
        )

    manifest_path = launch_dir / "launch_manifest.json"
    manifest_path.write_text(json.dumps(launch_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"launch_manifest": os.fspath(manifest_path), "workers": launch_manifest["workers"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
