#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_BASE = Path(os.environ.get("MYVLA_SERVER_BASE") or Path(__file__).resolve().parents[3])
DEFAULT_MYVLA_ROOT = DEFAULT_BASE / "myVLA"
DEFAULT_CHECKPOINT_DIR = DEFAULT_MYVLA_ROOT / "pi05_droid_pytorch"
DEFAULT_OUTPUT_ROOT = DEFAULT_BASE / "artifacts" / "fold_tops_ll_pi05_bimanual_train"
DEFAULT_PYDEPS_DIR = DEFAULT_MYVLA_ROOT / "WorldModelDiffusionVlaRuntime" / "pydeps"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert native Dex Fold Tops expert data and launch 64/60 pi0.5 bimanual training."
    )
    parser.add_argument("--artifact_dirs", nargs="+", required=True)
    parser.add_argument("--output_root", default=os.fspath(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--checkpoint_dir", default=os.fspath(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--pydeps_dir", default=os.fspath(DEFAULT_PYDEPS_DIR))
    parser.add_argument("--goal", default="")
    parser.add_argument("--prompt_style", choices=("phase_structured", "goal_only"), default="goal_only")
    parser.add_argument("--max_episodes", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--eval_ratio", type=float, default=0.05)
    parser.add_argument("--train_scope", choices=("expert", "full"), default="expert")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--active_action_dim", type=int, default=60)
    parser.add_argument("--model_action_dim", type=int, default=64)
    parser.add_argument("--asset_id", default="droid")
    parser.add_argument("--target_mode", choices=("abs", "delta"), default="delta")
    parser.add_argument("--split_unit", choices=("episode", "garment"), default="episode")
    return parser.parse_args()


def _run(cmd: list[str], *, cwd: Path, pydeps_dir: Path | None = None) -> None:
    print(json.dumps({"event": "run_command", "cmd": cmd, "cwd": os.fspath(cwd)}, ensure_ascii=False), flush=True)
    env = os.environ.copy()
    if pydeps_dir is not None and pydeps_dir.is_dir():
        existing = str(env.get("PYTHONPATH", "")).strip()
        env["PYTHONPATH"] = os.fspath(pydeps_dir) if not existing else f"{os.fspath(pydeps_dir)}:{existing}"
    subprocess.run(cmd, cwd=os.fspath(cwd), env=env, check=True)


def main() -> None:
    args = _parse_args()
    myvla_root = DEFAULT_MYVLA_ROOT.resolve()
    output_root = Path(str(args.output_root)).expanduser().resolve()
    pydeps_dir = Path(str(args.pydeps_dir)).expanduser().resolve() if str(args.pydeps_dir).strip() else None
    converted_dir = output_root / "converted"
    train_dir = output_root / "train"
    output_root.mkdir(parents=True, exist_ok=True)

    prepare_cmd = [
        str(args.python_bin),
        os.fspath(myvla_root / "scripts" / "data" / "prepare_fold_tops_ll_pi05_from_native_dex.py"),
        "--artifact_dirs",
        *[str(Path(item).expanduser().resolve()) for item in args.artifact_dirs],
        "--output_dir",
        os.fspath(converted_dir),
    ]
    if str(args.goal).strip():
        prepare_cmd += ["--goal", str(args.goal).strip()]
    prepare_cmd += ["--prompt_style", str(args.prompt_style)]
    if int(args.max_episodes) > 0:
        prepare_cmd += ["--max_episodes", str(int(args.max_episodes))]
    _run(prepare_cmd, cwd=myvla_root, pydeps_dir=pydeps_dir)

    train_cmd = [
        str(args.python_bin),
        os.fspath(myvla_root / "scripts" / "train" / "train_fold_tops_ll_pi05_dexshadow_sft.py"),
        "--checkpoint_dir",
        os.fspath(Path(str(args.checkpoint_dir)).expanduser().resolve()),
        "--manifest",
        os.fspath(converted_dir / "manifest.jsonl"),
        "--output_dir",
        os.fspath(train_dir),
        "--asset_id",
        str(args.asset_id),
        "--seed",
        str(int(args.seed)),
        "--eval_ratio",
        str(float(args.eval_ratio)),
        "--epochs",
        str(float(args.epochs)),
        "--learning_rate",
        str(float(args.learning_rate)),
        "--train_batch_size",
        str(int(args.train_batch_size)),
        "--eval_batch_size",
        str(int(args.eval_batch_size)),
        "--gradient_accumulation_steps",
        str(int(args.gradient_accumulation_steps)),
        "--train_scope",
        str(args.train_scope),
        "--active_action_dim",
        str(int(args.active_action_dim)),
        "--model_action_dim",
        str(int(args.model_action_dim)),
        "--target_mode",
        str(args.target_mode),
        "--split_unit",
        str(args.split_unit),
    ]
    if int(args.max_episodes) > 0:
        train_cmd += ["--max_episodes", str(int(args.max_episodes))]
    if int(args.max_samples) > 0:
        train_cmd += ["--max_samples", str(int(args.max_samples))]
    if bool(args.bf16):
        train_cmd.append("--bf16")
    if bool(args.gradient_checkpointing):
        train_cmd.append("--gradient_checkpointing")
    _run(train_cmd, cwd=myvla_root, pydeps_dir=pydeps_dir)

    summary = {
        "ok": True,
        "output_root": os.fspath(output_root),
        "converted_manifest": os.fspath(converted_dir / "manifest.jsonl"),
        "train_output_dir": os.fspath(train_dir),
        "pydeps_dir": None if pydeps_dir is None else os.fspath(pydeps_dir),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
