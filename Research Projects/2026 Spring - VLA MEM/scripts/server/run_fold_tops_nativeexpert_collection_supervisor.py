#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_BASE = Path(os.environ.get("MYVLA_SERVER_BASE") or Path(__file__).resolve().parents[3])
DEFAULT_DEX_ROOT = DEFAULT_BASE / "DexGarmentLab-main"
DEFAULT_MYVLA_ROOT = DEFAULT_BASE / "myVLA"
DEFAULT_ISAAC_PY = DEFAULT_BASE / "isaac-sim-standalone@4.5.0" / "python.sh"
DEFAULT_LAUNCHER = DEFAULT_MYVLA_ROOT / "scripts" / "server" / "run_myvla_envstandalone_wrapped.py"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect Fold Tops low-level data using DexGarmentLab's native Env_StandAlone expert trajectory "
            "and harvest only successful samples created by the built-in data_collection path."
        )
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--target_successes", type=int, default=10)
    parser.add_argument("--max_attempts", type=int, default=50)
    parser.add_argument("--attempt_timeout_s", type=float, default=7200.0)
    parser.add_argument("--dex_root", default=os.fspath(DEFAULT_DEX_ROOT))
    parser.add_argument("--myvla_root", default=os.fspath(DEFAULT_MYVLA_ROOT))
    parser.add_argument("--isaac_python", default=os.fspath(DEFAULT_ISAAC_PY))
    parser.add_argument("--launcher_py", default=os.fspath(DEFAULT_LAUNCHER))
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--record_video_flag", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _prepare_output_dir(output_dir: Path, *, resume: bool) -> tuple[int, int]:
    manifest_path = output_dir / "manifest.jsonl"
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        return 0, 0

    existing = _read_jsonl(manifest_path)
    if not existing:
        if resume:
            return 0, 0
        child_runs_dir = output_dir / "child_runs"
        attempt_dirs = list(output_dir.glob("attempt_*"))
        if child_runs_dir.exists() or attempt_dirs:
            raise FileExistsError(f"Output dir already contains data; use --resume to continue: {output_dir}")
        return 0, 0

    if not resume:
        raise FileExistsError(f"Manifest already exists; use --resume to continue: {manifest_path}")

    attempts = 0
    successes = 0
    for item in existing:
        attempts = max(attempts, int(item.get("attempt", 0)))
        if bool(item.get("success", False)):
            successes += 1
    return attempts, successes


def _kill_process_group(proc: subprocess.Popen[str], *, sig: int, timeout_s: float) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        return
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.2)


def _list_sorted_files(root: Path, pattern: str) -> list[Path]:
    return sorted(path.expanduser().resolve() for path in root.glob(pattern) if path.is_file())


def _last_nonempty_line(path: Path) -> str:
    if not path.is_file():
        return ""
    for raw_line in reversed(path.read_text(encoding="utf-8", errors="replace").splitlines()):
        line = raw_line.strip()
        if line:
            return line
    return ""


def _run_child_attempt(
    *,
    args: argparse.Namespace,
    child_log_path: Path,
) -> tuple[int, str | None]:
    command = [
        str(args.python_bin),
        os.fspath(Path(args.launcher_py).expanduser().resolve()),
        "--gpu",
        str(int(args.gpu)),
        "--gpu_binding_mode",
        "omniverse",
        "--dex_root",
        os.fspath(Path(args.dex_root).expanduser().resolve()),
        "--isaac_python",
        os.fspath(Path(args.isaac_python).expanduser().resolve()),
        "--runtime_dir",
        os.fspath(Path(args.myvla_root).expanduser().resolve() / "WorldModelDiffusionVlaRuntime"),
        "--script_rel",
        "Env_StandAlone/Fold_Tops_Env.py",
        "--run_as_main",
        "--",
        "--data_collection_flag",
        "True",
        "--record_video_flag",
        "True" if bool(args.record_video_flag) else "False",
        "--garment_random_flag",
        "True",
    ]

    child_log_path.parent.mkdir(parents=True, exist_ok=True)
    with child_log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        proc = subprocess.Popen(
            command,
            cwd=os.fspath(Path(args.dex_root).expanduser().resolve()),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        timeout_reason: str | None = None
        try:
            return_code = int(proc.wait(timeout=float(args.attempt_timeout_s)))
        except subprocess.TimeoutExpired:
            timeout_reason = f"attempt timed out after {float(args.attempt_timeout_s):.1f}s"
            _kill_process_group(proc, sig=signal.SIGTERM, timeout_s=20.0)
            if proc.poll() is None:
                _kill_process_group(proc, sig=signal.SIGKILL, timeout_s=5.0)
            return_code = int(proc.wait(timeout=10))
        return return_code, timeout_reason


def main() -> int:
    args = _parse_args()
    output_dir = Path(str(args.output_dir)).expanduser().resolve()
    dex_root = Path(str(args.dex_root)).expanduser().resolve()
    data_root = dex_root / "Data" / "Fold_Tops"
    train_data_root = data_root / "train_data"
    final_pic_root = data_root / "final_state_pic"
    video_root = data_root / "video"
    data_log_path = data_root / "data_collection_log.txt"
    child_runs_dir = output_dir / "child_runs"
    manifest_path = output_dir / "manifest.jsonl"
    summary_path = output_dir / "summary.json"

    attempts, successes = _prepare_output_dir(output_dir, resume=bool(args.resume))
    child_runs_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "argv": sys.argv,
        "output_dir": os.fspath(output_dir),
        "gpu": int(args.gpu),
        "target_successes": int(args.target_successes),
        "max_attempts": int(args.max_attempts),
        "attempt_timeout_s": float(args.attempt_timeout_s),
        "dex_root": os.fspath(dex_root),
        "myvla_root": os.fspath(Path(args.myvla_root).expanduser().resolve()),
        "isaac_python": os.fspath(Path(args.isaac_python).expanduser().resolve()),
        "launcher_py": os.fspath(Path(args.launcher_py).expanduser().resolve()),
        "record_video_flag": bool(args.record_video_flag),
        "native_data_root": os.fspath(data_root),
        "resume": bool(args.resume),
    }
    _write_json(output_dir / "config.json", config)

    while successes < int(args.target_successes) and attempts < int(args.max_attempts):
        attempts += 1
        child_output_dir = child_runs_dir / f"attempt_{attempts:04d}"
        child_output_dir.mkdir(parents=True, exist_ok=True)
        child_log_path = child_output_dir / "launcher_output.log"

        before_train = set(_list_sorted_files(train_data_root, "data_*.npz"))
        before_final_pic = set(_list_sorted_files(final_pic_root, "img_*.png"))
        before_video = set(_list_sorted_files(video_root, "video_*.mp4"))
        before_log_tail = _last_nonempty_line(data_log_path)

        print(
            json.dumps(
                {
                    "event": "nativeexpert_supervisor_attempt_start",
                    "attempt": int(attempts),
                    "successes": int(successes),
                    "child_output_dir": os.fspath(child_output_dir),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        return_code, timeout_reason = _run_child_attempt(args=args, child_log_path=child_log_path)

        after_train = set(_list_sorted_files(train_data_root, "data_*.npz"))
        after_final_pic = set(_list_sorted_files(final_pic_root, "img_*.png"))
        after_video = set(_list_sorted_files(video_root, "video_*.mp4"))
        after_log_tail = _last_nonempty_line(data_log_path)

        new_train = sorted(after_train - before_train)
        new_final_pic = sorted(after_final_pic - before_final_pic)
        new_video = sorted(after_video - before_video)
        success = bool(new_train)

        copied_train: list[str] = []
        copied_final_pic: list[str] = []
        copied_video: list[str] = []

        if new_train:
            native_dir = child_output_dir / "native_raw"
            native_dir.mkdir(parents=True, exist_ok=True)
            for src in new_train:
                dst = native_dir / src.name
                shutil.copy2(src, dst)
                copied_train.append(os.fspath(dst))

        if new_final_pic:
            pic_dir = child_output_dir / "native_final_state"
            pic_dir.mkdir(parents=True, exist_ok=True)
            for src in new_final_pic:
                dst = pic_dir / src.name
                shutil.copy2(src, dst)
                copied_final_pic.append(os.fspath(dst))

        if new_video:
            video_dir = child_output_dir / "native_video"
            video_dir.mkdir(parents=True, exist_ok=True)
            for src in new_video:
                dst = video_dir / src.name
                shutil.copy2(src, dst)
                copied_video.append(os.fspath(dst))

        record = {
            "attempt": int(attempts),
            "success": bool(success),
            "launcher_return_code": int(return_code),
            "timeout_reason": str(timeout_reason or ""),
            "native_train_files": [os.fspath(path) for path in new_train],
            "copied_train_files": copied_train,
            "native_final_state_pics": [os.fspath(path) for path in new_final_pic],
            "copied_final_state_pics": copied_final_pic,
            "native_videos": [os.fspath(path) for path in new_video],
            "copied_videos": copied_video,
            "data_collection_log_before": str(before_log_tail),
            "data_collection_log_after": str(after_log_tail),
        }
        _append_jsonl(manifest_path, record)

        print(
            json.dumps(
                {
                    "event": "nativeexpert_supervisor_attempt_finish",
                    "attempt": int(attempts),
                    "success": bool(success),
                    "successes": int(successes + (1 if success else 0)),
                    "launcher_return_code": int(return_code),
                    "timeout_reason": str(timeout_reason or ""),
                    "new_train_files": [os.fspath(path) for path in new_train],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        if success:
            successes += 1

        summary = {
            "ok": True,
            "output_dir": os.fspath(output_dir),
            "gpu": int(args.gpu),
            "target_successes": int(args.target_successes),
            "max_attempts": int(args.max_attempts),
            "attempts": int(attempts),
            "successes": int(successes),
            "complete": bool(successes >= int(args.target_successes)),
        }
        _write_json(summary_path, summary)

    print(json.dumps(json.loads(summary_path.read_text(encoding="utf-8")), ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
