#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
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
            "Collect Fold Tops expert demonstrations by directly running DexGarmentLab "
            "Env_StandAlone/Fold_Tops_Env.py in one isolated Isaac process per attempt."
        )
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--target_successes", type=int, default=8)
    parser.add_argument("--max_attempts", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--record_video_flag", action="store_true")
    parser.add_argument("--attempt_timeout_s", type=float, default=7200.0)
    parser.add_argument("--dex_root", default=os.fspath(DEFAULT_DEX_ROOT))
    parser.add_argument("--myvla_root", default=os.fspath(DEFAULT_MYVLA_ROOT))
    parser.add_argument("--isaac_python", default=os.fspath(DEFAULT_ISAAC_PY))
    parser.add_argument("--launcher_py", default=os.fspath(DEFAULT_LAUNCHER))
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _snapshot_files(root: Path, pattern: str) -> dict[str, str]:
    if not root.is_dir():
        return {}
    result: dict[str, str] = {}
    for path in sorted(root.glob(pattern)):
        if path.is_file():
            result[path.name] = os.fspath(path.resolve())
    return result


def _snapshot_log_lines(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def _capture_dex_snapshot(dex_root: Path) -> dict[str, Any]:
    data_root = dex_root / "Data" / "Fold_Tops"
    return {
        "train_data": _snapshot_files(data_root / "train_data", "data_*.npz"),
        "video": _snapshot_files(data_root / "video", "video_*.mp4"),
        "final_state_pic": _snapshot_files(data_root / "final_state_pic", "img_*.png"),
        "log_lines": _snapshot_log_lines(data_root / "data_collection_log.txt"),
    }


def _diff_new_files(before: dict[str, str], after: dict[str, str]) -> list[str]:
    return [after[name] for name in sorted(after.keys()) if name not in before]


def _parse_final_result(log_text: str) -> bool | None:
    match = re.search(r"final result:\s*(True|False)", str(log_text or ""), flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).lower() == "true"


def _run_child_attempt(
    *,
    args: argparse.Namespace,
    attempt_seed: int,
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
        "--env_random_flag",
        "False",
    ]

    child_log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = str(int(attempt_seed))
    with child_log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        proc = subprocess.Popen(
            command,
            cwd=os.fspath(Path(args.myvla_root).expanduser().resolve()),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
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


def _copy_optional(src_path: str, dst_path: Path) -> str:
    if not src_path:
        return ""
    src = Path(src_path)
    if not src.is_file():
        return ""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_path)
    return os.fspath(dst_path.resolve())


def main() -> int:
    args = _parse_args()
    output_dir = Path(str(args.output_dir)).expanduser().resolve()
    child_runs_dir = output_dir / "child_runs"
    manifest_path = output_dir / "manifest.jsonl"
    summary_path = output_dir / "summary.json"
    dex_root = Path(str(args.dex_root)).expanduser().resolve()

    attempts, successes = _prepare_output_dir(output_dir, resume=bool(args.resume))
    child_runs_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "argv": sys.argv,
        "output_dir": os.fspath(output_dir),
        "gpu": int(args.gpu),
        "target_successes": int(args.target_successes),
        "max_attempts": int(args.max_attempts),
        "seed": int(args.seed),
        "record_video_flag": bool(args.record_video_flag),
        "attempt_timeout_s": float(args.attempt_timeout_s),
        "dex_root": os.fspath(dex_root),
        "myvla_root": os.fspath(Path(args.myvla_root).expanduser().resolve()),
        "isaac_python": os.fspath(Path(args.isaac_python).expanduser().resolve()),
        "launcher_py": os.fspath(Path(args.launcher_py).expanduser().resolve()),
        "resume": bool(args.resume),
    }
    _write_json(output_dir / "config.json", config)

    while successes < int(args.target_successes) and attempts < int(args.max_attempts):
        attempts += 1
        attempt_seed = int(args.seed) + int(attempts) - 1
        attempt_dir = output_dir / f"attempt_{attempts:04d}"
        child_dir = child_runs_dir / f"attempt_{attempts:04d}"
        child_log_path = child_dir / "launcher_output.log"
        child_dir.mkdir(parents=True, exist_ok=True)
        attempt_dir.mkdir(parents=True, exist_ok=True)

        before = _capture_dex_snapshot(dex_root)
        print(
            json.dumps(
                {
                    "event": "dex_native_attempt_start",
                    "attempt": int(attempts),
                    "successes": int(successes),
                    "attempt_seed": int(attempt_seed),
                    "gpu": int(args.gpu),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        return_code, timeout_reason = _run_child_attempt(
            args=args,
            attempt_seed=int(attempt_seed),
            child_log_path=child_log_path,
        )

        after = _capture_dex_snapshot(dex_root)
        launcher_log = child_log_path.read_text(encoding="utf-8", errors="replace") if child_log_path.is_file() else ""
        final_result = _parse_final_result(launcher_log)
        new_npz = _diff_new_files(before["train_data"], after["train_data"])
        new_video = _diff_new_files(before["video"], after["video"])
        new_final_pic = _diff_new_files(before["final_state_pic"], after["final_state_pic"])
        new_log_lines = after["log_lines"][len(before["log_lines"]) :]
        success = bool(new_npz) or bool(final_result)

        copied_dataset = ""
        copied_video = ""
        copied_final_pic = ""
        if new_npz:
            copied_dataset = _copy_optional(new_npz[-1], attempt_dir / "dataset.npz")
        if new_video:
            copied_video = _copy_optional(new_video[-1], attempt_dir / "video.mp4")
        if new_final_pic:
            copied_final_pic = _copy_optional(new_final_pic[-1], attempt_dir / "final_state.png")
        if child_log_path.is_file():
            shutil.copy2(child_log_path, attempt_dir / "launcher_output.log")
        if new_log_lines:
            (attempt_dir / "data_collection_log_tail.txt").write_text(
                "\n".join(new_log_lines) + "\n",
                encoding="utf-8",
            )

        record = {
            "attempt": int(attempts),
            "attempt_seed": int(attempt_seed),
            "gpu": int(args.gpu),
            "success": bool(success),
            "launcher_return_code": int(return_code),
            "timeout_reason": timeout_reason or "",
            "final_result": final_result,
            "new_dataset_paths": new_npz,
            "new_video_paths": new_video,
            "new_final_state_paths": new_final_pic,
            "new_log_lines": new_log_lines,
            "copied_dataset": copied_dataset,
            "copied_video": copied_video,
            "copied_final_state": copied_final_pic,
            "attempt_dir": os.fspath(attempt_dir),
            "launcher_log": os.fspath(child_log_path),
        }
        _write_json(attempt_dir / "meta.json", record)
        _append_jsonl(manifest_path, record)

        if success:
            successes += 1

        _write_json(
            summary_path,
            {
                "ok": True,
                "output_dir": os.fspath(output_dir),
                "gpu": int(args.gpu),
                "target_successes": int(args.target_successes),
                "max_attempts": int(args.max_attempts),
                "attempts": int(attempts),
                "successes": int(successes),
                "record_video_flag": bool(args.record_video_flag),
                "complete": bool(successes >= int(args.target_successes)),
            },
        )
        print(
            json.dumps(
                {
                    "event": "dex_native_attempt_finish",
                    "attempt": int(attempts),
                    "success": bool(success),
                    "successes": int(successes),
                    "launcher_return_code": int(return_code),
                    "timeout_reason": timeout_reason or "",
                    "copied_dataset": copied_dataset,
                    "copied_video": copied_video,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    final_summary = {
        "ok": True,
        "output_dir": os.fspath(output_dir),
        "gpu": int(args.gpu),
        "target_successes": int(args.target_successes),
        "max_attempts": int(args.max_attempts),
        "attempts": int(attempts),
        "successes": int(successes),
        "complete": bool(successes >= int(args.target_successes)),
    }
    _write_json(summary_path, final_summary)
    print(json.dumps(final_summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
