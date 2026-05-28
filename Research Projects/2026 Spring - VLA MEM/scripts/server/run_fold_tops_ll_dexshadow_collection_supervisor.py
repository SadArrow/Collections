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
            "Collect Fold Tops DexShadow low-level data with one isolated Isaac process per attempt "
            "to avoid scene residue across attempts."
        )
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--target_successes", type=int, default=10)
    parser.add_argument("--max_attempts", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--record_video_flag", action="store_true")
    parser.add_argument("--validation_threshold", type=float, default=0.12)
    parser.add_argument("--attempt_timeout_s", type=float, default=7200.0)
    parser.add_argument("--goal", default="")
    parser.add_argument("--prompt_style", choices=("phase_structured", "goal_only"), default="goal_only")
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


def _remap_child_path(path_text: str, *, src_root: Path, dst_root: Path) -> str:
    raw = str(path_text or "").strip()
    if not raw:
        return raw
    src_path = Path(raw).expanduser().resolve()
    try:
        rel = src_path.relative_to(src_root)
    except ValueError:
        return raw
    return os.fspath((dst_root / rel).resolve())


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


def _run_child_attempt(
    *,
    args: argparse.Namespace,
    attempt_seed: int,
    child_output_dir: Path,
    child_log_path: Path,
) -> tuple[int, str | None]:
    child_output_dir.mkdir(parents=True, exist_ok=True)
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
        "../myVLA/scripts/data/collect_fold_tops_ll_dexshadow_dataset.py",
        "--run_as_main",
        "--",
        "--dex_root",
        os.fspath(Path(args.dex_root).expanduser().resolve()),
        "--output_dir",
        os.fspath(child_output_dir),
        "--target_successes",
        "1",
        "--max_attempts",
        "1",
        "--seed",
        str(int(attempt_seed)),
        "--validation_threshold",
        f"{float(args.validation_threshold):.6f}",
        "--prompt_style",
        str(args.prompt_style),
    ]
    if str(args.goal).strip():
        command += ["--goal", str(args.goal).strip()]
    if bool(args.record_video_flag):
        command.append("--record_video_flag")

    child_log_path.parent.mkdir(parents=True, exist_ok=True)
    with child_log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        env = dict(os.environ)
        env["MYVLA_FORCE_PROCESS_EXIT_AFTER_COLLECT"] = "1"
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


def _copy_success_attempt(*, src_attempt_dir: Path, dst_attempt_dir: Path, child_log_path: Path) -> None:
    if dst_attempt_dir.exists():
        raise FileExistsError(f"Destination attempt dir already exists: {dst_attempt_dir}")
    shutil.copytree(src_attempt_dir, dst_attempt_dir)
    if child_log_path.is_file():
        shutil.copy2(child_log_path, dst_attempt_dir / "launcher_output.log")


def main() -> int:
    args = _parse_args()
    output_dir = Path(str(args.output_dir)).expanduser().resolve()
    child_runs_dir = output_dir / "child_runs"
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "manifest.jsonl"

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
        "validation_threshold": float(args.validation_threshold),
        "attempt_timeout_s": float(args.attempt_timeout_s),
        "goal": str(args.goal),
        "prompt_style": str(args.prompt_style),
        "dex_root": os.fspath(Path(args.dex_root).expanduser().resolve()),
        "myvla_root": os.fspath(Path(args.myvla_root).expanduser().resolve()),
        "isaac_python": os.fspath(Path(args.isaac_python).expanduser().resolve()),
        "launcher_py": os.fspath(Path(args.launcher_py).expanduser().resolve()),
        "resume": bool(args.resume),
    }
    _write_json(output_dir / "config.json", config)

    while successes < int(args.target_successes) and attempts < int(args.max_attempts):
        attempts += 1
        attempt_seed = int(args.seed) + int(attempts) - 1
        child_output_dir = child_runs_dir / f"attempt_{attempts:04d}"
        child_log_path = child_output_dir / "launcher_output.log"

        print(
            json.dumps(
                {
                    "event": "dexshadow_supervisor_attempt_start",
                    "attempt": int(attempts),
                    "successes": int(successes),
                    "attempt_seed": int(attempt_seed),
                    "child_output_dir": os.fspath(child_output_dir),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        return_code, timeout_reason = _run_child_attempt(
            args=args,
            attempt_seed=int(attempt_seed),
            child_output_dir=child_output_dir,
            child_log_path=child_log_path,
        )

        child_manifest_rows = _read_jsonl(child_output_dir / "manifest.jsonl")
        child_summary = _read_json(child_output_dir / "stats.json") if (child_output_dir / "stats.json").is_file() else {}
        child_record = child_manifest_rows[-1] if child_manifest_rows else {}
        success = bool(child_record.get("success", False))

        aggregate_record: dict[str, Any] = {
            "attempt": int(attempts),
            "attempt_seed": int(attempt_seed),
            "success": bool(success),
            "launcher_return_code": int(return_code),
            "launcher_log": os.fspath(child_log_path),
            "child_output_dir": os.fspath(child_output_dir),
            "timeout_reason": timeout_reason or "",
            "child_summary": child_summary,
        }
        aggregate_record.update(child_record)
        aggregate_record["attempt"] = int(attempts)

        if success:
            src_attempt_dir = child_output_dir / "attempt_0001"
            dst_attempt_dir = output_dir / f"attempt_{attempts:04d}"
            if not src_attempt_dir.is_dir():
                raise FileNotFoundError(f"Successful child attempt is missing directory: {src_attempt_dir}")
            _copy_success_attempt(src_attempt_dir=src_attempt_dir, dst_attempt_dir=dst_attempt_dir, child_log_path=child_log_path)
            aggregate_record["dataset_path"] = _remap_child_path(
                str(aggregate_record.get("dataset_path", "")),
                src_root=src_attempt_dir,
                dst_root=dst_attempt_dir,
            )
            aggregate_record["video_path"] = _remap_child_path(
                str(aggregate_record.get("video_path", "")),
                src_root=src_attempt_dir,
                dst_root=dst_attempt_dir,
            )
            aggregate_record["meta_path"] = os.fspath((dst_attempt_dir / "meta.json").resolve())
            _write_json(dst_attempt_dir / "meta.json", aggregate_record)
            successes += 1

        _append_jsonl(manifest_path, aggregate_record)
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
                "validation_threshold": float(args.validation_threshold),
            },
        )
        print(
            json.dumps(
                {
                    "event": "dexshadow_supervisor_attempt_finish",
                    "attempt": int(attempts),
                    "success": bool(success),
                    "successes": int(successes),
                    "launcher_return_code": int(return_code),
                    "timeout_reason": timeout_reason or "",
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
