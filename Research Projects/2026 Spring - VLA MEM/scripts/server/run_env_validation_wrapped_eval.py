#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from halo14_task_registry import TASK_BY_NAME


DEFAULT_BASE = Path(os.environ.get("MYVLA_SERVER_BASE", "/root/workspace/qianyupeng"))
DEFAULT_DEX_ROOT = DEFAULT_BASE / "DexGarmentLab-main"
DEFAULT_ISAAC_PY = DEFAULT_BASE / "isaac-sim-standalone@4.5.0" / "python.sh"
DEFAULT_RESULTS_ROOT = DEFAULT_BASE / "eval_results" / "wrapped_env_validation"
DEFAULT_RUNTIME_DIR = DEFAULT_BASE / "myVLA" / "WorldModelDiffusionVlaRuntime"
DEFAULT_WRAPPER = DEFAULT_BASE / "myVLA" / "scripts" / "server" / "env_validation_entry_wrapper.py"
DEFAULT_CARBONITE_SEMAPHORE = Path("/dev/shm/sem.carbonite-sharedmemory")
DEFAULT_VK_ICD_JSON = DEFAULT_BASE / "downloads" / "nvidia570" / "nvidia_egl_icd_570.86.10.json"
DEFAULT_EXTRA_PYDEPS = DEFAULT_RUNTIME_DIR / "pydeps"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one DexGarmentLab Env_Validation task through a stable wrapper.")
    parser.add_argument("--task_name", default="Fold_Tops")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0, help="Physical GPU id to isolate via CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--label", default="wrapped_env_validation")
    parser.add_argument("--dex_root", default=os.fspath(DEFAULT_DEX_ROOT))
    parser.add_argument("--isaac_python", default=os.fspath(DEFAULT_ISAAC_PY))
    parser.add_argument("--results_root", default=os.fspath(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--runtime_dir", default=os.fspath(DEFAULT_RUNTIME_DIR))
    parser.add_argument("--extra_pydeps", default=os.fspath(DEFAULT_EXTRA_PYDEPS))
    parser.add_argument("--wrapper_script", default=os.fspath(DEFAULT_WRAPPER))
    parser.add_argument("--carbonite_semaphore", default=os.fspath(DEFAULT_CARBONITE_SEMAPHORE))
    parser.add_argument("--vk_icd_json", default=os.fspath(DEFAULT_VK_ICD_JSON))
    parser.add_argument("--env_random_flag", default="True")
    parser.add_argument("--garment_random_flag", default="True")
    parser.add_argument("--record_video_flag", default="True")
    parser.add_argument("--validation_flag", default="True")
    parser.add_argument("--training_data_num", type=int, default=100)
    parser.add_argument("--stage_1_checkpoint_num", type=int, default=1500)
    parser.add_argument("--stage_2_checkpoint_num", type=int, default=1500)
    parser.add_argument("--stage_3_checkpoint_num", type=int, default=1500)
    parser.add_argument(
        "--headless_excluded_extensions",
        default="isaacsim.asset.importer.urdf,isaacsim.asset.importer.mjcf",
    )
    parser.add_argument("--disable_fabric_delegate", action="store_true")
    parser.add_argument("--verbose_wrapper", action="store_true")
    return parser.parse_args()


def _safe_symlink(*, link_path: Path, candidates: list[Path]) -> None:
    if link_path.exists():
        return
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except Exception:
            resolved = candidate
        if resolved.exists():
            link_path.parent.mkdir(parents=True, exist_ok=True)
            link_path.symlink_to(resolved)
            return


def _build_extcache_library_path(isaac_root: Path) -> str:
    extscache_root = isaac_root / "extscache"
    if not extscache_root.is_dir():
        return ""
    items: list[str] = []
    seen: set[str] = set()
    for pattern in ("*/bin", "*/lib", "*/bin/deps", "*/lib/deps"):
        for path in sorted(extscache_root.glob(pattern)):
            if not path.is_dir():
                continue
            text = os.fspath(path)
            if text in seen:
                continue
            seen.add(text)
            items.append(text)
    return ":".join(items)


def _build_env(args: argparse.Namespace) -> dict[str, str]:
    runtime_dir = Path(args.runtime_dir).expanduser().resolve()
    extra_pydeps = Path(args.extra_pydeps).expanduser().resolve()
    local_syslib_dir = runtime_dir / "syslibs" / "usr" / "lib" / "x86_64-linux-gnu"
    local_syslib_dir.mkdir(parents=True, exist_ok=True)
    isaac_root = Path(args.isaac_python).expanduser().resolve().parents[0]
    base = Path(args.dex_root).expanduser().resolve().parents[0]

    _safe_symlink(
        link_path=local_syslib_dir / "libcuda.so",
        candidates=[
            Path("/usr/lib/x86_64-linux-gnu/libcuda.so"),
            Path("/usr/lib/x86_64-linux-gnu/libcuda.so.1"),
        ],
    )
    _safe_symlink(
        link_path=local_syslib_dir / "libcuda.so.1",
        candidates=[
            Path("/usr/lib/x86_64-linux-gnu/libcuda.so.1"),
            local_syslib_dir / "libcuda.so",
        ],
    )
    _safe_symlink(
        link_path=local_syslib_dir / "libGLU.so.1",
        candidates=[
            Path("/usr/lib/x86_64-linux-gnu/libGLU.so.1"),
            Path("/lib/x86_64-linux-gnu/libGLU.so.1"),
        ],
    )

    nvidia_gl_lib_dir = base / "downloads" / "nvidia570" / "libnvidia-gl-570_570.86.10-0ubuntu1_amd64" / "usr" / "lib" / "x86_64-linux-gnu"
    egl_vendor_dir = base / "downloads" / "nvidia570" / "libnvidia-gl-570_570.86.10-0ubuntu1_amd64" / "usr" / "share" / "glvnd" / "egl_vendor.d"
    egl_platform_dir = base / "downloads" / "nvidia570" / "libnvidia-gl-570_570.86.10-0ubuntu1_amd64" / "usr" / "share" / "egl" / "egl_external_platform.d"
    vk_layer_dir = base / "downloads" / "nvidia570" / "libnvidia-gl-570_570.86.10-0ubuntu1_amd64" / "usr" / "share" / "vulkan" / "implicit_layer.d"
    x11_runtime_lib_dir = base / "downloads" / "x11_runtime_libs" / "usr" / "lib" / "x86_64-linux-gnu"
    libxt_lib_dir = base / "downloads" / "libxt6" / "usr" / "lib" / "x86_64-linux-gnu"

    ld_parts = [os.fspath(local_syslib_dir)]
    for candidate in (x11_runtime_lib_dir, libxt_lib_dir, nvidia_gl_lib_dir):
        if candidate.is_dir():
            ld_parts.append(os.fspath(candidate))
    ld_parts.extend(["/lib/x86_64-linux-gnu", "/usr/lib/x86_64-linux-gnu"])
    extscache_paths = _build_extcache_library_path(isaac_root)
    if extscache_paths:
        ld_parts.extend([item for item in extscache_paths.split(":") if item])

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["CUDA_VISIBLE_DEVICES"] = str(int(args.gpu))
    env["LD_LIBRARY_PATH"] = ":".join(ld_parts + [str(env.get("LD_LIBRARY_PATH", ""))]).rstrip(":")
    if str(args.vk_icd_json).strip():
        env["VK_ICD_FILENAMES"] = os.fspath(Path(args.vk_icd_json).expanduser().resolve())
    if egl_vendor_dir.is_dir():
        env["__EGL_VENDOR_LIBRARY_DIRS"] = os.fspath(egl_vendor_dir)
    if egl_platform_dir.is_dir():
        env["__EGL_EXTERNAL_PLATFORM_CONFIG_DIRS"] = os.fspath(egl_platform_dir)
    if vk_layer_dir.is_dir():
        env["VK_LAYER_PATH"] = os.fspath(vk_layer_dir)
    if extra_pydeps.is_dir():
        env["ENV_VALIDATION_EXTRA_PYTHONPATH"] = os.fspath(extra_pydeps)
    return env


def _clear_carbonite_semaphore(path_text: str) -> str:
    path = Path(str(path_text).strip())
    if not str(path):
        return ""
    try:
        if path.exists():
            path.unlink()
            return f"cleared:{os.fspath(path)}"
        return f"missing:{os.fspath(path)}"
    except Exception as exc:
        return f"clear_failed:{os.fspath(path)}:{type(exc).__name__}:{exc}"


def _list_relative_files(root: Path) -> set[str]:
    if not root.exists():
        return set()
    return {
        os.fspath(path.relative_to(root)).replace("\\", "/")
        for path in root.rglob("*")
        if path.is_file()
    }


def _read_lines(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def _copy_new_artifacts(*, source_root: Path, rel_paths: set[str], dest_dir: Path) -> list[str]:
    copied: list[str] = []
    for rel_path in sorted(rel_paths):
        source = source_root / rel_path
        if not source.is_file():
            continue
        target = dest_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(os.fspath(target))
    return copied


def _parse_success_from_lines(lines: list[str]) -> bool | None:
    pattern = re.compile(r"result:\s*(True|False)", re.IGNORECASE)
    for line in reversed(lines):
        match = pattern.search(line)
        if match:
            return match.group(1).lower() == "true"
    return None


def _detect_runtime_exception(lines: list[str]) -> str | None:
    markers = (
        "Traceback (most recent call last):",
        "ModuleNotFoundError:",
        "ImportError:",
        "[py stderr]:",
    )
    matched = [line for line in lines if any(marker in line for marker in markers)]
    if not matched:
        return None
    return "\n".join(matched[-20:])


def _episode_command(args: argparse.Namespace, *, spec: Any, episode_portable_root: Path) -> list[str]:
    command = [
        os.fspath(Path(args.isaac_python).expanduser().resolve()),
        os.fspath(Path(args.wrapper_script).expanduser().resolve()),
        "--dex_root",
        os.fspath(Path(args.dex_root).expanduser().resolve()),
        "--script_rel",
        os.fspath(Path("Env_Validation") / spec.script_name).replace("\\", "/"),
        "--portable_root",
        os.fspath(episode_portable_root),
        "--active_gpu",
        "0",
        "--physics_gpu",
        "0",
        "--headless_excluded_extensions",
        str(args.headless_excluded_extensions),
    ]
    if bool(args.disable_fabric_delegate):
        command.append("--disable_fabric_delegate")
    if bool(args.verbose_wrapper):
        command.append("--verbose")
    command.extend(
        [
            "--",
            "--env_random_flag",
            str(args.env_random_flag),
            "--garment_random_flag",
            str(args.garment_random_flag),
            "--record_video_flag",
            str(args.record_video_flag),
            "--validation_flag",
            str(args.validation_flag),
            "--training_data_num",
            str(int(args.training_data_num)),
            "--stage_1_checkpoint_num",
            str(int(args.stage_1_checkpoint_num)),
            "--stage_2_checkpoint_num",
            str(int(args.stage_2_checkpoint_num)),
            "--stage_3_checkpoint_num",
            str(int(args.stage_3_checkpoint_num)),
        ]
    )
    return command


def _run_episode(*, args: argparse.Namespace, spec: Any, eval_dir: Path, episode_index: int) -> dict[str, Any]:
    dex_root = Path(args.dex_root).expanduser().resolve()
    data_root = dex_root / spec.data_dir_rel
    final_pic_dir = data_root / "final_state_pic"
    video_dir = data_root / "video"
    validation_log = data_root / "validation_log.txt"

    before_pic = _list_relative_files(final_pic_dir)
    before_video = _list_relative_files(video_dir)
    before_log_lines = _read_lines(validation_log)

    episode_dir = eval_dir / spec.task_name / f"episode_{int(episode_index):03d}"
    episode_dir.mkdir(parents=True, exist_ok=False)
    episode_log = episode_dir / "stdout.log"
    portable_root = Path(args.runtime_dir).expanduser().resolve() / "wrapped_env_validation" / spec.task_name / f"episode_{int(episode_index):03d}"
    portable_root.mkdir(parents=True, exist_ok=True)
    semaphore_status = _clear_carbonite_semaphore(str(args.carbonite_semaphore))
    command = _episode_command(args, spec=spec, episode_portable_root=portable_root)
    started_at = time.time()
    env = _build_env(args)
    with episode_log.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(
            command,
            cwd=os.fspath(dex_root),
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env=env,
        )
    elapsed_s = float(time.time() - started_at)
    stdout_lines = _read_lines(episode_log)

    after_pic = _list_relative_files(final_pic_dir)
    after_video = _list_relative_files(video_dir)
    after_log_lines = _read_lines(validation_log)

    copied_pics = _copy_new_artifacts(
        source_root=final_pic_dir,
        rel_paths=after_pic - before_pic,
        dest_dir=episode_dir / "final_state_pic",
    )
    copied_videos = _copy_new_artifacts(
        source_root=video_dir,
        rel_paths=after_video - before_video,
        dest_dir=episode_dir / "video",
    )
    new_log_lines = after_log_lines[len(before_log_lines) :]
    copied_log = None
    if new_log_lines:
        copied_log_path = episode_dir / "validation_log_delta.txt"
        copied_log_path.write_text("\n".join(new_log_lines) + "\n", encoding="utf-8")
        copied_log = os.fspath(copied_log_path)

    success = _parse_success_from_lines(new_log_lines)
    detected_runtime_exception = _detect_runtime_exception(stdout_lines)
    row = {
        "task_name": spec.task_name,
        "episode_index": int(episode_index),
        "return_code": int(proc.returncode),
        "elapsed_s": elapsed_s,
        "success": success,
        "stdout_log": os.fspath(episode_log),
        "copied_validation_log_delta": copied_log,
        "copied_final_state_pics": copied_pics,
        "copied_videos": copied_videos,
        "source_data_dir": os.fspath(data_root),
        "portable_root": os.fspath(portable_root),
        "carbonite_semaphore_status": semaphore_status,
        "detected_runtime_exception": detected_runtime_exception,
        "command": command,
    }
    (episode_dir / "episode_summary.json").write_text(
        json.dumps(row, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return row


def main() -> int:
    args = _parse_args()
    if str(args.task_name) not in TASK_BY_NAME:
        raise KeyError(f"Unknown HALO task: {args.task_name!r}")
    spec = TASK_BY_NAME[str(args.task_name)]

    results_root = Path(args.results_root).expanduser().resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    eval_dir = results_root / f"{args.label}_{timestamp}"
    eval_dir.mkdir(parents=True, exist_ok=False)

    config = {
        "task_name": spec.task_name,
        "script_name": spec.script_name,
        "episodes": int(args.episodes),
        "gpu": int(args.gpu),
        "dex_root": os.fspath(Path(args.dex_root).expanduser().resolve()),
        "isaac_python": os.fspath(Path(args.isaac_python).expanduser().resolve()),
        "wrapper_script": os.fspath(Path(args.wrapper_script).expanduser().resolve()),
        "runtime_dir": os.fspath(Path(args.runtime_dir).expanduser().resolve()),
    }
    (eval_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    rows: list[dict[str, Any]] = []
    print(f"[wrapped-env] task={spec.task_name} script={spec.script_name}", flush=True)
    for episode_index in range(int(args.episodes)):
        print(f"[wrapped-env] episode={episode_index:03d}", flush=True)
        row = _run_episode(args=args, spec=spec, eval_dir=eval_dir, episode_index=episode_index)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    summary = {
        "task_name": spec.task_name,
        "episodes": int(len(rows)),
        "successes": int(sum(1 for row in rows if row.get("success") is True)),
        "return_codes": {str(code): int(sum(1 for row in rows if int(row.get("return_code", 0)) == code)) for code in sorted({int(row.get("return_code", 0)) for row in rows})},
        "rows": rows,
        "eval_dir": os.fspath(eval_dir),
    }
    (eval_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[wrapped-env] eval_dir={os.fspath(eval_dir)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
