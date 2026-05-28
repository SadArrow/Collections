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
from collections import Counter
from pathlib import Path
from typing import Any

from halo14_task_registry import HaloTaskSpec, parse_task_names


DEFAULT_BASE = Path(os.environ.get("MYVLA_SERVER_BASE", "/home/nvme04/qianyupeng"))
DEFAULT_DEX_ROOT = DEFAULT_BASE / "DexGarmentLab-main"
DEFAULT_ISAAC_PY = DEFAULT_BASE / "isaac-sim-standalone@4.5.0" / "python.sh"
DEFAULT_RESULTS_ROOT = DEFAULT_BASE / "eval_results" / "halo14_official"
DEFAULT_MYVLA_ROOT = DEFAULT_BASE / "myVLA"
DEFAULT_RUNTIME_DIR = DEFAULT_MYVLA_ROOT / "WorldModelDiffusionVlaRuntime"
DEFAULT_VK_ICD_JSON = DEFAULT_BASE / "vulkan_test" / "nvidia_abs_egl.json"
DEFAULT_CARBONITE_SEMAPHORE = Path("/dev/shm/sem.carbonite-sharedmemory")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the official 14-task HALO benchmark entrypoints non-interactively.")
    parser.add_argument("--tasks", default="all", help="Comma-separated task names or 'all'.")
    parser.add_argument("--episodes_per_task", type=int, default=1, help="Episodes to run for each selected task.")
    parser.add_argument("--seed", type=int, default=0, help="Recorded in metadata; the official scripts sample from time-based seeds.")
    parser.add_argument("--label", default="official_halo_smoke", help="Short label for the eval directory.")
    parser.add_argument("--dex_root", default=os.fspath(DEFAULT_DEX_ROOT))
    parser.add_argument("--isaac_python", default=os.fspath(DEFAULT_ISAAC_PY))
    parser.add_argument("--results_root", default=os.fspath(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--runtime_dir", default=os.fspath(DEFAULT_RUNTIME_DIR))
    parser.add_argument("--vk_icd_json", default=os.fspath(DEFAULT_VK_ICD_JSON))
    parser.add_argument("--carbonite_semaphore", default=os.fspath(DEFAULT_CARBONITE_SEMAPHORE))
    parser.add_argument("--training_data_num", type=int, default=100)
    parser.add_argument("--default_checkpoint_num", type=int, default=1500)
    parser.add_argument("--checkpoint_overrides_json", default="")
    parser.add_argument("--env_random_flag", default="True")
    parser.add_argument("--garment_random_flag", default="True")
    parser.add_argument("--record_video_flag", default="True")
    parser.add_argument("--validation_flag", default="True")
    parser.add_argument("--sleep_s_between_episodes", type=float, default=5.0)
    parser.add_argument("--fail_fast", action="store_true")
    return parser.parse_args()


def _load_checkpoint_overrides(path_text: str) -> dict[str, dict[str, int]]:
    if not str(path_text).strip():
        return {}
    path = Path(path_text).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected checkpoint overrides json object, got: {type(payload).__name__}")
    result: dict[str, dict[str, int]] = {}
    for task_name, item in payload.items():
        if not isinstance(item, dict):
            raise ValueError(f"Checkpoint override for {task_name!r} must be an object.")
        result[str(task_name)] = {str(k): int(v) for k, v in item.items()}
    return result


def _episode_arg_values(*, spec: HaloTaskSpec, args: argparse.Namespace, overrides: dict[str, dict[str, int]]) -> dict[str, int]:
    base = {
        "stage_1_checkpoint_num": int(args.default_checkpoint_num),
        "stage_2_checkpoint_num": int(args.default_checkpoint_num if spec.stage_count >= 2 else 0),
        "stage_3_checkpoint_num": int(args.default_checkpoint_num if spec.stage_count >= 3 else 0),
    }
    task_overrides = overrides.get(spec.task_name, {})
    for key, value in task_overrides.items():
        if key in base:
            base[key] = int(value)
    return base


def _list_relative_files(root: Path) -> set[str]:
    if not root.exists():
        return set()
    result: set[str] = set()
    for path in root.rglob("*"):
        if path.is_file():
            result.add(os.fspath(path.relative_to(root)).replace("\\", "/"))
    return result


def _read_lines(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def _safe_symlink(*, link_path: Path, candidates: list[Path]) -> None:
    if link_path.exists():
        return
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except Exception:
            resolved = candidate
        if resolved.exists():
            link_path.symlink_to(resolved)
            return


def _build_episode_env(args: argparse.Namespace) -> dict[str, str]:
    runtime_dir = Path(args.runtime_dir).expanduser().resolve()
    local_syslib_dir = runtime_dir / "syslibs" / "usr" / "lib" / "x86_64-linux-gnu"
    local_syslib_dir.mkdir(parents=True, exist_ok=True)

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

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    if str(args.vk_icd_json).strip():
        env["VK_ICD_FILENAMES"] = os.fspath(Path(args.vk_icd_json).expanduser().resolve())
    env["LD_LIBRARY_PATH"] = (
        f"{os.fspath(local_syslib_dir)}:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:"
        + str(env.get("LD_LIBRARY_PATH", ""))
    )
    return env


def _clear_allowed_carbonite_semaphore(args: argparse.Namespace) -> str:
    path_text = str(getattr(args, "carbonite_semaphore", "")).strip()
    if not path_text:
        return ""
    path = Path(path_text)
    try:
        if path.exists():
            path.unlink()
            return f"cleared:{os.fspath(path)}"
        return f"missing:{os.fspath(path)}"
    except Exception as exc:
        return f"clear_failed:{os.fspath(path)}:{type(exc).__name__}:{exc}"


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


def _run_episode(
    *,
    spec: HaloTaskSpec,
    episode_index: int,
    eval_dir: Path,
    args: argparse.Namespace,
    checkpoint_overrides: dict[str, dict[str, int]],
) -> dict[str, Any]:
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
    semaphore_status = _clear_allowed_carbonite_semaphore(args)

    episode_arg_values = _episode_arg_values(spec=spec, args=args, overrides=checkpoint_overrides)
    command = [
        os.fspath(Path(args.isaac_python).expanduser().resolve()),
        os.fspath(dex_root / "Env_Validation" / spec.script_name),
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
        str(int(episode_arg_values["stage_1_checkpoint_num"])),
        "--stage_2_checkpoint_num",
        str(int(episode_arg_values["stage_2_checkpoint_num"])),
        "--stage_3_checkpoint_num",
        str(int(episode_arg_values["stage_3_checkpoint_num"])),
    ]

    started_at = time.time()
    env = _build_episode_env(args)
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

    after_pic = _list_relative_files(final_pic_dir)
    after_video = _list_relative_files(video_dir)
    after_log_lines = _read_lines(validation_log)

    new_pic = after_pic - before_pic
    new_video = after_video - before_video
    new_log_lines = after_log_lines[len(before_log_lines) :]

    copied_pics = _copy_new_artifacts(source_root=final_pic_dir, rel_paths=new_pic, dest_dir=episode_dir / "final_state_pic")
    copied_videos = _copy_new_artifacts(source_root=video_dir, rel_paths=new_video, dest_dir=episode_dir / "video")
    copied_log = None
    if new_log_lines:
        copied_log_path = episode_dir / "validation_log_delta.txt"
        copied_log_path.write_text("\n".join(new_log_lines) + "\n", encoding="utf-8")
        copied_log = os.fspath(copied_log_path)

    success = _parse_success_from_lines(new_log_lines)
    if success is None and proc.returncode == 0:
        success = False

    episode_summary = {
        "task_name": spec.task_name,
        "episode_index": int(episode_index),
        "return_code": int(proc.returncode),
        "elapsed_s": elapsed_s,
        "success": success,
        "stage_count": int(spec.stage_count),
        "stdout_log": os.fspath(episode_log),
        "copied_validation_log_delta": copied_log,
        "copied_final_state_pics": copied_pics,
        "copied_videos": copied_videos,
        "source_data_dir": os.fspath(data_root),
        "stage_checkpoint_nums": episode_arg_values,
        "carbonite_semaphore_status": semaphore_status,
        "command": command,
    }
    (episode_dir / "episode_summary.json").write_text(
        json.dumps(episode_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return episode_summary


def _task_summary(*, spec: HaloTaskSpec, rows: list[dict[str, Any]]) -> dict[str, Any]:
    success_counter = 0
    known_successes = 0
    unknown_successes = 0
    return_codes = Counter()
    for row in rows:
        return_codes[int(row.get("return_code", 0))] += 1
        success_value = row.get("success")
        if success_value is None:
            unknown_successes += 1
        elif bool(success_value):
            known_successes += 1
            success_counter += 1
    denominator = max(1, len(rows) - unknown_successes)
    return {
        "task_name": spec.task_name,
        "episodes": int(len(rows)),
        "successes": int(success_counter),
        "unknown_successes": int(unknown_successes),
        "known_success_rate": float(success_counter) / float(denominator),
        "return_codes": {str(key): int(value) for key, value in sorted(return_codes.items())},
        "results": rows,
    }


def main() -> int:
    args = _parse_args()
    specs = parse_task_names(args.tasks)
    checkpoint_overrides = _load_checkpoint_overrides(args.checkpoint_overrides_json)

    results_root = Path(args.results_root).expanduser().resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    eval_dir = results_root / f"{args.label}_{timestamp}"
    eval_dir.mkdir(parents=True, exist_ok=False)

    config_payload = {
        "argv": sys.argv,
        "tasks": [item.task_name for item in specs],
        "episodes_per_task": int(args.episodes_per_task),
        "seed": int(args.seed),
        "dex_root": os.fspath(Path(args.dex_root).expanduser().resolve()),
        "isaac_python": os.fspath(Path(args.isaac_python).expanduser().resolve()),
        "training_data_num": int(args.training_data_num),
        "default_checkpoint_num": int(args.default_checkpoint_num),
        "runtime_dir": os.fspath(Path(args.runtime_dir).expanduser().resolve()),
        "vk_icd_json": os.fspath(Path(args.vk_icd_json).expanduser().resolve()) if str(args.vk_icd_json).strip() else "",
        "carbonite_semaphore": os.fspath(Path(args.carbonite_semaphore).expanduser().resolve()) if str(args.carbonite_semaphore).strip() else "",
        "checkpoint_overrides": checkpoint_overrides,
        "env_random_flag": str(args.env_random_flag),
        "garment_random_flag": str(args.garment_random_flag),
        "record_video_flag": str(args.record_video_flag),
        "validation_flag": str(args.validation_flag),
    }
    (eval_dir / "config.json").write_text(json.dumps(config_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    all_task_summaries: list[dict[str, Any]] = []
    overall_counter = Counter()
    for spec in specs:
        rows: list[dict[str, Any]] = []
        print(f"[halo14] task={spec.task_name} stage_count={spec.stage_count}", flush=True)
        for episode_index in range(int(args.episodes_per_task)):
            print(f"[halo14] task={spec.task_name} episode={episode_index:03d}", flush=True)
            row = _run_episode(
                spec=spec,
                episode_index=int(episode_index),
                eval_dir=eval_dir,
                args=args,
                checkpoint_overrides=checkpoint_overrides,
            )
            rows.append(row)
            overall_counter[int(row.get("return_code", 0))] += 1
            if bool(args.fail_fast) and int(row.get("return_code", 0)) != 0:
                print(f"[halo14] fail_fast triggered by task={spec.task_name} episode={episode_index:03d}", flush=True)
                break
            time.sleep(max(0.0, float(args.sleep_s_between_episodes)))
        task_summary = _task_summary(spec=spec, rows=rows)
        (eval_dir / spec.task_name / "task_summary.json").write_text(
            json.dumps(task_summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        all_task_summaries.append(task_summary)
        if bool(args.fail_fast) and any(int(item.get("return_code", 0)) != 0 for item in rows):
            break

    total_known = sum(int(item["episodes"]) - int(item["unknown_successes"]) for item in all_task_summaries)
    total_success = sum(int(item["successes"]) for item in all_task_summaries)
    summary = {
        "label": str(args.label),
        "eval_dir": os.fspath(eval_dir),
        "tasks": all_task_summaries,
        "episodes_total": int(sum(int(item["episodes"]) for item in all_task_summaries)),
        "successes_total": int(total_success),
        "known_success_denominator": int(total_known),
        "known_success_rate": (float(total_success) / float(max(1, total_known))) if all_task_summaries else 0.0,
        "return_codes": {str(key): int(value) for key, value in sorted(overall_counter.items())},
    }
    (eval_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[halo14] eval_dir={os.fspath(eval_dir)}", flush=True)
    print(
        f"[halo14] overall={json.dumps({'episodes_total': summary['episodes_total'], 'successes_total': summary['successes_total'], 'known_success_rate': summary['known_success_rate']}, ensure_ascii=False)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
