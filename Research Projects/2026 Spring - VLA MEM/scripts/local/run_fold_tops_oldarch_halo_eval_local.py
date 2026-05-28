#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import socket
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description="Run local Fold_Tops HALO-style evaluation for the old myVLA architecture."
    )
    parser.add_argument(
        "--old_myvla_root",
        default=os.fspath(repo_root / "artifacts" / "myvla_arch_backup_20260401_202638" / "myVLA"),
    )
    parser.add_argument(
        "--dex_root",
        default=os.fspath(repo_root / "DexGarmentLab-main" / "DexGarmentLab-main"),
    )
    parser.add_argument(
        "--isaac_python",
        default=r"E:\isaacsim_4.5.0\python.bat",
    )
    parser.add_argument(
        "--checkpoint_dir",
        default=os.fspath(repo_root / "myVLA" / "pi05_droid_pytorch"),
    )
    parser.add_argument(
        "--tokenizer_model",
        default=os.fspath(repo_root / "myVLA" / "assets" / "paligemma_tokenizer.model"),
    )
    parser.add_argument(
        "--hl_vlm_dir",
        default=os.fspath(repo_root / "myVLA" / "pretrained_vlm" / "google_paligemma-3b-mix-224-bfloat16"),
    )
    parser.add_argument("--hl_device", default="cpu")
    parser.add_argument("--rpc_device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0, help="0 = auto pick free port")
    parser.add_argument("--episodes_per_seed", type=int, default=50)
    parser.add_argument("--seeds", default="0,1,2", help="Comma-separated random seeds")
    parser.add_argument(
        "--results_root",
        default=os.fspath(repo_root / "eval_results" / "oldarch_fold_tops_halo_local"),
    )
    parser.add_argument("--label", default="oldarch_fold_tops")
    parser.add_argument(
        "--goal_text",
        default="Fold the shirt into a compact square block by folding both sleeves inward and then folding the lower hem upward using two robot arms.",
    )
    parser.add_argument("--validation_threshold", type=float, default=0.12)
    parser.add_argument("--num_steps", type=int, default=1)
    parser.add_argument("--min_outer_steps", type=int, default=50)
    parser.add_argument("--outer_steps", type=int, default=0)
    parser.add_argument("--active_gpu", type=int, default=0)
    parser.add_argument("--physics_gpu", type=int, default=0)
    parser.add_argument("--camera_width", type=int, default=640)
    parser.add_argument("--camera_height", type=int, default=480)
    parser.add_argument("--capture_samples", type=int, default=1)
    parser.add_argument("--capture_rt_subframes", type=int, default=1)
    parser.add_argument("--capture_median_filter", type=int, default=1)
    parser.add_argument("--capture_nlm_strength", type=float, default=0.0)
    parser.add_argument("--capture_sharpen_amount", type=float, default=0.0)
    parser.add_argument("--capture_warmup_renders", type=int, default=1)
    parser.add_argument("--rpc_timeout_s", type=float, default=300.0)
    parser.add_argument("--server_start_timeout_s", type=float, default=900.0)
    parser.add_argument("--episode_timeout_s", type=float, default=1800.0)
    parser.add_argument(
        "--resume_eval_dir",
        default="",
        help="Resume an existing evaluation directory instead of creating a new one.",
    )
    parser.add_argument(
        "--max_episodes_this_run",
        type=int,
        default=0,
        help="Optional cap on how many pending episodes to execute in this invocation across all seeds. 0 = run all pending.",
    )
    parser.add_argument(
        "--demo_episodes_per_seed",
        type=int,
        default=1,
        help="How many episodes per seed keep videos/step artifacts for demo review.",
    )
    parser.add_argument(
        "--keep_all_episode_artifacts",
        action="store_true",
        help="Keep videos and step artifacts for every executed episode in this run.",
    )
    parser.add_argument("--export_clear_videos", action="store_true")
    parser.add_argument("--keep_failed_demos", action="store_true")
    parser.add_argument("--fail_fast", action="store_true")
    return parser.parse_args()


@dataclass(frozen=True)
class EpisodePlan:
    seed: int
    episode_index: int
    garment_usd: str
    garment_pos_x: float
    garment_pos_y: float
    garment_yaw_deg: float


def _pick_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _load_assets_list(dex_root: Path) -> list[str]:
    assets_path = dex_root / "Model_HALO" / "GAM" / "checkpoints" / "Tops_LongSleeve" / "assets_list.txt"
    items: list[str] = []
    for raw_line in assets_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        asset_path = Path(line)
        if not asset_path.is_absolute():
            asset_path = dex_root / line
        items.append(os.fspath(asset_path.resolve()))
    if not items:
        raise RuntimeError(f"No Fold_Tops garment assets found in {assets_path}")
    return items


def _generate_plan(*, dex_root: Path, seed: int, episodes_per_seed: int) -> list[EpisodePlan]:
    assets = _load_assets_list(dex_root)
    rng = random.Random(int(seed))
    plan: list[EpisodePlan] = []
    for episode_index in range(int(episodes_per_seed)):
        plan.append(
            EpisodePlan(
                seed=int(seed),
                episode_index=int(episode_index),
                garment_usd=str(rng.choice(assets)),
                garment_pos_x=float(rng.uniform(-0.1, 0.1)),
                garment_pos_y=float(rng.uniform(0.7, 0.9)),
                garment_yaw_deg=0.0,
            )
        )
    return plan


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _spawn_rpc_server(
    *,
    args: argparse.Namespace,
    results_dir: Path,
    port: int,
) -> tuple[subprocess.Popen[str], Path]:
    old_myvla_root = Path(args.old_myvla_root).expanduser().resolve()
    state_file = results_dir / "rpc_server_state.json"
    server_py = old_myvla_root / "isaac_sim" / "policy_rpc_server.py"
    if not server_py.is_file():
        raise FileNotFoundError(f"policy_rpc_server.py not found: {server_py}")

    cmd = [
        sys.executable,
        os.fspath(server_py),
        "--host",
        str(args.host),
        "--port",
        str(int(port)),
        "--checkpoint_dir",
        os.fspath(Path(args.checkpoint_dir).expanduser().resolve()),
        "--tokenizer_model",
        os.fspath(Path(args.tokenizer_model).expanduser().resolve()),
        "--device",
        str(args.rpc_device),
        "--timeout_s",
        str(float(args.rpc_timeout_s)),
        "--state_file",
        os.fspath(state_file),
    ]
    hl_vlm_dir = str(args.hl_vlm_dir).strip()
    if hl_vlm_dir:
        cmd += [
            "--hl_vlm_dir",
            os.fspath(Path(hl_vlm_dir).expanduser().resolve()),
            "--hl_device",
            str(args.hl_device),
        ]

    stdout_path = results_dir / "rpc_server_stdout.log"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_fh = stdout_path.open("w", encoding="utf-8", buffering=1)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    proc = subprocess.Popen(
        cmd,
        cwd=os.fspath(old_myvla_root),
        stdout=stdout_fh,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    return proc, state_file


def _wait_for_server_ready(proc: subprocess.Popen[str], state_file: Path, timeout_s: float) -> dict[str, Any]:
    started = time.time()
    while True:
        if proc.poll() is not None:
            raise RuntimeError(f"RPC server exited early with code {proc.returncode}")
        if state_file.is_file():
            payload = json.loads(state_file.read_text(encoding="utf-8"))
            if bool(payload.get("ready")):
                return payload
        if time.time() - started > float(timeout_s):
            raise TimeoutError(f"Timed out waiting for RPC server to become ready: {state_file}")
        time.sleep(1.0)


def _stop_server(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def _run_episode(
    *,
    args: argparse.Namespace,
    dex_root: Path,
    isaac_python: Path,
    plan: EpisodePlan,
    port: int,
    seed_dir: Path,
) -> dict[str, Any]:
    episode_dir = seed_dir / f"episode_{int(plan.episode_index):03d}"
    episode_dir.mkdir(parents=True, exist_ok=False)
    rollout_dir = episode_dir / "rollout"
    final_state_png = episode_dir / "final_state.png"
    validation_log = episode_dir / "validation.jsonl"
    stdout_log = episode_dir / "stdout.log"

    keep_demo_artifacts = bool(args.keep_all_episode_artifacts) or (
        int(plan.episode_index) < int(args.demo_episodes_per_seed)
    )
    cmd = [
        os.fspath(isaac_python),
        os.fspath(dex_root / "tools" / "run_myvla_fold_tops_demo.py"),
        "--headless",
        "--policy_mode",
        "rpc",
        "--rpc_host",
        str(args.host),
        "--rpc_port",
        str(int(port)),
        "--rpc_timeout_s",
        str(float(args.rpc_timeout_s)),
        "--goal",
        str(args.goal_text),
        "--garment_usd",
        str(plan.garment_usd),
        "--garment_pos_x",
        f"{float(plan.garment_pos_x):.6f}",
        "--garment_pos_y",
        f"{float(plan.garment_pos_y):.6f}",
        "--garment_yaw_deg",
        f"{float(plan.garment_yaw_deg):.6f}",
        "--validation_flag",
        "--validation_log",
        os.fspath(validation_log),
        "--validation_threshold",
        f"{float(args.validation_threshold):.6f}",
        "--final_state_png",
        os.fspath(final_state_png),
        "--viz_dir",
        os.fspath(episode_dir),
        "--viz_name",
        "rollout",
        "--active_gpu",
        str(int(args.active_gpu)),
        "--physics_gpu",
        str(int(args.physics_gpu)),
        "--camera_width",
        str(int(args.camera_width)),
        "--camera_height",
        str(int(args.camera_height)),
        "--capture_samples",
        str(int(args.capture_samples)),
        "--capture_rt_subframes",
        str(int(args.capture_rt_subframes)),
        "--capture_median_filter",
        str(int(args.capture_median_filter)),
        "--capture_nlm_strength",
        f"{float(args.capture_nlm_strength):.6f}",
        "--capture_sharpen_amount",
        f"{float(args.capture_sharpen_amount):.6f}",
        "--capture_warmup_renders",
        str(int(args.capture_warmup_renders)),
        "--num_steps",
        str(int(args.num_steps)),
        "--min_outer_steps",
        str(int(args.min_outer_steps)),
        "--outer_steps",
        str(int(args.outer_steps)),
    ]
    if not keep_demo_artifacts:
        cmd += ["--disable_step_artifacts", "--disable_videos"]
    elif bool(args.export_clear_videos):
        cmd += ["--export_clear_videos"]

    started = time.time()
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    with stdout_log.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(
            cmd,
            cwd=os.fspath(dex_root),
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            timeout=float(args.episode_timeout_s),
            check=False,
        )
    elapsed_s = float(time.time() - started)

    validation_result = None
    validation_path = rollout_dir / "validation_result.json"
    if validation_path.is_file():
        validation_result = json.loads(validation_path.read_text(encoding="utf-8"))

    summary_json = None
    summary_path = rollout_dir / "summary.json"
    if summary_path.is_file():
        summary_json = json.loads(summary_path.read_text(encoding="utf-8"))

    run_status_json = None
    run_status_path = rollout_dir / "run_status.json"
    if run_status_path.is_file():
        run_status_json = json.loads(run_status_path.read_text(encoding="utf-8"))

    success = bool(validation_result and validation_result.get("success"))
    stop_reason = ""
    if validation_result is not None:
        stop_reason = str(validation_result.get("stop_reason", "")).strip()
    if not stop_reason and summary_json is not None:
        stop_reason = str(summary_json.get("stop_reason", "")).strip()
    if not stop_reason and run_status_json is not None:
        stop_reason = str(run_status_json.get("stop_reason", "")).strip()
    if not stop_reason and proc.returncode != 0:
        stop_reason = "nonzero_exit"

    videos = sorted(str(path.name) for path in rollout_dir.glob("*.mp4")) if rollout_dir.is_dir() else []
    record = {
        "seed": int(plan.seed),
        "episode_index": int(plan.episode_index),
        "plan": asdict(plan),
        "success": bool(success),
        "return_code": int(proc.returncode),
        "elapsed_s": elapsed_s,
        "stop_reason": stop_reason,
        "episode_dir": os.fspath(episode_dir),
        "rollout_dir": os.fspath(rollout_dir),
        "stdout_log": os.fspath(stdout_log),
        "validation_result": validation_result,
        "summary_json": summary_json,
        "run_status_json": run_status_json,
        "videos": videos,
        "final_state_png": os.fspath(final_state_png) if final_state_png.is_file() else "",
    }
    _write_json(episode_dir / "episode_summary.json", record)

    if not keep_demo_artifacts and not bool(args.keep_failed_demos):
        for video_name in videos:
            video_path = rollout_dir / video_name
            if video_path.is_file():
                video_path.unlink()

    return record


def _load_or_create_seed_plan(
    *,
    dex_root: Path,
    seed_dir: Path,
    seed: int,
    episodes_per_seed: int,
) -> list[EpisodePlan]:
    plan_path = seed_dir / "episode_plan.json"
    if plan_path.is_file():
        raw_plan = _load_json(plan_path)
        return [EpisodePlan(**item) for item in raw_plan]

    plan = _generate_plan(dex_root=dex_root, seed=int(seed), episodes_per_seed=int(episodes_per_seed))
    _write_json(plan_path, [asdict(item) for item in plan])
    return plan


def _load_existing_episode_rows(seed_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode_summary in sorted(seed_dir.glob("episode_*/episode_summary.json")):
        try:
            rows.append(_load_json(episode_summary))
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "seed": -1,
                    "episode_index": -1,
                    "success": False,
                    "return_code": -1,
                    "elapsed_s": 0.0,
                    "stop_reason": f"corrupt_episode_summary:{type(exc).__name__}",
                    "episode_summary_path": os.fspath(episode_summary),
                    "error_message": str(exc),
                }
            )
    rows.sort(key=lambda row: int(row.get("episode_index", -1)))
    return rows


def _seed_summary(seed: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = int(sum(1 for row in rows if bool(row.get("success"))))
    total = int(len(rows))
    success_rate = float(successes) / float(total) if total else 0.0
    return {
        "seed": int(seed),
        "episodes": total,
        "successes": successes,
        "success_rate": success_rate,
        "results": rows,
    }


def _summarize(
    eval_dir: Path,
    *,
    args: argparse.Namespace,
    all_rows: dict[int, list[dict[str, Any]]],
    expected_episodes_per_seed: int,
) -> dict[str, Any]:
    seed_rows = []
    per_seed_rates = []
    demo_candidates: list[dict[str, Any]] = []
    completed_episodes = 0
    completed_successes = 0
    for seed in sorted(all_rows):
        summary = _seed_summary(seed, all_rows[seed])
        seed_rows.append(summary)
        per_seed_rates.append(float(summary["success_rate"]))
        completed_episodes += int(summary["episodes"])
        completed_successes += int(summary["successes"])
        for row in summary["results"]:
            if row.get("videos") and row.get("success"):
                demo_candidates.append(row)

    mean_rate = statistics.mean(per_seed_rates) if per_seed_rates else 0.0
    std_rate = statistics.stdev(per_seed_rates) if len(per_seed_rates) >= 2 else 0.0
    expected_total = int(expected_episodes_per_seed) * int(len(all_rows))
    all_finished = all(int(summary["episodes"]) >= int(expected_episodes_per_seed) for summary in seed_rows)
    completed_rate = float(completed_successes) / float(completed_episodes) if completed_episodes else 0.0

    demo_dir = eval_dir / "demos"
    copied_demos: list[str] = []
    if demo_candidates:
        demo_dir.mkdir(parents=True, exist_ok=True)
        for row in demo_candidates[: max(1, int(args.demo_episodes_per_seed))]:
            rollout_dir = Path(row["rollout_dir"])
            video_names = list(row.get("videos") or [])
            if not video_names:
                continue
            src = rollout_dir / video_names[0]
            if not src.is_file():
                continue
            dst = demo_dir / f"seed{int(row['seed']):02d}_ep{int(row['episode_index']):03d}_{src.name}"
            shutil.copy2(src, dst)
            copied_demos.append(os.fspath(dst))

    payload = {
        "task_name": "Fold_Tops",
        "protocol": {
            "episodes_per_seed": int(args.episodes_per_seed),
            "seeds": [int(token) for token in str(args.seeds).split(",") if token.strip()],
            "report": "success_rate mean ± std across seeds",
            "paper_reference": "DexGarmentLab, Experiment 6.1: 50 episodes with three different seeds; Mean ± Std",
        },
        "model_variant": "old_myvla_architecture_backup_20260401_202638",
        "old_myvla_root": os.fspath(Path(args.old_myvla_root).expanduser().resolve()),
        "checkpoint_dir": os.fspath(Path(args.checkpoint_dir).expanduser().resolve()),
        "hl_vlm_dir": os.fspath(Path(args.hl_vlm_dir).expanduser().resolve()) if str(args.hl_vlm_dir).strip() else "",
        "results_root": os.fspath(eval_dir),
        "all_finished": all_finished,
        "expected_episodes_per_seed": int(expected_episodes_per_seed),
        "expected_total_episodes": expected_total,
        "completed_episodes": completed_episodes,
        "remaining_episodes": max(0, expected_total - completed_episodes),
        "completed_successes": completed_successes,
        "completed_success_rate": completed_rate,
        "mean_success_rate": mean_rate,
        "std_success_rate": std_rate,
        "per_seed_success_rate": per_seed_rates,
        "seed_summaries": seed_rows,
        "demo_videos": copied_demos,
    }
    _write_json(eval_dir / "benchmark_summary.json", payload)
    _write_json(eval_dir / "progress_summary.json", payload)
    return payload


def main() -> int:
    args = _parse_args()
    dex_root = Path(args.dex_root).expanduser().resolve()
    isaac_python = Path(args.isaac_python).expanduser().resolve()
    if not isaac_python.is_file():
        raise FileNotFoundError(f"Isaac python not found: {isaac_python}")
    if not dex_root.is_dir():
        raise FileNotFoundError(f"DexGarmentLab root not found: {dex_root}")

    seed_values = [int(token.strip()) for token in str(args.seeds).split(",") if token.strip()]
    if len(seed_values) != 3:
        raise ValueError(f"Expected exactly three seeds to match the paper protocol, got: {seed_values}")

    results_root = Path(args.results_root).expanduser().resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    if str(args.resume_eval_dir).strip():
        eval_dir = Path(args.resume_eval_dir).expanduser().resolve()
        if not eval_dir.is_dir():
            raise FileNotFoundError(f"resume_eval_dir does not exist: {eval_dir}")
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        eval_dir = results_root / f"{args.label}_{timestamp}"
        eval_dir.mkdir(parents=True, exist_ok=False)

    _write_json(eval_dir / "run_config.json", vars(args))

    all_rows: dict[int, list[dict[str, Any]]] = {}
    pending_items: list[tuple[Path, EpisodePlan]] = []
    for seed in seed_values:
        seed_dir = eval_dir / f"seed_{int(seed):02d}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        plan = _load_or_create_seed_plan(
            dex_root=dex_root,
            seed_dir=seed_dir,
            seed=int(seed),
            episodes_per_seed=int(args.episodes_per_seed),
        )
        rows = _load_existing_episode_rows(seed_dir)
        all_rows[int(seed)] = rows
        completed_indices = {
            int(row.get("episode_index", -1))
            for row in rows
            if int(row.get("episode_index", -1)) >= 0
        }
        for item in plan:
            if int(item.episode_index) not in completed_indices:
                pending_items.append((seed_dir, item))

    if int(args.max_episodes_this_run) > 0:
        pending_items = pending_items[: int(args.max_episodes_this_run)]

    if not pending_items:
        summary = _summarize(
            eval_dir,
            args=args,
            all_rows=all_rows,
            expected_episodes_per_seed=int(args.episodes_per_seed),
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    port = int(args.port) if int(args.port) > 0 else _pick_free_port(str(args.host))
    server_proc, state_file = _spawn_rpc_server(args=args, results_dir=eval_dir, port=port)
    try:
        server_state = _wait_for_server_ready(server_proc, state_file, float(args.server_start_timeout_s))
        _write_json(eval_dir / "rpc_server_ready.json", server_state)

        for seed_dir, item in pending_items:
            rows = all_rows[int(item.seed)]
            try:
                record = _run_episode(
                    args=args,
                    dex_root=dex_root,
                    isaac_python=isaac_python,
                    plan=item,
                    port=int(port),
                    seed_dir=seed_dir,
                )
            except subprocess.TimeoutExpired:
                record = {
                    "seed": int(item.seed),
                    "episode_index": int(item.episode_index),
                    "plan": asdict(item),
                    "success": False,
                    "return_code": -9,
                    "elapsed_s": float(args.episode_timeout_s),
                    "stop_reason": "timeout",
                }
                _write_json(seed_dir / f"episode_{int(item.episode_index):03d}" / "episode_summary.json", record)
                if bool(args.fail_fast):
                    raise
            except Exception as exc:  # noqa: BLE001
                record = {
                    "seed": int(item.seed),
                    "episode_index": int(item.episode_index),
                    "plan": asdict(item),
                    "success": False,
                    "return_code": -1,
                    "elapsed_s": 0.0,
                    "stop_reason": f"runner_exception:{type(exc).__name__}",
                    "error_message": str(exc),
                }
                episode_dir = seed_dir / f"episode_{int(item.episode_index):03d}"
                episode_dir.mkdir(parents=True, exist_ok=True)
                _write_json(episode_dir / "episode_summary.json", record)
                if bool(args.fail_fast):
                    raise

            rows.append(record)
            rows.sort(key=lambda row: int(row.get("episode_index", -1)))
            episode_progress = _summarize(
                eval_dir,
                args=args,
                all_rows=all_rows,
                expected_episodes_per_seed=int(args.episodes_per_seed),
            )
            print(
                json.dumps(
                    {
                        "event": "episode_finished",
                        "seed": int(item.seed),
                        "episode_index": int(item.episode_index),
                        "success": bool(record.get("success")),
                        "stop_reason": str(record.get("stop_reason", "")),
                        "elapsed_s": float(record.get("elapsed_s", 0.0)),
                        "completed_episodes": int(episode_progress["completed_episodes"]),
                        "remaining_episodes": int(episode_progress["remaining_episodes"]),
                    },
                    ensure_ascii=False,
                )
            )

        summary = _summarize(
            eval_dir,
            args=args,
            all_rows=all_rows,
            expected_episodes_per_seed=int(args.episodes_per_seed),
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    finally:
        _stop_server(server_proc)


if __name__ == "__main__":
    raise SystemExit(main())
