#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import signal
import socket
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_BASE = Path(os.environ.get("MYVLA_SERVER_BASE") or Path(__file__).resolve().parents[3])
DEFAULT_DEX_ROOT = DEFAULT_BASE / "DexGarmentLab-main"
DEFAULT_MYVLA_ROOT = DEFAULT_BASE / "myVLA"
DEFAULT_RPC_CODE_ROOT = DEFAULT_MYVLA_ROOT
DEFAULT_ISAAC_PY = DEFAULT_BASE / "isaac-sim-standalone@4.5.0" / "python.sh"
DEFAULT_LAUNCHER = DEFAULT_MYVLA_ROOT / "scripts" / "server" / "run_myvla_envstandalone_wrapped.py"
DEFAULT_RESULTS_ROOT = DEFAULT_BASE / "eval_results" / "fold_tops_envstandalone_eval"
DEFAULT_CHECKPOINT_DIR = DEFAULT_MYVLA_ROOT / "pi05_droid_pytorch"
DEFAULT_TOKENIZER_MODEL = DEFAULT_MYVLA_ROOT / "assets" / "paligemma_tokenizer.model"
DEFAULT_HL_VLM_DIR = DEFAULT_MYVLA_ROOT / "pretrained_vlm" / "google_paligemma-3b-mix-224-bfloat16"
PROVEN_DEX_MOTION_ENV_DEFAULTS = {
    # Match the older stable Fold Tops rollout profile so the direct 60D
    # controller does not silently fall back to more aggressive robot stepping.
    "DEXGARMENTLAB_ARM_MAX_JOINT_STEP": "0.08",
    "DEXGARMENTLAB_HAND_MAX_JOINT_STEP": "0.12",
    "DEXGARMENTLAB_ARM_MAX_JOINT_ACCEL_STEP": "0.06",
    "DEXGARMENTLAB_HAND_MAX_JOINT_ACCEL_STEP": "0.12",
    "DEXGARMENTLAB_JOINT_WORLD_STEPS": "2",
    "DEXGARMENTLAB_HAND_SETTLE_STEPS": "10",
    "DEXGARMENTLAB_MAX_JOINT_SMOOTH_SUBSTEPS": "24",
}
DEFAULT_GOAL_ONLY_FOLD_TOPS_PROMPT = (
    "Use two robot arms to fold the shirt into a neat compact square. Start by visually aligning and flattening "
    "the garment on the table. Fold the left sleeve inward toward the center of the shirt, then fold the right "
    "sleeve inward toward the center, while keeping the cloth low and controlled. Next, grasp the lower hem, lift "
    "it only as much as needed, and fold the lower part of the shirt upward toward the center or upper body so the "
    "shirt becomes a compact rectangular or square block. Finish by gently pressing and aligning the folded shirt "
    "so the edges look tidy, symmetric, and stable. Use the current visual observation to decide the next local "
    "motion. If the shirt already appears neatly folded with the sleeves tucked in and the lower hem folded up, "
    "stop making large manipulation motions and only keep the folded shirt stable."
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Fold Tops evaluation on DexGarmentLab Env_StandAlone with a myVLA RPC policy."
    )
    parser.add_argument("--model_label", required=True, help="Short label used in run directories and summaries.")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plan_path", default="", help="If set, load/save the sampled episode plan at this path.")
    parser.add_argument(
        "--episode_indices",
        default="",
        help="Optional comma-separated episode indices or ranges (for example: 0,2,4-7).",
    )
    parser.add_argument(
        "--eval_name_suffix",
        default="",
        help="Optional suffix appended to the eval directory name to distinguish parallel shards.",
    )
    parser.add_argument("--results_root", default=os.fspath(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--dex_root", default=os.fspath(DEFAULT_DEX_ROOT))
    parser.add_argument("--myvla_root", default=os.fspath(DEFAULT_MYVLA_ROOT))
    parser.add_argument("--rpc_code_root", default=os.fspath(DEFAULT_RPC_CODE_ROOT))
    parser.add_argument("--isaac_python", default=os.fspath(DEFAULT_ISAAC_PY))
    parser.add_argument("--launcher_py", default=os.fspath(DEFAULT_LAUNCHER))
    parser.add_argument("--entry_script_rel", default="tools/myvla_fold_tops_envstandalone_entry.py")
    parser.add_argument("--rpc_server_rel", default="isaac_sim/policy_rpc_server.py")
    parser.add_argument("--checkpoint_dir", default=os.fspath(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--tokenizer_model", default=os.fspath(DEFAULT_TOKENIZER_MODEL))
    parser.add_argument("--hl_vlm_dir", default=os.fspath(DEFAULT_HL_VLM_DIR))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--launcher_gpu_binding_mode",
        choices=("cuda_visible_devices", "omniverse"),
        default="cuda_visible_devices",
    )
    parser.add_argument("--rpc_host", default="127.0.0.1")
    parser.add_argument("--rpc_port", type=int, default=0, help="0 = auto-pick a free local port.")
    parser.add_argument("--rpc_timeout_s", type=float, default=900.0)
    parser.add_argument("--server_start_timeout_s", type=float, default=900.0)
    parser.add_argument("--episode_timeout_s", type=float, default=5400.0)
    parser.add_argument("--rpc_device", default="cuda:0")
    parser.add_argument("--hl_device", default="cuda:0")
    parser.add_argument("--low_level_prompt_style", choices=("phase_structured", "goal_only"), default="goal_only")
    parser.add_argument(
        "--goal_text",
        default=DEFAULT_GOAL_ONLY_FOLD_TOPS_PROMPT,
    )
    parser.add_argument("--outer_steps", type=int, default=0)
    parser.add_argument("--num_steps", type=int, default=4)
    parser.add_argument("--validation_threshold", type=float, default=0.12)
    parser.add_argument("--keep_videos", action="store_true")
    parser.add_argument("--keep_step_artifacts", action="store_true")
    parser.add_argument("--fail_fast", action="store_true")
    return parser.parse_args()


@dataclass(frozen=True)
class EpisodePlan:
    episode_index: int
    garment_usd: str
    garment_pos_x: float
    garment_pos_y: float
    garment_yaw_deg: float


def _safe_name(text: str) -> str:
    out = []
    for ch in str(text):
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("._")


def _pick_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _parse_episode_indices(spec: str) -> list[int]:
    raw = str(spec).strip()
    if not raw:
        return []
    indices: set[int] = set()
    for token in raw.split(","):
        item = token.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start_idx = int(start_text.strip())
            end_idx = int(end_text.strip())
            if end_idx < start_idx:
                raise ValueError(f"Invalid episode range: {item!r}")
            indices.update(range(start_idx, end_idx + 1))
            continue
        indices.add(int(item))
    return sorted(indices)


def _candidate_dex_roots(dex_root: Path) -> list[Path]:
    raw_candidates = [dex_root, dex_root.parent, dex_root / dex_root.name]
    unique: list[Path] = []
    seen: set[str] = set()
    for item in raw_candidates:
        resolved = item.expanduser().resolve()
        text = os.fspath(resolved)
        if text in seen:
            continue
        seen.add(text)
        unique.append(resolved)
    return unique


def _resolve_dex_root(dex_root: Path, *, entry_script_rel: str) -> Path:
    tool_rel = Path(str(entry_script_rel))
    assets_rel = Path("Model_HALO") / "GAM" / "checkpoints" / "Tops_LongSleeve" / "assets_list.txt"
    best_candidate: Path | None = None
    best_score = -1
    for candidate in _candidate_dex_roots(dex_root):
        score = 0
        if (candidate / tool_rel).is_file():
            score += 2
        if (candidate / assets_rel).is_file():
            score += 2
        if score > best_score:
            best_score = score
            best_candidate = candidate
    if best_candidate is None or best_score < 2:
        raise FileNotFoundError(
            f"Unable to resolve a usable DexGarmentLab root from {dex_root}. "
            f"Expected at least {tool_rel} to exist."
        )
    return best_candidate


def _load_assets_list(dex_root: Path) -> list[str]:
    assets_path = dex_root / "Model_HALO" / "GAM" / "checkpoints" / "Tops_LongSleeve" / "assets_list.txt"
    items = []
    for raw_line in assets_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        asset_path = Path(line)
        if not asset_path.is_absolute():
            asset_path = dex_root / line
        items.append(os.fspath(asset_path))
    if not items:
        raise RuntimeError(f"No garment assets found in {assets_path}")
    return items


def _generate_plan(*, dex_root: Path, episodes: int, seed: int) -> list[EpisodePlan]:
    assets = _load_assets_list(dex_root)
    rng = random.Random(int(seed))
    plan = []
    for episode_index in range(int(episodes)):
        plan.append(
            EpisodePlan(
                episode_index=int(episode_index),
                garment_usd=str(rng.choice(assets)),
                garment_pos_x=float(rng.uniform(-0.1, 0.1)),
                garment_pos_y=float(rng.uniform(0.7, 0.9)),
                garment_yaw_deg=0.0,
            )
        )
    return plan


def _load_or_create_plan(*, dex_root: Path, episodes: int, seed: int, plan_path: Path) -> list[EpisodePlan]:
    if plan_path.is_file():
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
        raw_plan = payload.get("episodes", []) if isinstance(payload, dict) else payload
        return [EpisodePlan(**item) for item in raw_plan]
    plan = _generate_plan(dex_root=dex_root, episodes=episodes, seed=seed)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(
        json.dumps(
            {
                "seed": int(seed),
                "episodes": [asdict(item) for item in plan],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return plan


def _select_plan(plan: list[EpisodePlan], requested_indices: list[int]) -> list[EpisodePlan]:
    if not requested_indices:
        return list(plan)
    by_index = {int(item.episode_index): item for item in plan}
    missing = [idx for idx in requested_indices if idx not in by_index]
    if missing:
        raise ValueError(
            f"Requested episode indices are outside the sampled plan: {missing} "
            f"(available: 0..{max(by_index) if by_index else -1})"
        )
    return [by_index[idx] for idx in requested_indices]


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_last_jsonl(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return None
    return json.loads(lines[-1])


def _spawn_rpc_server(
    *,
    args: argparse.Namespace,
    episode_dir: Path,
    rpc_port: int,
) -> tuple[subprocess.Popen[str], Path, Path]:
    rpc_code_root = Path(args.rpc_code_root).expanduser().resolve()
    isaac_python = Path(args.isaac_python).expanduser().resolve()
    server_py = rpc_code_root / str(args.rpc_server_rel)
    if not server_py.is_file():
        raise FileNotFoundError(f"RPC server script not found: {server_py}")

    state_file = episode_dir / "rpc_server_state.json"
    log_path = episode_dir / "rpc_server.log"
    cmd = [
        os.fspath(isaac_python),
        "-u",
        os.fspath(server_py),
        "--host",
        str(args.rpc_host),
        "--port",
        str(int(rpc_port)),
        "--timeout_s",
        str(float(args.rpc_timeout_s)),
        "--checkpoint_dir",
        os.fspath(Path(args.checkpoint_dir).expanduser().resolve()),
        "--tokenizer_model",
        os.fspath(Path(args.tokenizer_model).expanduser().resolve()),
        "--device",
        str(args.rpc_device),
        "--state_file",
        os.fspath(state_file),
        "--viz_dir",
        os.fspath(episode_dir / "rpc_viz"),
        "--viz_name",
        "rpc_server",
    ]
    hl_vlm_dir = str(args.hl_vlm_dir).strip()
    if hl_vlm_dir:
        cmd += [
            "--hl_vlm_dir",
            os.fspath(Path(hl_vlm_dir).expanduser().resolve()),
            "--hl_device",
            str(args.hl_device),
        ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(int(args.gpu))
    env["PYTHONUNBUFFERED"] = "1"
    env["MYVLA_DISABLE_TORCH_COMPILE"] = "1"
    env["MYVLA_LL_PROMPT_STYLE"] = str(args.low_level_prompt_style)
    log_fh = log_path.open("w", encoding="utf-8", buffering=1)
    proc = subprocess.Popen(
        cmd,
        cwd=os.fspath(rpc_code_root),
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    return proc, state_file, log_path


def _wait_for_server_ready(proc: subprocess.Popen[str], state_file: Path, timeout_s: float) -> dict[str, Any]:
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"RPC server exited early with code {proc.returncode}")
        payload = _load_json_if_exists(state_file)
        if payload and bool(payload.get("ready")):
            return payload
        time.sleep(1.0)
    raise TimeoutError(f"Timed out waiting for RPC server to become ready: {state_file}")


def _stop_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


def _kill_pid_with_timeout(pid: int, *, sig: int, timeout_s: float) -> None:
    try:
        os.kill(int(pid), sig)
    except OSError:
        return
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        if not _pid_is_alive(int(pid)):
            return
        time.sleep(0.2)


def _cleanup_rpc_server(proc: subprocess.Popen[str], state_file: Path) -> None:
    _stop_process(proc)
    payload = _load_json_if_exists(state_file) or {}
    pid = int(payload.get("pid", 0) or 0)
    if pid <= 0 or not _pid_is_alive(pid):
        return
    _kill_pid_with_timeout(pid, sig=signal.SIGTERM, timeout_s=10.0)
    if _pid_is_alive(pid):
        _kill_pid_with_timeout(pid, sig=signal.SIGKILL, timeout_s=5.0)


def _run_episode(
    *,
    args: argparse.Namespace,
    launcher_py: Path,
    dex_root: Path,
    myvla_root: Path,
    isaac_python: Path,
    plan: EpisodePlan,
    episode_dir: Path,
) -> dict[str, Any]:
    episode_dir.mkdir(parents=True, exist_ok=False)
    validation_log = episode_dir / "validation.jsonl"
    final_state_png = episode_dir / "final_state.png"
    launcher_output_path = episode_dir / "launcher_output.log"
    rpc_port = int(args.rpc_port) if int(args.rpc_port) > 0 else _pick_free_port(str(args.rpc_host))
    rpc_proc, state_file, rpc_log = _spawn_rpc_server(args=args, episode_dir=episode_dir, rpc_port=rpc_port)
    server_ready_payload = None

    try:
        server_ready_payload = _wait_for_server_ready(
            rpc_proc,
            state_file,
            timeout_s=float(args.server_start_timeout_s),
        )

        tag = f"{_safe_name(args.model_label)}_s{int(args.seed)}_ep{int(plan.episode_index):03d}"
        cmd = [
            os.fspath(isaac_python),
            os.fspath(launcher_py),
            "--gpu",
            str(int(args.gpu)),
            "--gpu_binding_mode",
            str(args.launcher_gpu_binding_mode),
            "--dex_root",
            os.fspath(dex_root),
            "--runtime_dir",
            os.fspath(myvla_root / "WorldModelDiffusionVlaRuntime"),
            "--script_rel",
            str(args.entry_script_rel),
            "--",
            "--policy_mode",
            "rpc",
            "--rpc_host",
            str(args.rpc_host),
            "--rpc_port",
            str(int(rpc_port)),
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
            "--outer_steps",
            str(int(args.outer_steps)),
            "--num_steps",
            str(int(args.num_steps)),
            "--validation_flag",
            "True",
            "--validation_log",
            os.fspath(validation_log),
            "--validation_threshold",
            f"{float(args.validation_threshold):.6f}",
            "--final_state_png",
            os.fspath(final_state_png),
            "--tag",
            tag,
        ]
        if not bool(args.keep_videos):
            cmd.append("--disable_videos")
        if not bool(args.keep_step_artifacts):
            cmd.append("--disable_step_artifacts")

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["TOKENIZERS_PARALLELISM"] = "false"
        for key, value in PROVEN_DEX_MOTION_ENV_DEFAULTS.items():
            env.setdefault(key, value)
        print(
            "[eval] dex_motion_env="
            + json.dumps({key: env.get(key, "") for key in PROVEN_DEX_MOTION_ENV_DEFAULTS}, ensure_ascii=False),
            flush=True,
        )
        started_at = time.time()
        with launcher_output_path.open("w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(
                cmd,
                cwd=os.fspath(myvla_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            captured_lines: list[str] = []
            assert proc.stdout is not None
            while True:
                line = proc.stdout.readline()
                if line:
                    captured_lines.append(line)
                    log_file.write(line)
                    log_file.flush()
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    continue
                if proc.poll() is not None:
                    break
                if time.time() - started_at > float(args.episode_timeout_s):
                    proc.terminate()
                    try:
                        proc.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=10)
                    raise TimeoutError(
                        f"Timed out waiting for Fold Tops episode {int(plan.episode_index):03d} "
                        f"after {float(args.episode_timeout_s):.1f}s"
                    )
                time.sleep(0.2)
            for line in proc.stdout:
                captured_lines.append(line)
                log_file.write(line)
                log_file.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
            return_code = int(proc.wait())
        elapsed_s = float(time.time() - started_at)

        validation_payload = _load_last_jsonl(validation_log)
        run_root = Path(validation_payload["run_root"]) if validation_payload and validation_payload.get("run_root") else None
        summary_json = _load_json_if_exists(run_root / "summary.json") if run_root else None
        success = bool(validation_payload and validation_payload.get("success"))
        result = {
            "episode_index": int(plan.episode_index),
            "seed": int(args.seed),
            "model_label": str(args.model_label),
            "return_code": int(return_code),
            "elapsed_s": elapsed_s,
            "success": bool(success),
            "run_root": os.fspath(run_root) if run_root else "",
            "video_path": str(validation_payload.get("video_path", "")) if validation_payload else "",
            "validation_result": validation_payload,
            "summary_json": summary_json,
            "plan": asdict(plan),
            "launcher_output_log": os.fspath(launcher_output_path),
            "rpc_server_log": os.fspath(rpc_log),
            "rpc_server_state": server_ready_payload,
            "final_state_png": os.fspath(final_state_png),
        }
        (episode_dir / "episode_summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"[eval] episode={int(plan.episode_index):03d} seed={int(args.seed)} "
            f"success={bool(success)} elapsed_s={elapsed_s:.1f} "
            f"video={result['video_path'] or 'none'}",
            flush=True,
        )
        return result
    finally:
        _cleanup_rpc_server(rpc_proc, state_file)


def _aggregate(results: list[dict[str, Any]], *, args: argparse.Namespace, eval_dir: Path, plan_path: Path) -> dict[str, Any]:
    stop_reasons = Counter()
    successes = 0
    for item in results:
        if item.get("success"):
            successes += 1
        validation_result = item.get("validation_result") or {}
        reason = str(validation_result.get("status", "")).strip()
        if not reason and int(item.get("return_code", 0)) != 0:
            reason = "launcher_nonzero_exit"
        if reason:
            stop_reasons[reason] += 1
    payload = {
        "model_label": str(args.model_label),
        "seed": int(args.seed),
        "episodes": int(len(results)),
        "successes": int(successes),
        "success_rate": (float(successes) / float(len(results))) if results else 0.0,
        "status_counts": dict(stop_reasons),
        "results": results,
        "plan_path": os.fspath(plan_path),
        "eval_dir": os.fspath(eval_dir),
        "rpc_code_root": os.fspath(Path(args.rpc_code_root).expanduser().resolve()),
        "myvla_root": os.fspath(Path(args.myvla_root).expanduser().resolve()),
        "checkpoint_dir": os.fspath(Path(args.checkpoint_dir).expanduser().resolve()),
        "tokenizer_model": os.fspath(Path(args.tokenizer_model).expanduser().resolve()),
        "hl_vlm_dir": os.fspath(Path(args.hl_vlm_dir).expanduser().resolve()) if str(args.hl_vlm_dir).strip() else "",
        "dex_root": os.fspath(Path(args.dex_root).expanduser().resolve()),
        "gpu": int(args.gpu),
        "low_level_prompt_style": str(args.low_level_prompt_style),
    }
    return payload


def main() -> int:
    args = _parse_args()
    dex_root = _resolve_dex_root(
        Path(args.dex_root).expanduser(),
        entry_script_rel=str(args.entry_script_rel),
    )
    myvla_root = Path(args.myvla_root).expanduser().resolve()
    isaac_python = Path(args.isaac_python).expanduser().resolve()
    launcher_py = Path(args.launcher_py).expanduser().resolve()
    results_root = Path(args.results_root).expanduser().resolve()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    eval_name = f"{_safe_name(args.model_label)}_{timestamp}_seed{int(args.seed)}_gpu{int(args.gpu)}"
    suffix = _safe_name(str(args.eval_name_suffix).strip())
    if suffix:
        eval_name = f"{eval_name}_{suffix}"
    eval_dir = results_root / eval_name
    eval_dir.mkdir(parents=True, exist_ok=False)

    plan_path = (
        Path(args.plan_path).expanduser().resolve()
        if str(args.plan_path).strip()
        else (eval_dir / "episode_plan.json")
    )
    plan = _load_or_create_plan(
        dex_root=dex_root,
        episodes=int(args.episodes),
        seed=int(args.seed),
        plan_path=plan_path,
    )
    requested_indices = _parse_episode_indices(args.episode_indices)
    selected_plan = _select_plan(plan, requested_indices)

    config_payload = {
        "argv": sys.argv,
        "model_label": str(args.model_label),
        "episodes": int(args.episodes),
        "seed": int(args.seed),
        "requested_episode_indices": requested_indices,
        "selected_episode_count": int(len(selected_plan)),
        "rpc_code_root": os.fspath(Path(args.rpc_code_root).expanduser().resolve()),
        "myvla_root": os.fspath(myvla_root),
        "dex_root": os.fspath(dex_root),
        "isaac_python": os.fspath(isaac_python),
        "launcher_py": os.fspath(launcher_py),
        "gpu": int(args.gpu),
        "launcher_gpu_binding_mode": str(args.launcher_gpu_binding_mode),
        "low_level_prompt_style": str(args.low_level_prompt_style),
        "checkpoint_dir": os.fspath(Path(args.checkpoint_dir).expanduser().resolve()),
        "tokenizer_model": os.fspath(Path(args.tokenizer_model).expanduser().resolve()),
        "hl_vlm_dir": str(Path(args.hl_vlm_dir).expanduser().resolve()) if str(args.hl_vlm_dir).strip() else "",
        "keep_videos": bool(args.keep_videos),
        "keep_step_artifacts": bool(args.keep_step_artifacts),
    }
    (eval_dir / "config.json").write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    results: list[dict[str, Any]] = []
    for episode in selected_plan:
        print(
            f"[eval] model={args.model_label} seed={int(args.seed)} "
            f"episode={int(episode.episode_index):03d} "
            f"garment={episode.garment_usd} pos=({episode.garment_pos_x:.4f}, {episode.garment_pos_y:.4f})",
            flush=True,
        )
        episode_dir = eval_dir / f"episode_{int(episode.episode_index):03d}"
        episode_result = _run_episode(
            args=args,
            launcher_py=launcher_py,
            dex_root=dex_root,
            myvla_root=myvla_root,
            isaac_python=isaac_python,
            plan=episode,
            episode_dir=episode_dir,
        )
        results.append(episode_result)
        if bool(args.fail_fast) and (int(episode_result.get("return_code", 0)) != 0 or not bool(episode_result.get("success"))):
            print(f"[eval] fail_fast triggered at episode={int(episode.episode_index):03d}", flush=True)
            break

    summary = _aggregate(results, args=args, eval_dir=eval_dir, plan_path=plan_path)
    (eval_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"[eval] summary={json.dumps({'successes': summary['successes'], 'episodes': summary['episodes'], 'success_rate': summary['success_rate']}, ensure_ascii=False)}",
        flush=True,
    )
    print(f"[eval] eval_dir={os.fspath(eval_dir)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
