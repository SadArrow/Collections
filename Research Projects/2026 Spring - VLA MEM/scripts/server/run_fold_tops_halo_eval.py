#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


DEFAULT_BASE = Path(
    os.environ.get("MYVLA_SERVER_BASE")
    or Path(__file__).resolve().parents[3]
)
DEFAULT_DEX_ROOT = DEFAULT_BASE / "DexGarmentLab-main"
DEFAULT_MYVLA_ROOT = DEFAULT_BASE / "myVLA"
DEFAULT_LAUNCHER = DEFAULT_MYVLA_ROOT / "scripts" / "server" / "run_world_model_diffusion_vla_square_fold_isaac45_autowait.sh"
DEFAULT_RESULTS_ROOT = DEFAULT_BASE / "eval_results" / "fold_tops_halo"
DEFAULT_RUNTIME_DIR = DEFAULT_MYVLA_ROOT / "WorldModelDiffusionVlaRuntime"
DEFAULT_CHECKPOINT_DIR = DEFAULT_MYVLA_ROOT / "pi05_droid_pytorch"
DEFAULT_TOKENIZER_MODEL = DEFAULT_MYVLA_ROOT / "assets" / "paligemma_tokenizer.model"
DEFAULT_HL_VLM_DIR = DEFAULT_MYVLA_ROOT / "pretrained_vlm" / "google_paligemma-3b-mix-224-bfloat16"
DEFAULT_VIZ_DIR = DEFAULT_DEX_ROOT / "server_viz"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Fold Tops HALO-style eval for a given myVLA code root.")
    parser.add_argument("--model_label", required=True, help="Short label used in run directories and summaries.")
    parser.add_argument("--episodes", type=int, default=1)
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
    parser.add_argument("--wait_interval_s", type=int, default=60)
    parser.add_argument("--idle_confirm_polls", type=int, default=3)
    parser.add_argument("--preferred_gpus", default="")
    parser.add_argument("--rollout_timeout_s", type=int, default=21600)
    parser.add_argument("--max_abs_joint_position", type=float, default=20.0)
    parser.add_argument("--implausible_joint_delta_factor", type=float, default=10.0)
    parser.add_argument("--capture_samples", type=int, default=3)
    parser.add_argument("--capture_rt_subframes", type=int, default=16)
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
        if isinstance(payload, dict):
            raw_plan = payload.get("episodes", [])
        else:
            raw_plan = payload
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


def _extract_keyed_value(text: str, key: str) -> str:
    prefix = f"{key}="
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if prefix in line:
            return line.split(prefix, 1)[1].strip()
    return ""


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_args_file(
    *,
    path: Path,
    plan: EpisodePlan,
    validation_log: Path,
    final_state_png: Path,
    validation_threshold: float,
    keep_videos: bool,
    keep_step_artifacts: bool,
) -> None:
    items = [
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
        f"{float(validation_threshold):.6f}",
        "--final_state_png",
        os.fspath(final_state_png),
    ]
    if not keep_step_artifacts:
        items.append("--disable_step_artifacts")
    if not keep_videos:
        items.append("--disable_videos")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(items) + "\n", encoding="utf-8")


def _run_episode(
    *,
    launcher: Path,
    episode_dir: Path,
    model_label: str,
    plan: EpisodePlan,
    args: argparse.Namespace,
) -> dict[str, Any]:
    validation_log = episode_dir / "validation.jsonl"
    final_state_png = episode_dir / "final_state.png"
    args_file = episode_dir / "rollout_args.txt"
    launcher_output_path = episode_dir / "launcher_output.log"
    _write_args_file(
        path=args_file,
        plan=plan,
        validation_log=validation_log,
        final_state_png=final_state_png,
        validation_threshold=float(args.validation_threshold),
        keep_videos=bool(args.keep_videos),
        keep_step_artifacts=bool(args.keep_step_artifacts),
    )

    env = os.environ.copy()
    env.update(
        {
            "MYVLA_ROOT": os.fspath(Path(args.myvla_root)),
            "DEX_ROOT": os.fspath(Path(args.dex_root)),
            "RUNTIME_DIR": os.fspath(Path(args.runtime_dir)),
            "CHECKPOINT_DIR": os.fspath(Path(args.checkpoint_dir)),
            "TOKENIZER_MODEL": os.fspath(Path(args.tokenizer_model)),
            "HL_VLM_DIR": os.fspath(Path(args.hl_vlm_dir)),
            "VIZ_DIR": os.fspath(Path(args.viz_dir)),
            "GOAL_TEXT": str(args.goal_text),
            "OUTER_STEPS": str(int(args.outer_steps)),
            "MIN_OUTER_STEPS": str(int(args.min_outer_steps)),
            "NUM_STEPS": str(int(args.num_steps)),
            "ROLLOUT_TIMEOUT_S": str(int(args.rollout_timeout_s)),
            "ROLLOUT_ENABLE_CLEAR_VIDEOS": "1" if bool(args.keep_videos) else "0",
            "ROLLOUT_EXTRA_ARGS_FILE": os.fspath(args_file),
            "RUN_PREFIX": "FoldTopsHALOEval",
            "RUN_SUFFIX": f"{model_label}_ep{int(plan.episode_index):03d}",
            "WAIT_INTERVAL_S": str(int(args.wait_interval_s)),
            "IDLE_CONFIRM_POLLS": str(int(args.idle_confirm_polls)),
            "PREFERRED_GPUS": str(args.preferred_gpus).strip(),
            "CAPTURE_SAMPLES": str(int(args.capture_samples)),
            "CAPTURE_RT_SUBFRAMES": str(int(args.capture_rt_subframes)),
            "DEXGARMENTLAB_MAX_ABS_JOINT_POSITION": f"{float(args.max_abs_joint_position):.6f}",
            "DEXGARMENTLAB_IMPLAUSIBLE_JOINT_DELTA_FACTOR": f"{float(args.implausible_joint_delta_factor):.6f}",
        }
    )

    started_at = time.time()
    with launcher_output_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            [os.fspath(launcher)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        captured_lines: list[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            captured_lines.append(line)
            log_file.write(line)
            log_file.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
        return_code = int(proc.wait())
    elapsed_s = float(time.time() - started_at)
    launcher_output = "".join(captured_lines)
    run_dir_text = _extract_keyed_value(launcher_output, "run_dir")
    run_dir = Path(run_dir_text) if run_dir_text else None
    validation_result = _load_json_if_exists(run_dir / "validation_result.json") if run_dir else None
    summary_json = _load_json_if_exists(run_dir / "summary.json") if run_dir else None
    run_status_json = _load_json_if_exists(run_dir / "run_status.json") if run_dir else None
    error_json = _load_json_if_exists(run_dir / "error.json") if run_dir else None
    success = bool(validation_result and validation_result.get("success"))
    stop_reason = ""
    if summary_json is not None:
        stop_reason = str(summary_json.get("stop_reason", "")).strip()
    if not stop_reason and run_status_json is not None:
        stop_reason = str(run_status_json.get("stop_reason", "")).strip()
    if not stop_reason and error_json is not None:
        stop_reason = str(error_json.get("stop_reason", "")).strip()
    if not stop_reason and return_code != 0:
        stop_reason = "launcher_nonzero_exit"
    result = {
        "episode_index": int(plan.episode_index),
        "model_label": str(model_label),
        "return_code": int(return_code),
        "elapsed_s": elapsed_s,
        "success": bool(success),
        "stop_reason": stop_reason,
        "run_dir": os.fspath(run_dir) if run_dir else "",
        "validation_result": validation_result,
        "summary_json": summary_json,
        "run_status_json": run_status_json,
        "error_json": error_json,
        "plan": asdict(plan),
        "launcher_output_log": os.fspath(launcher_output_path),
        "validation_log": os.fspath(validation_log),
        "final_state_png": os.fspath(final_state_png),
    }
    (episode_dir / "episode_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _aggregate(results: list[dict[str, Any]], *, args: argparse.Namespace, eval_dir: Path, plan_path: Path) -> dict[str, Any]:
    stop_reasons = Counter()
    failure_reasons = Counter()
    successes = 0
    for item in results:
        if item.get("success"):
            successes += 1
        stop_reason = str(item.get("stop_reason", "")).strip()
        if stop_reason:
            stop_reasons[stop_reason] += 1
        if not item.get("success"):
            reason = stop_reason or "unspecified_failure"
            failure_reasons[reason] += 1
    payload = {
        "model_label": str(args.model_label),
        "episodes": int(len(results)),
        "successes": int(successes),
        "success_rate": (float(successes) / float(len(results))) if results else 0.0,
        "stop_reasons": dict(stop_reasons),
        "failure_reasons": dict(failure_reasons),
        "results": results,
        "plan_path": os.fspath(plan_path),
        "eval_dir": os.fspath(eval_dir),
        "myvla_root": os.fspath(Path(args.myvla_root)),
        "checkpoint_dir": os.fspath(Path(args.checkpoint_dir)),
        "tokenizer_model": os.fspath(Path(args.tokenizer_model)),
        "hl_vlm_dir": os.fspath(Path(args.hl_vlm_dir)),
        "runtime_dir": os.fspath(Path(args.runtime_dir)),
        "dex_root": os.fspath(Path(args.dex_root)),
        "viz_dir": os.fspath(Path(args.viz_dir)),
        "seed": int(args.seed),
    }
    return payload


def main() -> int:
    args = _parse_args()
    launcher = Path(args.launcher).expanduser().resolve()
    dex_root = Path(args.dex_root).expanduser().resolve()
    results_root = Path(args.results_root).expanduser().resolve()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    eval_name = f"{args.model_label}_{timestamp}"
    suffix = _safe_name(str(args.eval_name_suffix).strip())
    if suffix:
        eval_name = f"{eval_name}_{suffix}"
    eval_dir = results_root / eval_name
    eval_dir.mkdir(parents=True, exist_ok=False)
    plan_path = Path(args.plan_path).expanduser().resolve() if str(args.plan_path).strip() else (eval_dir / "episode_plan.json")
    plan = _load_or_create_plan(dex_root=dex_root, episodes=int(args.episodes), seed=int(args.seed), plan_path=plan_path)
    requested_indices = _parse_episode_indices(args.episode_indices)
    selected_plan = _select_plan(plan, requested_indices)

    config_payload = {
        "argv": sys.argv,
        "model_label": str(args.model_label),
        "episodes": int(args.episodes),
        "seed": int(args.seed),
        "plan_path": os.fspath(plan_path),
        "requested_episode_indices": requested_indices,
        "selected_episode_count": int(len(selected_plan)),
        "myvla_root": os.fspath(Path(args.myvla_root).expanduser().resolve()),
        "dex_root": os.fspath(dex_root),
        "launcher": os.fspath(launcher),
        "runtime_dir": os.fspath(Path(args.runtime_dir).expanduser().resolve()),
        "checkpoint_dir": os.fspath(Path(args.checkpoint_dir).expanduser().resolve()),
        "tokenizer_model": os.fspath(Path(args.tokenizer_model).expanduser().resolve()),
        "hl_vlm_dir": os.fspath(Path(args.hl_vlm_dir).expanduser().resolve()),
        "viz_dir": os.fspath(Path(args.viz_dir).expanduser().resolve()),
        "keep_videos": bool(args.keep_videos),
        "keep_step_artifacts": bool(args.keep_step_artifacts),
    }
    (eval_dir / "config.json").write_text(json.dumps(config_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    results: list[dict[str, Any]] = []
    for episode in selected_plan:
        episode_dir = eval_dir / f"episode_{int(episode.episode_index):03d}"
        episode_dir.mkdir(parents=True, exist_ok=False)
        print(
            f"[eval] model={args.model_label} episode={int(episode.episode_index):03d} "
            f"garment={episode.garment_usd} pos=({episode.garment_pos_x:.4f}, {episode.garment_pos_y:.4f})",
            flush=True,
        )
        episode_result = _run_episode(
            launcher=launcher,
            episode_dir=episode_dir,
            model_label=str(args.model_label),
            plan=episode,
            args=args,
        )
        results.append(episode_result)
        if bool(args.fail_fast) and (int(episode_result.get("return_code", 0)) != 0 or not bool(episode_result.get("success"))):
            print(f"[eval] fail_fast triggered at episode={int(episode.episode_index):03d}", flush=True)
            break

    summary = _aggregate(results, args=args, eval_dir=eval_dir, plan_path=plan_path)
    summary["requested_episode_indices"] = requested_indices
    summary["selected_episode_count"] = int(len(selected_plan))
    (eval_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[eval] summary={json.dumps({'successes': summary['successes'], 'episodes': summary['episodes'], 'success_rate': summary['success_rate']}, ensure_ascii=False)}", flush=True)
    print(f"[eval] eval_dir={os.fspath(eval_dir)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
