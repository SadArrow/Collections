from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _patch_simulation_app_for_headless() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("DEXGARMENTLAB_HEADLESS", "1")
    os.environ.setdefault("OPEN3D_DISABLE_WEB_VISUALIZER", "true")
    if os.environ.get("MYVLA_ENV_VALIDATION_WRAPPER_ACTIVE", "").strip().lower() in {"1", "true", "yes"}:
        print("[dexshadow-collect] external env-validation wrapper detected; reusing its SimulationApp patch", flush=True)
        return

    import isaacsim

    original_simulation_app = isaacsim.SimulationApp
    excluded_extensions = ["isaacsim.asset.importer.urdf", "isaacsim.asset.importer.mjcf"]
    active_gpu = str(os.environ.get("DEXGARMENTLAB_ACTIVE_GPU", "")).strip()
    physics_gpu = str(os.environ.get("DEXGARMENTLAB_PHYSICS_GPU", "")).strip()
    multi_gpu = str(os.environ.get("DEXGARMENTLAB_MULTI_GPU", "0")).strip().lower() in ("1", "true", "yes", "on")

    class WrappedSimulationApp:
        def __init__(self, launch_config: dict[str, Any] | None = None) -> None:
            config = dict(launch_config or {})
            if multi_gpu:
                config["multi_gpu"] = True
            if bool(config.get("headless", False)):
                config.setdefault("hide_ui", True)
            extra_args = list(config.get("extra_args", []))
            excluded_arg = f"--/app/extensions/excluded={json.dumps(excluded_extensions)}"
            if excluded_arg not in extra_args:
                extra_args.append(excluded_arg)
            if active_gpu:
                active_arg = f"--/renderer/activeGpu={int(active_gpu)}"
                if active_arg not in extra_args:
                    extra_args.append(active_arg)
            if physics_gpu:
                physics_arg = f"--/physics/cudaDevice={int(physics_gpu)}"
                if physics_arg not in extra_args:
                    extra_args.append(physics_arg)
            config["extra_args"] = extra_args
            self._app = original_simulation_app(config)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._app, name)

    isaacsim.SimulationApp = WrappedSimulationApp


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Fold Tops expert data in DexGarmentLab native 6+24 joint space for pi0.5 fine-tuning."
    )
    parser.add_argument("--dex_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--active_gpu", type=int, default=-1)
    parser.add_argument("--physics_gpu", type=int, default=-1)
    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument("--keep_cuda_visible_devices", action="store_true")
    parser.add_argument("--target_successes", type=int, default=10)
    parser.add_argument("--max_attempts", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--record_video_flag", action="store_true")
    parser.add_argument("--validation_threshold", type=float, default=0.12)
    parser.add_argument("--goal", default="")
    parser.add_argument("--prompt_style", choices=("phase_structured", "goal_only"), default="goal_only")
    return parser.parse_args()


def _configure_isaac_gpu_env(args: argparse.Namespace) -> None:
    os.environ.setdefault("DEXGARMENTLAB_HEADLESS", "1")
    if not bool(args.keep_cuda_visible_devices):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    if int(args.active_gpu) >= 0:
        os.environ["DEXGARMENTLAB_ACTIVE_GPU"] = str(int(args.active_gpu))
    if int(args.physics_gpu) >= 0:
        os.environ["DEXGARMENTLAB_PHYSICS_GPU"] = str(int(args.physics_gpu))
    if bool(args.multi_gpu):
        os.environ["DEXGARMENTLAB_MULTI_GPU"] = "1"


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
        items.append(os.fspath(asset_path.resolve()))
    if not items:
        raise RuntimeError(f"No garment assets found in {assets_path}")
    return items


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


def _resolve_dex_root(dex_root: Path) -> Path:
    tool_rel = Path("tools") / "myvla_fold_tops_envstandalone_entry.py"
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


def _sample_pose(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray([rng.uniform(-0.1, 0.1), rng.uniform(0.7, 0.9), 0.2], dtype=np.float32)
    ori = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    return pos, ori


def _full_joint_state(robot) -> np.ndarray:
    return np.asarray(robot.get_joint_positions(), dtype=np.float32).reshape(-1)


def _resize_square(entry_mod: Any, image: Any, *, size: int = 224) -> np.ndarray:
    rgb = entry_mod._ensure_rgb(image)
    from PIL import Image

    return np.asarray(Image.fromarray(rgb).resize((int(size), int(size)), Image.Resampling.BILINEAR), dtype=np.uint8)


def _pad_crop(entry_mod: Any, image: Any, *, center_x: float, center_y: float, span_px: int, output_size: int = 224) -> np.ndarray:
    rgb = entry_mod._ensure_rgb(image)
    h, w, _ = rgb.shape
    span = max(16, int(span_px))
    cx = float(np.clip(center_x, 0.0, max(0.0, float(w - 1))))
    cy = float(np.clip(center_y, 0.0, max(0.0, float(h - 1))))
    x0 = int(round(cx - 0.5 * span))
    y0 = int(round(cy - 0.5 * span))
    x1 = x0 + span
    y1 = y0 + span

    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - w)
    pad_bottom = max(0, y1 - h)
    if pad_left or pad_top or pad_right or pad_bottom:
        rgb = np.pad(
            rgb,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="edge",
        )
        x0 += pad_left
        x1 += pad_left
        y0 += pad_top
        y1 += pad_top
    crop = rgb[y0:y1, x0:x1]
    return _resize_square(entry_mod, crop, size=output_size)


def _topdown_bounds(env: Any) -> tuple[float, float, float, float]:
    left_base = np.asarray(env.bimanual_dex.dexleft.get_world_pose()[0], dtype=np.float32).reshape(3)
    right_base = np.asarray(env.bimanual_dex.dexright.get_world_pose()[0], dtype=np.float32).reshape(3)
    garment = np.asarray(env.position, dtype=np.float32).reshape(3)
    x_min = min(float(left_base[0]), float(right_base[0]), float(garment[0]) - 0.55) - 0.15
    x_max = max(float(left_base[0]), float(right_base[0]), float(garment[0]) + 0.55) + 0.15
    y_min = min(float(left_base[1]), float(right_base[1]), float(garment[1]) - 0.45) - 0.05
    y_max = max(float(left_base[1]), float(right_base[1]), float(garment[1]) + 0.35) + 0.20
    if x_max <= x_min:
        x_min, x_max = -1.0, 1.0
    if y_max <= y_min:
        y_min, y_max = 0.0, 1.4
    return x_min, x_max, y_min, y_max


def _world_xy_to_topdown_pixel(entry_mod: Any, env: Any, image: Any, world_xy: np.ndarray) -> tuple[float, float]:
    rgb = entry_mod._ensure_rgb(image)
    h, w, _ = rgb.shape
    x_min, x_max, y_min, y_max = _topdown_bounds(env)
    x = float(world_xy[0])
    y = float(world_xy[1])
    u = (x - x_min) / max(1.0e-6, (x_max - x_min))
    v = (y_max - y) / max(1.0e-6, (y_max - y_min))
    return float(np.clip(u, 0.0, 1.0) * float(w - 1)), float(np.clip(v, 0.0, 1.0) * float(h - 1))


def _build_local_views(entry_mod: Any, env: Any, *, topdown: Any) -> dict[str, np.ndarray]:
    topdown_rgb = entry_mod._ensure_rgb(topdown)
    h, w, _ = topdown_rgb.shape
    garment_xy = np.asarray(env.position[:2], dtype=np.float32).reshape(2)
    left_ee = np.asarray(env.bimanual_dex.dexleft.end_effector.get_world_pose()[0], dtype=np.float32).reshape(3)
    right_ee = np.asarray(env.bimanual_dex.dexright.end_effector.get_world_pose()[0], dtype=np.float32).reshape(3)

    left_mid_xy = 0.5 * (left_ee[:2] + garment_xy)
    right_mid_xy = 0.5 * (right_ee[:2] + garment_xy)

    left_mid_px = _world_xy_to_topdown_pixel(entry_mod, env, topdown_rgb, left_mid_xy)
    right_mid_px = _world_xy_to_topdown_pixel(entry_mod, env, topdown_rgb, right_mid_xy)
    left_ee_px = _world_xy_to_topdown_pixel(entry_mod, env, topdown_rgb, left_ee[:2])
    right_ee_px = _world_xy_to_topdown_pixel(entry_mod, env, topdown_rgb, right_ee[:2])

    wide_span = int(round(min(h, w) * 0.46))
    tight_span = int(round(min(h, w) * 0.24))

    return {
        "left_exterior": _pad_crop(entry_mod, topdown_rgb, center_x=left_mid_px[0], center_y=left_mid_px[1], span_px=wide_span),
        "right_exterior": _pad_crop(entry_mod, topdown_rgb, center_x=right_mid_px[0], center_y=right_mid_px[1], span_px=wide_span),
        "left_wrist": _pad_crop(entry_mod, topdown_rgb, center_x=left_ee_px[0], center_y=left_ee_px[1], span_px=tight_span),
        "right_wrist": _pad_crop(entry_mod, topdown_rgb, center_x=right_ee_px[0], center_y=right_ee_px[1], span_px=tight_span),
    }


def _gripper_scalar_from_hand_state(hand_state: str) -> float:
    return 1.0 if str(hand_state).strip().lower() == "open" else 0.0


def _current_dexshadow_observation(entry_mod: Any, env: Any, *, left_hand_state: str, right_hand_state: str) -> dict[str, Any]:
    overview = entry_mod._capture_rgb(env.env_camera, env)
    topdown = entry_mod._capture_rgb(env.garment_camera, env)
    local_views = _build_local_views(entry_mod, env, topdown=topdown)

    left_joints_all = np.asarray(env.bimanual_dex.dexleft.get_joint_positions(), dtype=np.float32).reshape(-1)
    right_joints_all = np.asarray(env.bimanual_dex.dexright.get_joint_positions(), dtype=np.float32).reshape(-1)
    left_arm = left_joints_all[np.asarray(env.bimanual_dex.dexleft.arm_dof_indices, dtype=np.int32)]
    right_arm = right_joints_all[np.asarray(env.bimanual_dex.dexright.arm_dof_indices, dtype=np.int32)]

    return {
        "overview": overview,
        "topdown": topdown,
        "left_exterior": local_views["left_exterior"],
        "right_exterior": local_views["right_exterior"],
        "left_wrist": local_views["left_wrist"],
        "right_wrist": local_views["right_wrist"],
        "left_full_state": left_joints_all.astype(np.float32),
        "right_full_state": right_joints_all.astype(np.float32),
        "left_arm_state": np.concatenate([left_arm, np.zeros(1, dtype=np.float32)], axis=0).astype(np.float32),
        "right_arm_state": np.concatenate([right_arm, np.zeros(1, dtype=np.float32)], axis=0).astype(np.float32),
        "left_gripper": np.asarray([_gripper_scalar_from_hand_state(left_hand_state)], dtype=np.float32),
        "right_gripper": np.asarray([_gripper_scalar_from_hand_state(right_hand_state)], dtype=np.float32),
    }


def main() -> None:
    args = _parse_args()
    requested_dex_root = Path(str(args.dex_root)).expanduser().resolve()
    dex_root = _resolve_dex_root(requested_dex_root)
    output_dir = Path(str(args.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_isaac_gpu_env(args)

    if os.fspath(dex_root) not in sys.path:
        sys.path.insert(0, os.fspath(dex_root))
    tools_root = dex_root / "tools"
    if os.fspath(tools_root) not in sys.path:
        sys.path.insert(0, os.fspath(tools_root))

    _patch_simulation_app_for_headless()

    import myvla_fold_tops_envstandalone_entry as env_entry

    myvla_root = env_entry._ensure_myvla_imports(dex_root)
    if os.fspath(myvla_root) not in sys.path:
        sys.path.insert(0, os.fspath(myvla_root))

    from policy_prompting import (
        compose_bimanual_low_level_prompt,
        compose_low_level_prompt,
        default_goal_only_fold_tops_prompt,
    )

    goal_text = str(args.goal).strip() or default_goal_only_fold_tops_prompt()

    rng = random.Random(int(args.seed))
    assets = _load_assets_list(dex_root)
    phase_order = list(env_entry._FOLD_CONTROL_PHASE_SPECS.keys())
    manifest_path = output_dir / "manifest.jsonl"
    successes = 0
    attempts = 0

    print(
        json.dumps(
                {
                    "event": "dexshadow_collect_start",
                    "requested_dex_root": os.fspath(requested_dex_root),
                    "resolved_dex_root": os.fspath(dex_root),
                    "output_dir": os.fspath(output_dir),
                    "active_gpu": int(args.active_gpu),
                    "physics_gpu": int(args.physics_gpu),
                    "multi_gpu": bool(args.multi_gpu),
                    "cuda_visible_devices": str(os.environ.get("CUDA_VISIBLE_DEVICES", "")),
                    "target_successes": int(args.target_successes),
                    "max_attempts": int(args.max_attempts),
                    "prompt_style": str(args.prompt_style),
                    "goal": goal_text,
                },
            ensure_ascii=False,
        ),
        flush=True,
    )

    with manifest_path.open("w", encoding="utf-8") as manifest:
        while successes < int(args.target_successes) and attempts < int(args.max_attempts):
            attempts += 1
            garment_usd = str(rng.choice(assets))
            pos, ori = _sample_pose(rng)
            env = None
            try:
                env = env_entry.MyVLAFoldTopsEnv(
                    pos=pos,
                    ori=ori,
                    usd_path=garment_usd,
                    ground_material_usd=None,
                    record_video_flag=bool(args.record_video_flag),
                )
                initial_obs = env_entry._sample_initial_observation(env)
                if bool(args.record_video_flag):
                    env.capture_video_frame()

                left_hand_state = "open"
                right_hand_state = "open"
                per_step: dict[str, list[Any]] = {
                    "overview": [],
                    "topdown": [],
                    "left_exterior": [],
                    "right_exterior": [],
                    "left_wrist": [],
                    "right_wrist": [],
                    "left_state_full": [],
                    "right_state_full": [],
                    "next_left_state_full": [],
                    "next_right_state_full": [],
                    "prompt_left": [],
                    "prompt_right": [],
                    "prompt_bimanual": [],
                    "subtask": [],
                    "control_phase_key": [],
                    "control_phase_name": [],
                    "left_hand_state": [],
                    "right_hand_state": [],
                }

                left_ori = np.asarray([0.579, -0.579, -0.406, 0.406], dtype=np.float32)
                right_ori = np.asarray([0.406, -0.406, -0.579, 0.579], dtype=np.float32)

                for phase_key in phase_order:
                    duration = int(env_entry._FOLD_CONTROL_PHASE_SPECS[phase_key]["duration"])
                    for local_step in range(duration):
                        obs = _current_dexshadow_observation(
                            env_entry,
                            env,
                            left_hand_state=left_hand_state,
                            right_hand_state=right_hand_state,
                        )
                        phase_left, phase_right, target_left_hand_state, target_right_hand_state, phase_name, _, _ = env_entry._control_target_pose(
                            phase_key,
                            local_step,
                        )
                        phase_alpha = 1.0 if int(duration) <= 1 else float(local_step) / float(max(1, int(duration) - 1))
                        phase_left, phase_right = env_entry._refine_phase_targets(
                            env,
                            control_phase_key=phase_key,
                            alpha=float(phase_alpha),
                            left_target=phase_left,
                            right_target=phase_right,
                        )
                        stabilizer_hand_state = "open"
                        if "close" in {target_left_hand_state, target_right_hand_state}:
                            stabilizer_hand_state = "close"
                        elif "pinch" in {target_left_hand_state, target_right_hand_state}:
                            stabilizer_hand_state = "pinch"
                        target_left, target_right = env_entry._stabilize_dual_targets(
                            phase_left,
                            phase_right,
                            phase_name=phase_name,
                            hand_state=stabilizer_hand_state,
                        )
                        prompt_left = compose_low_level_prompt(
                            goal=goal_text,
                            subtask=phase_name,
                            language_memory="",
                            arm_side="left",
                            phase_name=phase_name,
                            structured_state_summary=f"current_hand={left_hand_state}",
                            retrieved_semantic_hint="",
                            retrieved_visual_hint="",
                            prompt_style=str(args.prompt_style),
                        )
                        prompt_right = compose_low_level_prompt(
                            goal=goal_text,
                            subtask=phase_name,
                            language_memory="",
                            arm_side="right",
                            phase_name=phase_name,
                            structured_state_summary=f"current_hand={right_hand_state}",
                            retrieved_semantic_hint="",
                            retrieved_visual_hint="",
                            prompt_style=str(args.prompt_style),
                        )
                        prompt_bimanual = compose_bimanual_low_level_prompt(
                            goal=goal_text,
                            subtask=phase_name,
                            language_memory="",
                            phase_name=phase_name,
                            structured_state_summary=(
                                f"left_hand={left_hand_state}; right_hand={right_hand_state}"
                            ),
                            retrieved_semantic_hint="",
                            retrieved_visual_hint="",
                            prompt_style=str(args.prompt_style),
                        )

                        current_left = _full_joint_state(env.bimanual_dex.dexleft)
                        current_right = _full_joint_state(env.bimanual_dex.dexright)

                        env.bimanual_dex.set_both_hand_state(
                            left_hand_state=target_left_hand_state,
                            right_hand_state=target_right_hand_state,
                        )
                        env.bimanual_dex.dense_move_both_ik(
                            left_pos=target_left,
                            left_ori=left_ori,
                            right_pos=target_right,
                            right_ori=right_ori,
                        )
                        for _ in range(int(env_entry._phase_post_move_steps(phase_key))):
                            env.step()
                        if bool(args.record_video_flag):
                            env.capture_video_frame()

                        next_left = _full_joint_state(env.bimanual_dex.dexleft)
                        next_right = _full_joint_state(env.bimanual_dex.dexright)

                        per_step["overview"].append(np.asarray(obs["overview"], dtype=np.uint8))
                        per_step["topdown"].append(np.asarray(obs["topdown"], dtype=np.uint8))
                        per_step["left_exterior"].append(np.asarray(obs["left_exterior"], dtype=np.uint8))
                        per_step["right_exterior"].append(np.asarray(obs["right_exterior"], dtype=np.uint8))
                        per_step["left_wrist"].append(np.asarray(obs["left_wrist"], dtype=np.uint8))
                        per_step["right_wrist"].append(np.asarray(obs["right_wrist"], dtype=np.uint8))
                        per_step["left_state_full"].append(current_left)
                        per_step["right_state_full"].append(current_right)
                        per_step["next_left_state_full"].append(next_left)
                        per_step["next_right_state_full"].append(next_right)
                        per_step["prompt_left"].append(str(prompt_left))
                        per_step["prompt_right"].append(str(prompt_right))
                        per_step["prompt_bimanual"].append(str(prompt_bimanual))
                        per_step["subtask"].append(str(phase_name))
                        per_step["control_phase_key"].append(str(phase_key))
                        per_step["control_phase_name"].append(str(phase_name))
                        per_step["left_hand_state"].append(str(target_left_hand_state))
                        per_step["right_hand_state"].append(str(target_right_hand_state))

                        left_hand_state = str(target_left_hand_state)
                        right_hand_state = str(target_right_hand_state)

                evaluation = env_entry._evaluate_fold_success(
                    env,
                    initial_pcd=np.asarray(initial_obs["garment_point_cloud"], dtype=np.float32),
                    threshold=float(args.validation_threshold),
                )
                success = bool(evaluation.get("success", False))
                print(
                    json.dumps(
                        {
                            "event": "dexshadow_attempt_finish",
                            "attempt": int(attempts),
                            "success": bool(success),
                            "steps": int(len(per_step["subtask"])),
                            "evaluation": evaluation,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

                attempt_dir = output_dir / f"attempt_{attempts:04d}"
                meta = {
                    "attempt": int(attempts),
                    "success": bool(success),
                    "dataset_path": "",
                    "usd_path": garment_usd,
                    "pos": pos.tolist(),
                    "ori": ori.tolist(),
                    "steps": int(len(per_step["subtask"])),
                    "evaluation": evaluation,
                    "prompt_style": str(args.prompt_style),
                    "goal": goal_text,
                    "observation_source": "dex_garment_camera_local_views",
                }
                if success:
                    attempt_dir.mkdir(parents=True, exist_ok=True)
                    dataset_path = attempt_dir / "episode_ll_dexshadow.npz"
                    np.savez_compressed(
                        dataset_path,
                        overview=np.asarray(per_step["overview"], dtype=np.uint8),
                        topdown=np.asarray(per_step["topdown"], dtype=np.uint8),
                        left_exterior=np.asarray(per_step["left_exterior"], dtype=np.uint8),
                        right_exterior=np.asarray(per_step["right_exterior"], dtype=np.uint8),
                        left_wrist=np.asarray(per_step["left_wrist"], dtype=np.uint8),
                        right_wrist=np.asarray(per_step["right_wrist"], dtype=np.uint8),
                        left_state_full=np.asarray(per_step["left_state_full"], dtype=np.float32),
                        right_state_full=np.asarray(per_step["right_state_full"], dtype=np.float32),
                        next_left_state_full=np.asarray(per_step["next_left_state_full"], dtype=np.float32),
                        next_right_state_full=np.asarray(per_step["next_right_state_full"], dtype=np.float32),
                        prompt_left=np.asarray(per_step["prompt_left"], dtype=object),
                        prompt_right=np.asarray(per_step["prompt_right"], dtype=object),
                        prompt_bimanual=np.asarray(per_step["prompt_bimanual"], dtype=object),
                        subtask=np.asarray(per_step["subtask"], dtype=object),
                        control_phase_key=np.asarray(per_step["control_phase_key"], dtype=object),
                        control_phase_name=np.asarray(per_step["control_phase_name"], dtype=object),
                        left_hand_state=np.asarray(per_step["left_hand_state"], dtype=object),
                        right_hand_state=np.asarray(per_step["right_hand_state"], dtype=object),
                    )
                    meta["dataset_path"] = os.fspath(dataset_path)
                    if bool(args.record_video_flag):
                        video_dir = attempt_dir / "video"
                        video_dir.mkdir(parents=True, exist_ok=True)
                        video_path = video_dir / "episode.mp4"
                        env.env_camera.create_mp4(os.fspath(video_path))
                        meta["video_path"] = os.fspath(video_path)
                    (attempt_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                    successes += 1
                manifest.write(json.dumps(meta, ensure_ascii=False) + "\n")
            except Exception as exc:
                failure = {
                    "attempt": int(attempts),
                    "success": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "usd_path": garment_usd,
                    "pos": pos.tolist(),
                    "ori": ori.tolist(),
                }
                manifest.write(json.dumps(failure, ensure_ascii=False) + "\n")
                print(json.dumps({"event": "dexshadow_attempt_error", **failure}, ensure_ascii=False), flush=True)
            finally:
                if env is not None:
                    try:
                        if getattr(env, "thread_record", None) is not None:
                            env.env_camera.capture = False
                            env.thread_record.join(timeout=5.0)
                    except Exception:
                        pass

    stats = {
        "ok": True,
        "episodes": int(successes),
        "attempts": int(attempts),
        "manifest": os.fspath(manifest_path),
        "prompt_style": str(args.prompt_style),
        "goal": goal_text,
    }
    (output_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False), flush=True)
    if os.environ.get("MYVLA_FORCE_PROCESS_EXIT_AFTER_COLLECT", "").strip().lower() in {"1", "true", "yes", "on"}:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
