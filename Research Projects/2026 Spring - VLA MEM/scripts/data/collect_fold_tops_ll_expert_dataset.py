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
        print("[ll-collect] external env-validation wrapper detected; reusing its SimulationApp patch", flush=True)
        return

    import isaacsim

    original_simulation_app = isaacsim.SimulationApp
    excluded_extensions = ["isaacsim.asset.importer.urdf", "isaacsim.asset.importer.mjcf"]

    class WrappedSimulationApp:
        def __init__(self, launch_config: dict[str, Any] | None = None) -> None:
            config = dict(launch_config or {})
            if bool(config.get("headless", False)):
                config.setdefault("hide_ui", True)
            extra_args = list(config.get("extra_args", []))
            excluded_arg = f"--/app/extensions/excluded={json.dumps(excluded_extensions)}"
            if excluded_arg not in extra_args:
                extra_args.append(excluded_arg)
            config["extra_args"] = extra_args
            self._app = original_simulation_app(config)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._app, name)

    isaacsim.SimulationApp = WrappedSimulationApp


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect myVLA-compatible low-level Fold Tops expert data on top of DexGarmentLab Env_StandAlone."
    )
    parser.add_argument("--dex_root", required=True, help="Path to DexGarmentLab-main/DexGarmentLab-main")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_successes", type=int, default=10)
    parser.add_argument("--max_attempts", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--record_video_flag", action="store_true")
    parser.add_argument("--validation_threshold", type=float, default=0.12)
    return parser.parse_args()


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


def _sample_pose(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray([rng.uniform(-0.1, 0.1), rng.uniform(0.7, 0.9), 0.2], dtype=np.float32)
    ori = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    return pos, ori


def _state_summary_from_phase(phase_key: str, subtask: str) -> str:
    key = str(phase_key).strip()
    if key.startswith("approach_left") or key.startswith("grasp_left") or key.startswith("fold_left") or key.startswith("release_left"):
        return f"fold_stage=left_sleeve; left_sleeve=active; right_sleeve=out; hem=down; next_focus={subtask}"
    if key.startswith("approach_right") or key.startswith("grasp_right") or key.startswith("fold_right") or key.startswith("release_right"):
        return f"fold_stage=right_sleeve; left_sleeve=folded; right_sleeve=active; hem=down; next_focus={subtask}"
    if "hem" in key:
        return f"fold_stage=lower_hem; left_sleeve=folded; right_sleeve=folded; hem=active; next_focus={subtask}"
    return f"fold_stage=inspect; next_focus={subtask}"


def main() -> None:
    args = _parse_args()
    dex_root = Path(str(args.dex_root)).expanduser().resolve()
    output_dir = Path(str(args.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "event": "ll_collect_start",
                "dex_root": os.fspath(dex_root),
                "output_dir": os.fspath(output_dir),
                "target_successes": int(args.target_successes),
                "max_attempts": int(args.max_attempts),
                "seed": int(args.seed),
                "record_video_flag": bool(args.record_video_flag),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    if os.fspath(dex_root) not in sys.path:
        sys.path.insert(0, os.fspath(dex_root))

    _patch_simulation_app_for_headless()
    print("[ll-collect] importing DexGarmentLab env entry", flush=True)

    from tools import myvla_fold_tops_envstandalone_entry as env_entry
    print("[ll-collect] env entry imported", flush=True)

    myvla_root = env_entry._ensure_myvla_imports(dex_root)
    if os.fspath(myvla_root) not in sys.path:
        sys.path.insert(0, os.fspath(myvla_root))

    from policy_prompting import compose_low_level_prompt
    print(f"[ll-collect] myVLA imports ready from {myvla_root}", flush=True)

    rng = random.Random(int(args.seed))
    assets = _load_assets_list(dex_root)
    phase_order = list(env_entry._FOLD_CONTROL_PHASE_SPECS.keys())
    successes = 0
    attempts = 0
    manifest_path = output_dir / "manifest.jsonl"

    with manifest_path.open("w", encoding="utf-8") as manifest:
        while successes < int(args.target_successes) and attempts < int(args.max_attempts):
            attempts += 1
            garment_usd = str(rng.choice(assets))
            pos, ori = _sample_pose(rng)
            env = None
            record_video_flag = bool(args.record_video_flag)
            video_path = None
            try:
                print(
                    json.dumps(
                        {
                            "event": "attempt_start",
                            "attempt": int(attempts),
                            "successes": int(successes),
                            "garment_usd": garment_usd,
                            "pos": pos.tolist(),
                            "ori": ori.tolist(),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                env = env_entry.MyVLAFoldTopsEnv(
                    pos=pos,
                    ori=ori,
                    usd_path=garment_usd,
                    ground_material_usd=None,
                    record_video_flag=record_video_flag,
                )
                print(f"[ll-collect] attempt={attempts} environment created", flush=True)
                initial_obs = env_entry._sample_initial_observation(env)
                print(f"[ll-collect] attempt={attempts} initial observation captured", flush=True)
                if record_video_flag:
                    env.capture_video_frame()

                per_step: dict[str, list[Any]] = {
                    "overview": [],
                    "topdown": [],
                    "left_wrist": [],
                    "right_wrist": [],
                    "left_exterior": [],
                    "right_exterior": [],
                    "left_state": [],
                    "right_state": [],
                    "left_gripper": [],
                    "right_gripper": [],
                    "subtask": [],
                    "control_phase_key": [],
                    "control_phase_name": [],
                    "prompt_left": [],
                    "prompt_right": [],
                    "left_hand_state": [],
                    "right_hand_state": [],
                    "target_left_tcp": [],
                    "target_right_tcp": [],
                    "next_left_state": [],
                    "next_right_state": [],
                }

                left_ori = np.asarray([0.579, -0.579, -0.406, 0.406], dtype=np.float32)
                right_ori = np.asarray([0.406, -0.406, -0.579, 0.579], dtype=np.float32)

                for phase_key in phase_order:
                    duration = int(env_entry._FOLD_CONTROL_PHASE_SPECS[phase_key]["duration"])
                    for local_step in range(duration):
                        obs = env_entry._current_policy_observation(env)
                        phase_left, phase_right, left_hand_state, right_hand_state, phase_name, _, _ = env_entry._control_target_pose(
                            phase_key,
                            local_step,
                        )
                        subtask = phase_name
                        state_summary = _state_summary_from_phase(phase_key, subtask)
                        prompt_left = compose_low_level_prompt(
                            goal="Fold the shirt into a compact square block by folding both sleeves inward and then folding the lower hem upward using two robot arms.",
                            subtask=subtask,
                            language_memory="",
                            arm_side="left",
                            phase_name=phase_name,
                            structured_state_summary=state_summary,
                            retrieved_semantic_hint="",
                            retrieved_visual_hint="",
                        )
                        prompt_right = compose_low_level_prompt(
                            goal="Fold the shirt into a compact square block by folding both sleeves inward and then folding the lower hem upward using two robot arms.",
                            subtask=subtask,
                            language_memory="",
                            arm_side="right",
                            phase_name=phase_name,
                            structured_state_summary=state_summary,
                            retrieved_semantic_hint="",
                            retrieved_visual_hint="",
                        )

                        stabilizer_hand_state = "open"
                        if "close" in {left_hand_state, right_hand_state}:
                            stabilizer_hand_state = "close"
                        elif "pinch" in {left_hand_state, right_hand_state}:
                            stabilizer_hand_state = "pinch"
                        target_left, target_right = env_entry._stabilize_dual_targets(
                            phase_left,
                            phase_right,
                            phase_name=phase_name,
                            hand_state=stabilizer_hand_state,
                        )

                        env.bimanual_dex.set_both_hand_state(
                            left_hand_state=left_hand_state,
                            right_hand_state=right_hand_state,
                        )
                        env.bimanual_dex.dense_move_both_ik(
                            left_pos=target_left,
                            left_ori=left_ori,
                            right_pos=target_right,
                            right_ori=right_ori,
                        )
                        for _ in range(20):
                            env.step()
                        if record_video_flag:
                            env.capture_video_frame()
                        obs_after = env_entry._current_policy_observation(env)

                        per_step["overview"].append(np.asarray(obs["overview"], dtype=np.uint8))
                        per_step["topdown"].append(np.asarray(obs["topdown"], dtype=np.uint8))
                        per_step["left_wrist"].append(np.asarray(obs["left_wrist"], dtype=np.uint8))
                        per_step["right_wrist"].append(np.asarray(obs["right_wrist"], dtype=np.uint8))
                        per_step["left_exterior"].append(np.asarray(obs["left_exterior"], dtype=np.uint8))
                        per_step["right_exterior"].append(np.asarray(obs["right_exterior"], dtype=np.uint8))
                        per_step["left_state"].append(np.asarray(obs["left_arm_state"], dtype=np.float32))
                        per_step["right_state"].append(np.asarray(obs["right_arm_state"], dtype=np.float32))
                        per_step["left_gripper"].append(np.asarray(obs["left_gripper"], dtype=np.float32))
                        per_step["right_gripper"].append(np.asarray(obs["right_gripper"], dtype=np.float32))
                        per_step["subtask"].append(str(subtask))
                        per_step["control_phase_key"].append(str(phase_key))
                        per_step["control_phase_name"].append(str(phase_name))
                        per_step["prompt_left"].append(str(prompt_left))
                        per_step["prompt_right"].append(str(prompt_right))
                        per_step["left_hand_state"].append(str(left_hand_state))
                        per_step["right_hand_state"].append(str(right_hand_state))
                        per_step["target_left_tcp"].append(np.asarray(target_left, dtype=np.float32))
                        per_step["target_right_tcp"].append(np.asarray(target_right, dtype=np.float32))
                        per_step["next_left_state"].append(np.asarray(obs_after["left_arm_state"], dtype=np.float32))
                        per_step["next_right_state"].append(np.asarray(obs_after["right_arm_state"], dtype=np.float32))

                evaluation = env_entry._evaluate_fold_success(
                    env,
                    initial_pcd=np.asarray(initial_obs["garment_point_cloud"], dtype=np.float32),
                    threshold=float(args.validation_threshold),
                )
                success = bool(evaluation.get("success", False))
                print(
                    json.dumps(
                        {
                            "event": "attempt_finish",
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
                if success:
                    attempt_dir.mkdir(parents=True, exist_ok=True)
                    dataset_path = attempt_dir / "episode_ll_expert.npz"
                    np.savez_compressed(
                        dataset_path,
                        overview=np.asarray(per_step["overview"], dtype=np.uint8),
                        topdown=np.asarray(per_step["topdown"], dtype=np.uint8),
                        left_wrist=np.asarray(per_step["left_wrist"], dtype=np.uint8),
                        right_wrist=np.asarray(per_step["right_wrist"], dtype=np.uint8),
                        left_exterior=np.asarray(per_step["left_exterior"], dtype=np.uint8),
                        right_exterior=np.asarray(per_step["right_exterior"], dtype=np.uint8),
                        left_state=np.asarray(per_step["left_state"], dtype=np.float32),
                        right_state=np.asarray(per_step["right_state"], dtype=np.float32),
                        left_gripper=np.asarray(per_step["left_gripper"], dtype=np.float32),
                        right_gripper=np.asarray(per_step["right_gripper"], dtype=np.float32),
                        target_left_tcp=np.asarray(per_step["target_left_tcp"], dtype=np.float32),
                        target_right_tcp=np.asarray(per_step["target_right_tcp"], dtype=np.float32),
                        next_left_state=np.asarray(per_step["next_left_state"], dtype=np.float32),
                        next_right_state=np.asarray(per_step["next_right_state"], dtype=np.float32),
                        subtask=np.asarray(per_step["subtask"], dtype=object),
                        control_phase_key=np.asarray(per_step["control_phase_key"], dtype=object),
                        control_phase_name=np.asarray(per_step["control_phase_name"], dtype=object),
                        prompt_left=np.asarray(per_step["prompt_left"], dtype=object),
                        prompt_right=np.asarray(per_step["prompt_right"], dtype=object),
                        left_hand_state=np.asarray(per_step["left_hand_state"], dtype=object),
                        right_hand_state=np.asarray(per_step["right_hand_state"], dtype=object),
                    )
                    meta = {
                        "attempt": int(attempts),
                        "success_index": int(successes),
                        "success": True,
                        "dataset_path": os.fspath(dataset_path),
                        "usd_path": garment_usd,
                        "pos": pos.tolist(),
                        "ori": ori.tolist(),
                        "steps": int(len(per_step["subtask"])),
                        "evaluation": evaluation,
                    }
                    if record_video_flag:
                        video_dir = attempt_dir / "video"
                        video_dir.mkdir(parents=True, exist_ok=True)
                        video_path = video_dir / "episode.mp4"
                        env.env_camera.create_mp4(os.fspath(video_path))
                        meta["video_path"] = os.fspath(video_path)
                    (attempt_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                    manifest.write(json.dumps(meta, ensure_ascii=False) + "\n")
                    successes += 1
                else:
                    failure_meta = {
                        "attempt": int(attempts),
                        "success": False,
                        "usd_path": garment_usd,
                        "pos": pos.tolist(),
                        "ori": ori.tolist(),
                        "steps": int(len(per_step["subtask"])),
                        "evaluation": evaluation,
                    }
                    manifest.write(json.dumps(failure_meta, ensure_ascii=False) + "\n")
            except Exception as exc:
                print(
                    json.dumps(
                        {
                            "event": "attempt_exception",
                            "attempt": int(attempts),
                            "exc_type": type(exc).__name__,
                            "exc": str(exc),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                raise
            finally:
                if env is not None:
                    try:
                        if env.thread_record is not None and env.thread_record.is_alive():
                            env.env_camera.capture = False
                            env.thread_record.join(timeout=2.0)
                    except Exception:
                        pass

    summary = {
        "ok": True,
        "target_successes": int(args.target_successes),
        "max_attempts": int(args.max_attempts),
        "attempts": int(attempts),
        "successes": int(successes),
        "output_dir": os.fspath(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    env_entry.simulation_app.close()


if __name__ == "__main__":
    main()
