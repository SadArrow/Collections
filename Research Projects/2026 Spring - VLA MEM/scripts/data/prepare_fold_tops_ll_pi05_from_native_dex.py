from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if os.fspath(REPO_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPO_ROOT))

from isaac_sim.policy_prompting import compose_bimanual_low_level_prompt, default_goal_only_fold_tops_prompt


DEFAULT_GOAL = default_goal_only_fold_tops_prompt()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert DexGarmentLab native Fold Tops expert trajectories into pi0.5 dual-arm 60D SFT format."
    )
    parser.add_argument(
        "--artifact_dirs",
        nargs="+",
        required=True,
        help="One or more artifact dirs that contain manifest.jsonl from Dex native expert collection.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--goal", default="")
    parser.add_argument("--prompt_style", choices=("phase_structured", "goal_only"), default="goal_only")
    parser.add_argument("--max_episodes", type=int, default=0, help="0 = use all successful episodes")
    return parser.parse_args()


def _ensure_rgb(image: Any) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected RGB image, got shape={arr.shape}")
    if np.issubdtype(arr.dtype, np.floating):
        arr_f = arr.astype(np.float32)
        if arr_f.size and float(arr_f.max()) > 2.0:
            arr_f = arr_f / 255.0
        arr = (np.clip(arr_f, 0.0, 1.0) * 255.0).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return arr


def _center_crop(image: Any, *, frac: float) -> np.ndarray:
    rgb = _ensure_rgb(image)
    h, w, _ = rgb.shape
    frac = float(np.clip(frac, 0.1, 1.0))
    crop_w = max(8, int(round(w * frac)))
    crop_h = max(8, int(round(h * frac)))
    x0 = max(0, (w - crop_w) // 2)
    y0 = max(0, (h - crop_h) // 2)
    return rgb[y0 : y0 + crop_h, x0 : x0 + crop_w]


def _half_view(image: Any, *, side: str) -> np.ndarray:
    rgb = _ensure_rgb(image)
    _, width, _ = rgb.shape
    if str(side) == "left":
        return rgb[:, : max(1, width // 2), :]
    return rgb[:, max(0, width // 2) :, :]


def _raw_stage_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for stage_index in (1, 2, 3):
        stage_key = f"stage_{stage_index}"
        if stage_key not in payload:
            continue
        stage_records = list(np.asarray(payload[stage_key], dtype=object))
        total = len(stage_records)
        for local_index, item in enumerate(stage_records):
            merged.append(
                {
                    "stage_index": int(stage_index),
                    "local_index": int(local_index),
                    "stage_total": int(total),
                    "record": dict(item),
                }
            )
    return merged


def _phase_for_progress(stage_index: int, progress: float) -> tuple[str, str]:
    p = float(np.clip(progress, 0.0, 1.0))
    if int(stage_index) == 1:
        if p < 0.22:
            return ("approach_left_sleeve", "approach left sleeve")
        if p < 0.38:
            return ("grasp_left_sleeve", "grasp left sleeve cuff")
        if p < 0.82:
            return ("fold_left_sleeve", "fold left sleeve inward")
        return ("release_left_sleeve", "release left sleeve and retreat")
    if int(stage_index) == 2:
        if p < 0.22:
            return ("approach_right_sleeve", "approach right sleeve")
        if p < 0.38:
            return ("grasp_right_sleeve", "grasp right sleeve cuff")
        if p < 0.82:
            return ("fold_right_sleeve", "fold right sleeve inward")
        return ("release_right_sleeve", "release right sleeve and retreat")
    if p < 0.15:
        return ("approach_lower_hem", "approach lower hem corners")
    if p < 0.28:
        return ("grasp_lower_hem", "grasp lower hem corners")
    if p < 0.42:
        return ("lift_lower_hem", "lift lower hem slightly")
    if p < 0.64:
        return ("bring_lower_hem_to_center", "bring lower hem toward center")
    if p < 0.80:
        return ("lay_lower_hem_flat", "lay lower hem flat near center seam")
    if p < 0.90:
        return ("release_lower_hem", "release lower hem and lift clear")
    if p < 0.96:
        return ("flatten_square", "gently sweep side flaps inward and flatten")
    return ("inspect_finish", "inspect square fold and finish")


def _structured_state_summary(stage_index: int, progress: float, phase_name: str) -> str:
    p = float(np.clip(progress, 0.0, 1.0))
    fold_stage = "left_sleeve" if int(stage_index) == 1 else ("right_sleeve" if int(stage_index) == 2 else "lower_hem")
    left_sleeve = "out"
    right_sleeve = "out"
    hem = "down"
    shape = "spread"
    stability = "moving"
    last_effect = ""
    if int(stage_index) == 1:
        left_sleeve = "approached" if p < 0.38 else ("grasped" if p < 0.50 else "folded")
        last_effect = "left sleeve moved inward"
    elif int(stage_index) == 2:
        left_sleeve = "folded"
        right_sleeve = "approached" if p < 0.38 else ("grasped" if p < 0.50 else "folded")
        shape = "partially_folded"
        last_effect = "right sleeve moved inward"
    else:
        left_sleeve = "folded"
        right_sleeve = "folded"
        shape = "rectangular" if p < 0.85 else "compact"
        last_effect = "lower hem moved toward center"
        if p < 0.28:
            hem = "down"
        elif p < 0.42:
            hem = "grasped"
        elif p < 0.64:
            hem = "lifting"
        elif p < 0.80:
            hem = "centering"
        elif p < 0.90:
            hem = "laid"
        else:
            hem = "flattened"
    if int(stage_index) == 3 and p >= 0.94:
        shape = "square"
        stability = "stable"
    elif p >= 0.88:
        stability = "settling"
    return (
        f"fold_stage={fold_stage}; left_sleeve={left_sleeve}; right_sleeve={right_sleeve}; "
        f"hem={hem}; shape={shape}; stability={stability}; last_effect={last_effect}; next_focus={phase_name}"
    )


def _iter_success_records(artifact_dirs: list[Path], *, max_episodes: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for artifact_dir in artifact_dirs:
        manifest_path = artifact_dir / "manifest.jsonl"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"manifest not found: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if not bool(item.get("success", False)):
                    continue
                copied_dataset = str(item.get("copied_dataset", "")).strip()
                if not copied_dataset:
                    copied_train_files = [str(path).strip() for path in item.get("copied_train_files", []) if str(path).strip()]
                    if copied_train_files:
                        copied_dataset = str(copied_train_files[-1]).strip()
                if not copied_dataset:
                    continue
                rows.append(
                    {
                        "artifact_dir": os.fspath(artifact_dir),
                        "copied_dataset": copied_dataset,
                        "copied_video": str(item.get("copied_video", "")).strip(),
                        "copied_final_state": str(item.get("copied_final_state", "")).strip(),
                        "attempt": int(item.get("attempt", 0)),
                        "gpu": int(item.get("gpu", -1)),
                        "source_meta": item,
                    }
                )
                if int(max_episodes) > 0 and len(rows) >= int(max_episodes):
                    return rows
    return rows


def _convert_episode(source_npz: Path, *, goal: str, prompt_style: str) -> dict[str, np.ndarray]:
    payload = np.load(source_npz, allow_pickle=True)
    stage_records = _raw_stage_records(payload)
    if not stage_records:
        raise RuntimeError(f"No stage records found in {source_npz}")

    decoded: list[dict[str, Any]] = []
    for item in stage_records:
        stage_total = max(1, int(item["stage_total"]))
        progress = float(int(item["local_index"])) / float(max(1, stage_total - 1))
        phase_key, phase_name = _phase_for_progress(int(item["stage_index"]), progress)
        image = _ensure_rgb(item["record"]["image"])
        joint_state = np.asarray(item["record"]["joint_state"], dtype=np.float32).reshape(-1)
        if joint_state.shape[0] != 60:
            raise ValueError(f"Expected 60D joint_state, got {joint_state.shape} in {source_npz}")
        structured_state_summary = ""
        if str(prompt_style) == "phase_structured":
            structured_state_summary = _structured_state_summary(int(item["stage_index"]), progress, phase_name)
        prompt_bimanual = compose_bimanual_low_level_prompt(
            goal=goal,
            subtask=phase_name,
            language_memory="",
            phase_name=phase_name,
            structured_state_summary=structured_state_summary,
            retrieved_semantic_hint="",
            retrieved_visual_hint="",
            prompt_style=str(prompt_style),
        )
        decoded.append(
            {
                "image": image,
                "joint_state": joint_state.astype(np.float32),
                "phase_key": phase_key,
                "phase_name": phase_name,
                "prompt_bimanual": prompt_bimanual,
            }
        )

    per_step: dict[str, list[Any]] = {
        "left_wrist": [],
        "right_wrist": [],
        "left_exterior": [],
        "right_exterior": [],
        "joint_state": [],
        "next_joint_state": [],
        "subtask": [],
        "prompt_bimanual": [],
        "control_phase_key": [],
        "control_phase_name": [],
    }

    for index, item in enumerate(decoded):
        next_item = decoded[min(index + 1, len(decoded) - 1)]
        image = item["image"]
        per_step["left_wrist"].append(_center_crop(_half_view(image, side="left"), frac=0.75))
        per_step["right_wrist"].append(_center_crop(_half_view(image, side="right"), frac=0.75))
        per_step["left_exterior"].append(_center_crop(_half_view(image, side="left"), frac=0.92))
        per_step["right_exterior"].append(_center_crop(_half_view(image, side="right"), frac=0.92))
        per_step["joint_state"].append(item["joint_state"])
        per_step["next_joint_state"].append(next_item["joint_state"])
        per_step["subtask"].append(str(item["phase_name"]))
        per_step["prompt_bimanual"].append(str(item["prompt_bimanual"]))
        per_step["control_phase_key"].append(str(item["phase_key"]))
        per_step["control_phase_name"].append(str(item["phase_name"]))

    return {
        "left_wrist": np.asarray(per_step["left_wrist"], dtype=np.uint8),
        "right_wrist": np.asarray(per_step["right_wrist"], dtype=np.uint8),
        "left_exterior": np.asarray(per_step["left_exterior"], dtype=np.uint8),
        "right_exterior": np.asarray(per_step["right_exterior"], dtype=np.uint8),
        "joint_state": np.asarray(per_step["joint_state"], dtype=np.float32),
        "next_joint_state": np.asarray(per_step["next_joint_state"], dtype=np.float32),
        "subtask": np.asarray(per_step["subtask"], dtype=object),
        "prompt_bimanual": np.asarray(per_step["prompt_bimanual"], dtype=object),
        "control_phase_key": np.asarray(per_step["control_phase_key"], dtype=object),
        "control_phase_name": np.asarray(per_step["control_phase_name"], dtype=object),
    }


def main() -> None:
    args = _parse_args()
    artifact_dirs = [Path(str(item)).expanduser().resolve() for item in args.artifact_dirs]
    output_dir = Path(str(args.output_dir)).expanduser().resolve()
    episodes_dir = output_dir / "episodes"
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir.mkdir(parents=True, exist_ok=True)
    goal_text = str(args.goal).strip() or DEFAULT_GOAL

    rows = _iter_success_records(artifact_dirs, max_episodes=int(args.max_episodes))
    if not rows:
        raise RuntimeError("No successful Dex native episodes were found to convert.")

    manifest_path = output_dir / "manifest.jsonl"
    stats = {
        "ok": True,
        "goal": goal_text,
        "prompt_style": str(args.prompt_style),
        "input_artifact_dirs": [os.fspath(path) for path in artifact_dirs],
        "episodes": 0,
        "steps": 0,
        "source_success_episodes": len(rows),
        "format": "pi05_dex_bimanual_60d",
        "notes": [
            "joint_state and next_joint_state store the full 60D dual-arm UR10e+Shadow-Hand targets",
            "left/right exterior and wrist crops are reconstructed from the native Dex overview image",
        ],
    }

    with manifest_path.open("w", encoding="utf-8") as handle:
        for episode_index, row in enumerate(rows):
            source_npz = Path(str(row["copied_dataset"])).expanduser().resolve()
            converted = _convert_episode(source_npz, goal=goal_text, prompt_style=str(args.prompt_style))
            episode_dir = episodes_dir / f"episode_{episode_index:04d}"
            episode_dir.mkdir(parents=True, exist_ok=True)
            dataset_path = episode_dir / "episode_ll_pi05_bimanual.npz"
            np.savez_compressed(dataset_path, **converted)

            step_count = int(np.asarray(converted["subtask"], dtype=object).shape[0])
            meta = {
                "success": True,
                "episode_index": int(episode_index),
                "dataset_path": os.fspath(dataset_path),
                "source_dataset": os.fspath(source_npz),
                "source_video": str(row["copied_video"]),
                "source_final_state": str(row["copied_final_state"]),
                "source_attempt": int(row["attempt"]),
                "source_gpu": int(row["gpu"]),
                "source_usd_path": str((row.get("source_meta") or {}).get("usd_path", "")),
                "source_meta": row.get("source_meta", {}),
                "steps": int(step_count),
                "goal": goal_text,
                "prompt_style": str(args.prompt_style),
                "observation_source": "reconstructed_from_native_overview",
                "conversion_type": "dex_native_to_pi05_bimanual_ll_sft",
            }
            handle.write(json.dumps(meta, ensure_ascii=False) + "\n")
            stats["episodes"] += 1
            stats["steps"] += int(step_count)

    (output_dir / "summary.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
