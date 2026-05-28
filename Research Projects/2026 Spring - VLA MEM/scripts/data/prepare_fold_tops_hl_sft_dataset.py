from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
if os.fspath(REPO_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPO_ROOT))

from myvla_mem.fold_tops_prompt import build_fold_tops_hl_prompt, default_fold_tops_goal, render_fold_tops_hl_target


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert DexGarmentLab Fold_Tops native train_data/*.npz into a high-level VLM SFT dataset."
    )
    parser.add_argument("--dex_root", required=True, help="Path to DexGarmentLab-main/DexGarmentLab-main")
    parser.add_argument("--task_name", default="Fold_Tops")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sequence_len", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--limit", type=int, default=0, help="0 = use all episodes")
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


def _image_grid(frames: list[np.ndarray], *, image_size: int) -> np.ndarray:
    count = len(frames)
    if count <= 0:
        raise ValueError("frames must not be empty")
    side = int(math.ceil(math.sqrt(count)))
    tile = max(8, int(image_size // side))
    canvas = np.zeros((side * tile, side * tile, 3), dtype=np.uint8)
    for idx, frame in enumerate(frames):
        row = idx // side
        col = idx % side
        tile_img = np.asarray(Image.fromarray(_ensure_rgb(frame)).resize((tile, tile), Image.Resampling.BILINEAR))
        canvas[row * tile : (row + 1) * tile, col * tile : (col + 1) * tile] = tile_img
    return np.asarray(Image.fromarray(canvas).resize((image_size, image_size), Image.Resampling.BILINEAR))


def _canonical_subtask(stage_index: int, progress: float) -> str:
    p = float(np.clip(progress, 0.0, 1.0))
    if stage_index == 1:
        if p < 0.22:
            return "approach the left sleeve cuff from above"
        if p < 0.38:
            return "grasp the left sleeve cuff"
        if p < 0.82:
            return "fold the left sleeve inward toward the center"
        return "release the left sleeve and retreat"
    if stage_index == 2:
        if p < 0.22:
            return "approach the right sleeve cuff from above"
        if p < 0.38:
            return "grasp the right sleeve cuff"
        if p < 0.82:
            return "fold the right sleeve inward toward the center"
        return "release the right sleeve and retreat"
    if p < 0.15:
        return "approach the lower hem corners"
    if p < 0.28:
        return "grasp the lower hem corners"
    if p < 0.42:
        return "lift the lower hem slightly"
    if p < 0.64:
        return "bring the lower hem toward the center seam"
    if p < 0.80:
        return "lay the lower hem flat near the center seam"
    if p < 0.90:
        return "release the lower hem and lift clear"
    if p < 0.96:
        return "flatten and square the folded shirt into a compact block"
    return "task complete"


def _structured_state(stage_index: int, progress: float) -> dict[str, str]:
    p = float(np.clip(progress, 0.0, 1.0))
    state = {
        "fold_stage": "left_sleeve" if stage_index == 1 else ("right_sleeve" if stage_index == 2 else "lower_hem"),
        "left_sleeve": "out",
        "right_sleeve": "out",
        "hem": "down",
        "shape": "spread",
        "stability": "moving",
        "last_effect": "",
        "next_focus": _canonical_subtask(stage_index, p),
    }
    if stage_index == 1:
        state["left_sleeve"] = "approached" if p < 0.38 else ("grasped" if p < 0.50 else "folded")
        state["last_effect"] = "left sleeve moved inward"
    elif stage_index == 2:
        state["left_sleeve"] = "folded"
        state["right_sleeve"] = "approached" if p < 0.38 else ("grasped" if p < 0.50 else "folded")
        state["shape"] = "partially_folded"
        state["last_effect"] = "right sleeve moved inward"
    else:
        state["left_sleeve"] = "folded"
        state["right_sleeve"] = "folded"
        state["shape"] = "rectangular" if p < 0.85 else "compact"
        state["last_effect"] = "lower hem moved toward center"
        if p < 0.28:
            state["hem"] = "down"
        elif p < 0.42:
            state["hem"] = "grasped"
        elif p < 0.64:
            state["hem"] = "lifting"
        elif p < 0.80:
            state["hem"] = "centering"
        elif p < 0.90:
            state["hem"] = "laid"
        else:
            state["hem"] = "flattened"
    if stage_index == 3 and p >= 0.94:
        state["shape"] = "square"
        state["stability"] = "stable"
    elif p >= 0.88:
        state["stability"] = "settling"
    return state


def _completion(stage_index: int, progress: float) -> int:
    p = float(np.clip(progress, 0.0, 1.0))
    if stage_index == 1:
        return int(round(12 + 28 * p))
    if stage_index == 2:
        return int(round(40 + 25 * p))
    return int(round(68 + 32 * p))


def _memory_text(state: dict[str, str]) -> str:
    ordered = [
        "fold_stage",
        "left_sleeve",
        "right_sleeve",
        "hem",
        "shape",
        "stability",
        "last_effect",
        "next_focus",
    ]
    return "; ".join(f"{key}={state[key]}" for key in ordered if str(state.get(key, "")).strip())


def _phase_name_for_prompt(stage_index: int, step_index: int, stage_length: int) -> str:
    if int(step_index) <= 0:
        return ""
    prev_progress = float(step_index - 1) / float(max(1, stage_length - 1))
    return _canonical_subtask(stage_index, prev_progress)


def _iter_npz_files(train_data_dir: Path, limit: int) -> list[Path]:
    files = sorted(train_data_dir.glob("data_*.npz"))
    if int(limit) > 0:
        files = files[: int(limit)]
    return files


def main() -> None:
    args = _parse_args()
    dex_root = Path(str(args.dex_root)).expanduser().resolve()
    train_data_dir = dex_root / "Data" / str(args.task_name) / "train_data"
    if not train_data_dir.is_dir():
        raise FileNotFoundError(f"train_data dir not found: {train_data_dir}")

    output_dir = Path(str(args.output_dir)).expanduser().resolve()
    images_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.jsonl"
    stats = {
        "episodes": 0,
        "samples": 0,
        "stage_counts": {"1": 0, "2": 0, "3": 0},
    }

    goal = default_fold_tops_goal(
        "Fold the shirt into a compact square block by folding both sleeves inward and then folding the lower hem upward using two robot arms."
    )
    npz_files = _iter_npz_files(train_data_dir, int(args.limit))

    with manifest_path.open("w", encoding="utf-8") as handle:
        for episode_index, npz_path in enumerate(npz_files):
            payload = np.load(npz_path, allow_pickle=True)
            usd_path = str(payload.get("usd_path", ""))
            pos = np.asarray(payload.get("pos", []), dtype=np.float32).tolist()
            ori = np.asarray(payload.get("ori", []), dtype=np.float32).tolist()
            episode_written = False

            for stage_index in (1, 2, 3):
                key = f"stage_{stage_index}"
                if key not in payload:
                    continue
                stage_data = payload[key]
                if len(stage_data) <= 0:
                    continue
                frames = [_ensure_rgb(item["image"]) for item in stage_data]
                total_steps = max(1, len(stage_data))
                for step_index in range(total_steps):
                    progress = float(step_index) / float(max(1, total_steps - 1))
                    subtask = _canonical_subtask(stage_index, progress)
                    state = _structured_state(stage_index, progress)
                    done = bool(stage_index == 3 and progress >= 0.96)
                    completion = _completion(stage_index, progress)
                    reason = (
                        "shirt already looks compact, square, and stable"
                        if done
                        else "folding still in progress"
                    )
                    frame_window = []
                    for back in range(int(args.sequence_len)):
                        src_idx = max(0, step_index - (int(args.sequence_len) - 1 - back))
                        frame_window.append(frames[src_idx])
                    grid = _image_grid(frame_window, image_size=int(args.image_size))
                    image_name = f"ep{episode_index:04d}_s{stage_index}_t{step_index:04d}.png"
                    image_path = images_dir / image_name
                    Image.fromarray(grid).save(image_path)
                    prev_phase_name = _phase_name_for_prompt(stage_index, step_index, total_steps)
                    prev_memory = ""
                    if step_index > 0:
                        prev_progress = float(step_index - 1) / float(max(1, total_steps - 1))
                        prev_state = _structured_state(stage_index, prev_progress)
                        prev_memory = _memory_text(prev_state)
                    prompt_text = build_fold_tops_hl_prompt(
                        goal=goal,
                        prev_memory=prev_memory,
                        phase_name=prev_phase_name,
                        geometry_text="",
                        video_metrics={"frame_count": min(int(args.sequence_len), step_index + 1)},
                        retrieved_semantic="",
                        retrieved_visual="",
                    )
                    target_memory = _memory_text(state)

                    record = {
                        "episode_id": int(episode_index),
                        "source_npz": os.fspath(npz_path),
                        "task_name": str(args.task_name),
                        "goal": goal,
                        "stage_index": int(stage_index),
                        "step_index": int(step_index),
                        "stage_length": int(total_steps),
                        "progress": progress,
                        "image_path": os.fspath(image_path),
                        "prompt_text": prompt_text,
                        "phase_name": prev_phase_name,
                        "prev_memory": prev_memory,
                        "target_state": state,
                        "target_memory": target_memory,
                        "target_subtask": subtask,
                        "done": done,
                        "completion": int(completion),
                        "reason": reason,
                        "target_text": render_fold_tops_hl_target(
                            state=state,
                            subtask=subtask,
                            memory=target_memory,
                            done=done,
                            completion=completion,
                            reason=reason,
                        ),
                        "metadata": {
                            "usd_path": usd_path,
                            "pos": pos,
                            "ori": ori,
                        },
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stats["samples"] += 1
                    stats["stage_counts"][str(stage_index)] += 1
                    episode_written = True

            if episode_written:
                stats["episodes"] += 1

    (output_dir / "summary.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "output_dir": os.fspath(output_dir), **stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
