from __future__ import annotations

import dataclasses
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from myvla_pi05.image_tools import resize_with_pad_torch
from .fold_tops_prompt import build_fold_tops_hl_prompt
from .pcmb import PerceptualCognitiveMemoryBank, geometry_dict_to_vector, serialize_structured_state


EXTRA_SITE_DIR = os.environ.get("MYVLA_EXTRA_SITE_DIR", "").strip()
if EXTRA_SITE_DIR and EXTRA_SITE_DIR not in sys.path:
    # Keep Isaac/transformers defaults authoritative while making optional
    # inference extras (for example peft) discoverable from a local site dir.
    sys.path.append(EXTRA_SITE_DIR)


@dataclasses.dataclass(frozen=True)
class LongTermMemoryResult:
    """High-level (π_HL) output in MEM: updated language memory + next subtask."""

    memory: str
    subtask: str
    raw_text: str
    structured_state: dict[str, Any] = dataclasses.field(default_factory=dict)
    retrieved_semantic_summary: str = ""
    retrieved_visual_summary: str = ""
    retrieved_semantic_hint: str = ""
    retrieved_visual_hint: str = ""
    pcmb_debug: dict[str, Any] = dataclasses.field(default_factory=dict)
    done: bool = False
    completion_reason: str = ""
    completion_score: float | None = None


_FIELD_NAMES = ("STATE", "MEMORY", "SUBTASK", "SUBTASK_ID", "DONE", "COMPLETION", "REASON")

_SUBTASK_ID_TO_TEXT: dict[str, str] = {
    "APPROACH_LEFT_SLEEVE": "approach the left sleeve cuff from above",
    "GRASP_LEFT_SLEEVE": "grasp the left sleeve cuff",
    "FOLD_LEFT_SLEEVE": "fold the left sleeve inward toward the center",
    "RELEASE_LEFT_SLEEVE": "release the left sleeve and retreat",
    "APPROACH_RIGHT_SLEEVE": "approach the right sleeve cuff from above",
    "GRASP_RIGHT_SLEEVE": "grasp the right sleeve cuff",
    "FOLD_RIGHT_SLEEVE": "fold the right sleeve inward toward the center",
    "RELEASE_RIGHT_SLEEVE": "release the right sleeve and retreat",
    "APPROACH_LOWER_HEM": "approach the lower hem corners",
    "GRASP_LOWER_HEM": "grasp the lower hem corners",
    "LIFT_LOWER_HEM": "lift the lower hem slightly",
    "BRING_LOWER_HEM_TO_CENTER": "bring the lower hem toward the center seam",
    "LAY_LOWER_HEM_FLAT": "lay the lower hem flat near the center seam",
    "RELEASE_LOWER_HEM": "release the lower hem and lift clear",
    "FLATTEN_SQUARE": "flatten and square the folded shirt into a compact block",
    "INSPECT_FINISH": "inspect the square fold and prepare the next adjustment",
    "TASK_COMPLETE": "task complete",
}

_SUBTASK_TEXT_TO_ID: dict[str, str] = {
    "approach the left sleeve cuff from above": "APPROACH_LEFT_SLEEVE",
    "approach left sleeve": "APPROACH_LEFT_SLEEVE",
    "grasp the left sleeve cuff": "GRASP_LEFT_SLEEVE",
    "grasp left sleeve cuff": "GRASP_LEFT_SLEEVE",
    "fold the left sleeve inward toward the center": "FOLD_LEFT_SLEEVE",
    "fold left sleeve inward": "FOLD_LEFT_SLEEVE",
    "release the left sleeve and retreat": "RELEASE_LEFT_SLEEVE",
    "release left sleeve and retreat": "RELEASE_LEFT_SLEEVE",
    "approach the right sleeve cuff from above": "APPROACH_RIGHT_SLEEVE",
    "approach right sleeve": "APPROACH_RIGHT_SLEEVE",
    "grasp the right sleeve cuff": "GRASP_RIGHT_SLEEVE",
    "grasp right sleeve cuff": "GRASP_RIGHT_SLEEVE",
    "fold the right sleeve inward toward the center": "FOLD_RIGHT_SLEEVE",
    "fold right sleeve inward": "FOLD_RIGHT_SLEEVE",
    "release the right sleeve and retreat": "RELEASE_RIGHT_SLEEVE",
    "release right sleeve and retreat": "RELEASE_RIGHT_SLEEVE",
    "approach the lower hem corners": "APPROACH_LOWER_HEM",
    "approach lower hem corners": "APPROACH_LOWER_HEM",
    "grasp the lower hem corners": "GRASP_LOWER_HEM",
    "grasp lower hem corners": "GRASP_LOWER_HEM",
    "lift the lower hem slightly": "LIFT_LOWER_HEM",
    "lift lower hem slightly": "LIFT_LOWER_HEM",
    "bring the lower hem toward the center seam": "BRING_LOWER_HEM_TO_CENTER",
    "bring lower hem toward center": "BRING_LOWER_HEM_TO_CENTER",
    "lay the lower hem flat near the center seam": "LAY_LOWER_HEM_FLAT",
    "lay lower hem flat near center seam": "LAY_LOWER_HEM_FLAT",
    "release the lower hem and lift clear": "RELEASE_LOWER_HEM",
    "release lower hem and lift clear": "RELEASE_LOWER_HEM",
    "flatten and square the folded shirt into a compact block": "FLATTEN_SQUARE",
    "gently sweep side flaps inward and flatten": "FLATTEN_SQUARE",
    "press edges and square the fold": "FLATTEN_SQUARE",
    "inspect the square fold and prepare the next adjustment": "INSPECT_FINISH",
    "inspect square fold and finish": "INSPECT_FINISH",
    "task complete": "TASK_COMPLETE",
}


def _normalize_subtask_id(text: str) -> str:
    raw = str(text or "").strip().upper()
    if not raw:
        return ""
    raw = re.sub(r"[^A-Z0-9]+", "_", raw).strip("_")
    return raw


def _fallback_subtask_id_from_phase(phase_name: str) -> str:
    phase_l = str(phase_name or "").strip().lower()
    if "left sleeve" in phase_l and "grasp" in phase_l:
        return "GRASP_LEFT_SLEEVE"
    if "left sleeve" in phase_l and "fold" in phase_l:
        return "FOLD_LEFT_SLEEVE"
    if "left sleeve" in phase_l and "release" in phase_l:
        return "RELEASE_LEFT_SLEEVE"
    if "left sleeve" in phase_l:
        return "APPROACH_LEFT_SLEEVE"
    if "right sleeve" in phase_l and "grasp" in phase_l:
        return "GRASP_RIGHT_SLEEVE"
    if "right sleeve" in phase_l and "fold" in phase_l:
        return "FOLD_RIGHT_SLEEVE"
    if "right sleeve" in phase_l and "release" in phase_l:
        return "RELEASE_RIGHT_SLEEVE"
    if "right sleeve" in phase_l:
        return "APPROACH_RIGHT_SLEEVE"
    if "lower hem" in phase_l and "grasp" in phase_l:
        return "GRASP_LOWER_HEM"
    if "lower hem" in phase_l and "lift" in phase_l:
        return "LIFT_LOWER_HEM"
    if "lower hem" in phase_l and "bring" in phase_l:
        return "BRING_LOWER_HEM_TO_CENTER"
    if "lower hem" in phase_l and "lay" in phase_l:
        return "LAY_LOWER_HEM_FLAT"
    if "lower hem" in phase_l and "release" in phase_l:
        return "RELEASE_LOWER_HEM"
    if "lower hem" in phase_l:
        return "APPROACH_LOWER_HEM"
    if "inspect" in phase_l:
        return "INSPECT_FINISH"
    if "flatten" in phase_l or "square" in phase_l or "press" in phase_l or "sweep" in phase_l:
        return "FLATTEN_SQUARE"
    return "APPROACH_LEFT_SLEEVE"


def _subtask_text_from_id(subtask_id: str) -> str:
    normalized = _normalize_subtask_id(subtask_id)
    return _SUBTASK_ID_TO_TEXT.get(normalized, "")


def _infer_subtask_id(text: str, *, fallback_phase_name: str = "") -> str:
    normalized = _normalize_subtask_id(text)
    if normalized in _SUBTASK_ID_TO_TEXT:
        return normalized

    compact = " ".join(str(text or "").strip().lower().split())
    if compact in _SUBTASK_TEXT_TO_ID:
        return _SUBTASK_TEXT_TO_ID[compact]

    if any(token in compact for token in ("task complete", "fold complete", "already folded")):
        return "TASK_COMPLETE"
    if "left" in compact and "sleeve" in compact:
        if any(token in compact for token in ("release", "retreat", "lift clear")):
            return "RELEASE_LEFT_SLEEVE"
        if any(token in compact for token in ("grasp", "pinch", "close", "cuff")):
            return "GRASP_LEFT_SLEEVE"
        if any(token in compact for token in ("fold", "bring", "drag", "tuck")):
            return "FOLD_LEFT_SLEEVE"
        return "APPROACH_LEFT_SLEEVE"
    if "right" in compact and "sleeve" in compact:
        if any(token in compact for token in ("release", "retreat", "lift clear")):
            return "RELEASE_RIGHT_SLEEVE"
        if any(token in compact for token in ("grasp", "pinch", "close", "cuff")):
            return "GRASP_RIGHT_SLEEVE"
        if any(token in compact for token in ("fold", "bring", "drag", "tuck")):
            return "FOLD_RIGHT_SLEEVE"
        return "APPROACH_RIGHT_SLEEVE"
    if any(token in compact for token in ("hem", "bottom", "lower edge")):
        if any(token in compact for token in ("release", "retreat", "lift clear")):
            return "RELEASE_LOWER_HEM"
        if any(token in compact for token in ("lay", "place", "put down")):
            return "LAY_LOWER_HEM_FLAT"
        if any(token in compact for token in ("bring", "fold", "tuck", "center")):
            return "BRING_LOWER_HEM_TO_CENTER"
        if "lift" in compact:
            return "LIFT_LOWER_HEM"
        if any(token in compact for token in ("grasp", "pinch", "close")):
            return "GRASP_LOWER_HEM"
        return "APPROACH_LOWER_HEM"
    if any(token in compact for token in ("flatten", "square", "press", "sweep", "align")):
        if any(token in compact for token in ("inspect", "finish")):
            return "INSPECT_FINISH"
        return "FLATTEN_SQUARE"
    return _fallback_subtask_id_from_phase(fallback_phase_name)


def _extract_subtask_id(text: str, *, fallback_phase_name: str = "") -> str | None:
    labeled = _extract_labeled_field(text, "SUBTASK_ID")
    if labeled:
        inferred = _infer_subtask_id(labeled, fallback_phase_name=fallback_phase_name)
        if inferred:
            return inferred

    labeled_subtask = _extract_labeled_field(text, "SUBTASK")
    if labeled_subtask:
        inferred = _infer_subtask_id(labeled_subtask, fallback_phase_name=fallback_phase_name)
        if inferred:
            return inferred

    return None


def _extract_labeled_field(text: str, label: str) -> str | None:
    field_pat = "|".join(_FIELD_NAMES)
    match = re.search(
        rf"(?:^|\n)\s*{re.escape(label)}\s*[:：]\s*(?P<body>.*?)(?=(?:\n\s*(?:{field_pat})\s*[:：])|\Z)",
        str(text or ""),
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None
    value = (match.group("body") or "").strip()
    return value or None


def _as_video_stack_hwc(x: np.ndarray) -> np.ndarray | None:
    if x.ndim != 4:
        return None
    if x.shape[-1] == 3:
        return x
    if x.shape[1] == 3:
        return np.transpose(x, (0, 2, 3, 1))
    return None


def _resize_frame(frame: np.ndarray, *, width: int, height: int) -> np.ndarray:
    frame_np = np.asarray(frame)
    if np.issubdtype(frame_np.dtype, np.floating):
        frame_f = frame_np.astype(np.float32)
        if frame_f.size and float(frame_f.max()) <= 1.5:
            frame_f = frame_f * 255.0
        frame_np = np.clip(frame_f, 0.0, 255.0).astype(np.uint8)
    elif frame_np.dtype != np.uint8:
        frame_np = frame_np.astype(np.uint8)
    image = Image.fromarray(frame_np)
    return np.asarray(image.resize((int(width), int(height)), Image.Resampling.BILINEAR))


def _make_video_montage(video: np.ndarray, *, max_frames: int = 4) -> np.ndarray:
    video = np.asarray(video)
    if video.ndim != 4:
        raise ValueError(f"Expected 4D video stack, got shape: {video.shape}")

    if video.shape[0] <= 1:
        return np.asarray(video[-1])

    max_frames = max(1, int(max_frames))
    if video.shape[0] > max_frames:
        indices = np.linspace(0, video.shape[0] - 1, num=max_frames)
        video = video[np.round(indices).astype(np.int64)]

    frames = [np.asarray(frame) for frame in video]
    target_h = min(frame.shape[0] for frame in frames)
    target_w = min(frame.shape[1] for frame in frames)
    frames = [
        _resize_frame(frame, width=target_w, height=target_h) if frame.shape[:2] != (target_h, target_w) else frame
        for frame in frames
    ]

    if len(frames) == 2:
        return np.concatenate(frames, axis=1)

    cols = 2
    rows = int(math.ceil(len(frames) / cols))
    while len(frames) < rows * cols:
        frames.append(frames[-1].copy())
    row_tiles = [np.concatenate(frames[row * cols : (row + 1) * cols], axis=1) for row in range(rows)]
    return np.concatenate(row_tiles, axis=0)


def _subtask_looks_complete(subtask: str) -> bool:
    subtask_l = str(subtask or "").strip().lower()
    return any(
        token in subtask_l
        for token in ("task complete", "fold complete", "finish", "finished", "done", "final fold complete")
    )


def _parse_done_and_completion(text: str, *, subtask: str) -> tuple[bool, float | None, str]:
    done_text = (_extract_labeled_field(text, "DONE") or "").strip().lower()
    reason = (_extract_labeled_field(text, "REASON") or "").strip()
    completion_score = None
    completion_text = _extract_labeled_field(text, "COMPLETION")
    if completion_text:
        score_match = re.search(r"-?\d+(?:\.\d+)?", completion_text)
        if score_match:
            try:
                completion_score = float(score_match.group(0))
            except ValueError:
                completion_score = None

    done = False
    if done_text:
        done = done_text in ("yes", "true", "1", "done", "complete", "completed")
    elif completion_score is not None:
        done = completion_score >= 95.0
    else:
        done = _subtask_looks_complete(subtask)

    if not reason and done:
        reason = "high_level_policy_marked_task_complete"
    return bool(done), completion_score, reason


def _to_torch_pixel_values(
    image: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
    target_image_size: int | None,
) -> torch.Tensor:
    """Convert a single image (HWC/CHW, uint8/float, optional time stack) to [1, C, H, W] in [-1, 1]."""
    if torch.is_tensor(image):
        x = image.detach().cpu().numpy()
    else:
        x = np.asarray(image)

    if x.ndim == 5:
        # [B, T, H, W, C] / [B, T, C, H, W] -> assume batch size 1 for inference.
        x = x[0]

    video = _as_video_stack_hwc(x)
    if video is not None:
        x = _make_video_montage(video)

    if x.ndim != 3:
        raise ValueError(f"Expected image with 3 dims (HWC/CHW), got shape: {x.shape}")

    if x.shape[0] == 3 and x.shape[-1] != 3:
        # CHW -> HWC
        x = np.transpose(x, (1, 2, 0))

    if x.shape[-1] != 3:
        raise ValueError(f"Expected last dim == 3 (RGB), got shape: {x.shape}")

    if np.issubdtype(x.dtype, np.floating):
        xf = x.astype(np.float32)
        # Heuristic: accept either [0,1] or [-1,1] or [0,255]
        if xf.max() > 2.0:
            xf = xf / 255.0
        if xf.min() < -0.5:
            pixel_values = xf
        else:
            pixel_values = xf * 2.0 - 1.0
    else:
        pixel_values = x.astype(np.float32) / 255.0 * 2.0 - 1.0

    # HWC -> CHW and add batch dim
    pixel_values = torch.from_numpy(np.transpose(pixel_values, (2, 0, 1)))[None, ...].to(device=device)

    if target_image_size is not None:
        _, _, h, w = pixel_values.shape
        if h != target_image_size or w != target_image_size:
            pixel_values = resize_with_pad_torch(pixel_values.to(torch.float32), target_image_size, target_image_size)

    return pixel_values.to(dtype=dtype)


_TAG_RE = re.compile(r"<(?P<tag>MEMORY|SUBTASK)>\\s*(?P<body>.*?)\\s*</(?P=tag)>", flags=re.IGNORECASE | re.DOTALL)


def _parse_memory_and_subtask(text: str) -> tuple[str | None, str | None]:
    text = (text or "").strip()
    memory = None
    subtask = None
    for m in _TAG_RE.finditer(text):
        tag = m.group("tag").upper()
        body = (m.group("body") or "").strip()
        if tag == "MEMORY":
            memory = body
        elif tag == "SUBTASK":
            subtask = body

    if memory is not None or subtask is not None:
        return memory, subtask

    memory = _extract_labeled_field(text, "MEMORY")
    subtask = _extract_labeled_field(text, "SUBTASK")
    if memory is not None or subtask is not None:
        return memory, subtask

    # Fallback: accept "MEMORY: ..." and "SUBTASK: ..." style.
    # Support both ":" and Chinese "：" separators and allow multi-line blocks.
    m2 = re.search(
        r"(?:^|\n)\\s*MEMORY\\s*[:：]\\s*(?P<m>.*?)(?:\\n\\s*SUBTASK\\s*[:：]\\s*(?P<s>.*))?$",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m2:
        memory = (m2.group("m") or "").strip() or None
        subtask = (m2.group("s") or "").strip() or None
        return memory, subtask

    m3 = re.search(r"(?:^|\n)\\s*SUBTASK\\s*[:：]\\s*(?P<s>.*)$", text, flags=re.IGNORECASE | re.DOTALL)
    if m3:
        subtask = (m3.group("s") or "").strip() or None

    return memory, subtask


def _safe_dtype_for_device(dtype: torch.dtype, device: torch.device) -> torch.dtype:
    if device.type == "cpu" and dtype == torch.bfloat16:
        return torch.float32
    return dtype


def _truncate_text(text: str, limit: int = 320) -> str:
    compact = " ".join(str(text or "").split()).strip()
    if len(compact) <= int(limit):
        return compact
    return compact[: max(0, int(limit) - 3)].rstrip() + "..."


def _extract_current_rgb(image: Any) -> np.ndarray:
    x = np.asarray(image)
    if x.ndim == 4:
        x = x[-1]
    if x.ndim == 3 and x.shape[0] == 3 and x.shape[-1] != 3:
        x = np.transpose(x, (1, 2, 0))
    if x.ndim != 3 or x.shape[-1] != 3:
        raise ValueError(f"Expected RGB image frame (HWC), got shape: {x.shape}")
    if np.issubdtype(x.dtype, np.floating):
        xf = x.astype(np.float32)
        if xf.size and float(xf.max()) <= 1.5:
            xf = xf * 255.0
        x = np.clip(xf, 0.0, 255.0).astype(np.uint8)
    elif x.dtype != np.uint8:
        x = x.astype(np.uint8)
    return x


def _extract_cloth_view(image: Any) -> np.ndarray:
    rgb = _extract_current_rgb(image)
    height, width = rgb.shape[:2]
    if width >= int(height * 1.35):
        rgb = rgb[:, width // 2 :, :]
    return rgb


def _estimate_foreground_mask(rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(rgb, dtype=np.float32)
    height, width = image.shape[:2]
    patch = max(4, min(height, width) // 16)
    corners = np.concatenate(
        [
            image[:patch, :patch].reshape(-1, 3),
            image[:patch, -patch:].reshape(-1, 3),
            image[-patch:, :patch].reshape(-1, 3),
            image[-patch:, -patch:].reshape(-1, 3),
        ],
        axis=0,
    )
    background = np.median(corners, axis=0)
    dist = np.linalg.norm(image - background[None, None, :], axis=-1)
    threshold = max(18.0, float(np.percentile(dist, 82.0)) * 0.55)
    mask = dist > threshold
    ratio = float(mask.mean())
    if ratio < 0.01 or ratio > 0.85:
        threshold = max(12.0, float(np.percentile(dist, 90.0)) * 0.45)
        mask = dist > threshold
    return mask


def _extract_cloth_geometry(image: Any, video_metrics: dict[str, Any] | None = None) -> dict[str, float]:
    rgb = _extract_cloth_view(image)
    mask = _estimate_foreground_mask(rgb)
    height, width = mask.shape
    geometry: dict[str, float] = {
        "foreground_ratio": 0.0,
        "bbox_xmin": 0.0,
        "bbox_ymin": 0.0,
        "bbox_xmax": 0.0,
        "bbox_ymax": 0.0,
        "center_x": 0.5,
        "center_y": 0.5,
        "width_ratio": 0.0,
        "height_ratio": 0.0,
        "aspect_ratio": 0.0,
        "left_mass": 0.0,
        "right_mass": 0.0,
        "top_mass": 0.0,
        "bottom_mass": 0.0,
        "edge_density": 0.0,
        "motion_cloth": float((video_metrics or {}).get("cloth_tail_mean_delta", 0.0) or 0.0),
        "motion_overview": float((video_metrics or {}).get("overview_tail_mean_delta", 0.0) or 0.0),
    }
    count = int(mask.sum())
    if count <= 0:
        return geometry

    ys, xs = np.nonzero(mask)
    xmin = int(xs.min())
    xmax = int(xs.max())
    ymin = int(ys.min())
    ymax = int(ys.max())
    width_ratio = float((xmax - xmin + 1) / max(1, width))
    height_ratio = float((ymax - ymin + 1) / max(1, height))
    geometry.update(
        {
            "foreground_ratio": float(count / max(1, mask.size)),
            "bbox_xmin": float(xmin / max(1, width - 1)),
            "bbox_ymin": float(ymin / max(1, height - 1)),
            "bbox_xmax": float(xmax / max(1, width - 1)),
            "bbox_ymax": float(ymax / max(1, height - 1)),
            "center_x": float(xs.mean() / max(1, width - 1)),
            "center_y": float(ys.mean() / max(1, height - 1)),
            "width_ratio": width_ratio,
            "height_ratio": height_ratio,
            "aspect_ratio": float(width_ratio / max(height_ratio, 1e-6)),
            "left_mass": float(mask[:, : width // 2].sum() / max(1, count)),
            "right_mass": float(mask[:, width // 2 :].sum() / max(1, count)),
            "top_mass": float(mask[: height // 2, :].sum() / max(1, count)),
            "bottom_mass": float(mask[height // 2 :, :].sum() / max(1, count)),
        }
    )
    gray = rgb.astype(np.float32).mean(axis=-1)
    grad_x = np.abs(np.diff(gray, axis=1))
    grad_y = np.abs(np.diff(gray, axis=0))
    geometry["edge_density"] = float((grad_x.mean() + grad_y.mean()) * 0.5 / 255.0)
    return geometry


def _geometry_to_text(geometry: dict[str, Any]) -> str:
    return (
        f"area={float(geometry.get('foreground_ratio', 0.0)):.3f}, "
        f"center=({float(geometry.get('center_x', 0.0)):.2f},{float(geometry.get('center_y', 0.0)):.2f}), "
        f"bbox=({float(geometry.get('width_ratio', 0.0)):.2f}w,{float(geometry.get('height_ratio', 0.0)):.2f}h), "
        f"aspect={float(geometry.get('aspect_ratio', 0.0)):.2f}, "
        f"motion={float(geometry.get('motion_cloth', 0.0)):.2f}"
    )


def _normalize_slot(key: str, value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    if key in ("left_sleeve", "right_sleeve"):
        if "fold" in text or "tucked" in text or "inside" in text or "inward" in text:
            return "folded"
        if "grasp" in text or "hold" in text or "pinch" in text:
            return "grasped"
        if "approach" in text or "near" in text or "hover" in text:
            return "approached"
        if "out" in text or "open" in text or "unfold" in text or "outside" in text:
            return "out"
        return text
    if key == "hem":
        if "flat" in text or "flatten" in text or "pressed" in text:
            return "flattened"
        if "release" in text or "clear" in text:
            return "released"
        if "lay" in text or "place" in text:
            return "laid"
        if "center" in text or "bring" in text or "tuck" in text:
            return "centering"
        if "lift" in text:
            return "lifting"
        if "grasp" in text or "hold" in text:
            return "grasped"
        if "down" in text or "lower" in text:
            return "down"
        return text
    if key == "shape":
        if "compact" in text:
            return "compact"
        if "square" in text:
            return "square"
        if "rect" in text or "block" in text:
            return "rectangular"
        if "partial" in text or "folded" in text:
            return "partially_folded"
        if "spread" in text or "open" in text:
            return "spread"
        return text
    if key == "stability":
        if "stable" in text or "still" in text:
            return "stable"
        if "settl" in text:
            return "settling"
        if "move" in text or "unstable" in text:
            return "moving"
        return text
    return str(value or "").strip()


def _state_from_phase_and_metrics(
    *,
    phase_name: str,
    prev_state: dict[str, Any] | None,
    subtask: str,
    video_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    phase_l = str(phase_name or "").lower()
    state = dict(prev_state or {})
    state.setdefault("fold_stage", str(phase_name or "unknown").strip() or "unknown")
    state.setdefault("left_sleeve", "out")
    state.setdefault("right_sleeve", "out")
    state.setdefault("hem", "down")
    state.setdefault("shape", "spread")
    state.setdefault("stability", "uncertain")
    state.setdefault("last_effect", "")
    state.setdefault("next_focus", str(subtask or phase_name or "").strip())

    if "left sleeve" in phase_l:
        state["left_sleeve"] = "grasped" if "grasp" in phase_l else ("folded" if "fold" in phase_l or "release" in phase_l else "out")
        state["right_sleeve"] = "out"
        state["hem"] = "down"
        state["shape"] = "spread"
    elif "right sleeve" in phase_l:
        state["left_sleeve"] = "folded"
        state["right_sleeve"] = "grasped" if "grasp" in phase_l else ("folded" if "fold" in phase_l or "release" in phase_l else "out")
        state["hem"] = "down"
        state["shape"] = "partially_folded"
    elif "lower hem" in phase_l or "sweep" in phase_l or "press" in phase_l or "inspect" in phase_l:
        state["left_sleeve"] = "folded"
        state["right_sleeve"] = "folded"
        if "grasp" in phase_l:
            state["hem"] = "grasped"
        elif "lift" in phase_l:
            state["hem"] = "lifting"
        elif "bring" in phase_l:
            state["hem"] = "centering"
        elif "lay" in phase_l:
            state["hem"] = "laid"
        elif "release" in phase_l:
            state["hem"] = "released"
        elif "sweep" in phase_l or "press" in phase_l:
            state["hem"] = "flattened"
        if "inspect" in phase_l:
            state["shape"] = "square"
        elif "press" in phase_l or "sweep" in phase_l:
            state["shape"] = "compact"
        else:
            state["shape"] = "rectangular"

    cloth_motion = float((video_metrics or {}).get("cloth_tail_mean_delta", 0.0) or 0.0)
    if cloth_motion <= 2.0:
        state["stability"] = "stable"
    elif cloth_motion <= 5.0:
        state["stability"] = "settling"
    else:
        state["stability"] = "moving"

    state["fold_stage"] = str(phase_name or state.get("fold_stage", "unknown")).strip() or "unknown"
    state["next_focus"] = str(subtask or state.get("next_focus", "")).strip()
    return state


def _parse_state_field(text: str) -> dict[str, Any]:
    raw = (_extract_labeled_field(text, "STATE") or "").strip()
    if not raw:
        return {}
    candidate = raw
    if "{" in candidate and "}" in candidate:
        candidate = candidate[candidate.find("{") : candidate.rfind("}") + 1]
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return {str(k): v for k, v in parsed.items()}
    except json.JSONDecodeError:
        pass

    out: dict[str, Any] = {}
    for part in re.split(r"[;,]\s*", raw):
        if ":" not in part and "=" not in part:
            continue
        sep = ":" if ":" in part else "="
        key, value = part.split(sep, 1)
        out[str(key).strip()] = str(value).strip()
    return out


def _normalize_structured_state(
    *,
    parsed_state: dict[str, Any],
    phase_name: str,
    prev_state: dict[str, Any] | None,
    subtask: str,
    video_metrics: dict[str, Any] | None,
    completion_score: float | None,
) -> dict[str, Any]:
    state = _state_from_phase_and_metrics(
        phase_name=phase_name,
        prev_state=prev_state,
        subtask=subtask,
        video_metrics=video_metrics,
    )
    for key, value in dict(parsed_state or {}).items():
        norm_key = str(key).strip().lower()
        if norm_key not in state:
            continue
        normalized = _normalize_slot(norm_key, value)
        if normalized:
            state[norm_key] = normalized
    if completion_score is not None:
        if float(completion_score) >= 97.0:
            state["shape"] = "compact"
            state["stability"] = "stable"
        elif float(completion_score) >= 90.0 and state.get("shape") == "spread":
            state["shape"] = "rectangular"
    return state


def _state_to_memory_text(state: dict[str, Any], goal: str) -> str:
    summary = serialize_structured_state(state)
    if not summary:
        return str(goal).strip()
    return _truncate_text(summary, limit=420)


class PretrainedVlmLongTermMemoryProcessor:
    """MEM long-term memory processor implemented as a pretrained SigLIP+Gemma VLM (e.g. PaliGemma).

    Notes:
    - This module must use *pretrained* VLM weights (separate from the fine-tuned pi0.5 checkpoint).
    - It maintains language memory `m_t` and emits a language subtask `l_{t+1}`.
    """

    def __init__(
        self,
        model_dir: str,
        *,
        device: str | None = None,
        dtype: str = "bfloat16",
        revision: str | None = None,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        cognitive_mem_length: int = 24,
        perceptual_mem_length: int = 24,
        retrieval_top_k: int = 3,
    ) -> None:
        if not str(model_dir).strip():
            raise ValueError("model_dir must be a non-empty path or HF model id.")

        self._device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype_t = torch.bfloat16 if dtype == "bfloat16" else torch.float32
        self._dtype = _safe_dtype_for_device(dtype_t, self._device)
        self._max_new_tokens = int(max_new_tokens)
        self._temperature = float(temperature)
        self._revision = str(revision) if revision is not None and str(revision).strip() else None

        from transformers import AutoTokenizer, PaliGemmaForConditionalGeneration

        model_path = Path(str(model_dir)).expanduser()
        adapter_config_path = model_path / "adapter_config.json"
        tokenizer_source = str(model_dir)

        if adapter_config_path.is_file():
            try:
                from peft import PeftConfig, PeftModel
            except Exception as exc:
                raise RuntimeError(
                    "LoRA adapter detected for high-level VLM, but `peft` is not importable. "
                    "Set MYVLA_EXTRA_SITE_DIR to a site-packages directory containing peft."
                ) from exc

            peft_config = PeftConfig.from_pretrained(str(model_path))
            base_model_dir = str(peft_config.base_model_name_or_path)
            if not base_model_dir.strip():
                raise RuntimeError(f"Adapter at {model_path} does not define base_model_name_or_path.")

            if not any((model_path / name).exists() for name in ("tokenizer.json", "tokenizer_config.json", "tokenizer.model")):
                tokenizer_source = base_model_dir

            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, revision=self._revision)
            base_model = PaliGemmaForConditionalGeneration.from_pretrained(
                base_model_dir,
                revision=self._revision,
                torch_dtype=self._dtype,
            )
            self._model = PeftModel.from_pretrained(base_model, str(model_path))
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(model_dir, revision=self._revision)
            self._model = PaliGemmaForConditionalGeneration.from_pretrained(
                model_dir,
                revision=self._revision,
                torch_dtype=self._dtype,
            )
        self._model.to(self._device)
        self._model.eval()

        self._image_token_id = int(self._model.config.image_token_id)
        image_size = int(self._model.config.vision_config.image_size)
        patch_size = int(self._model.config.vision_config.patch_size)
        self._target_image_size = image_size
        self._num_image_tokens = (image_size // patch_size) ** 2
        self._pcmb = PerceptualCognitiveMemoryBank(
            cognitive_capacity=int(cognitive_mem_length),
            perceptual_capacity=int(perceptual_mem_length),
            top_k=int(retrieval_top_k),
        )
        self._step_counter = 0
        self._last_structured_state: dict[str, Any] = {}

    @property
    def device(self) -> torch.device:
        return self._device

    def reset(self) -> None:
        self._pcmb.reset()
        self._step_counter = 0
        self._last_structured_state = {}

    def _build_inputs(self, prompt: str) -> dict[str, torch.Tensor]:
        text = self._tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = text["input_ids"]
        attention_mask = text.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # Insert image tokens right after BOS (processor-style).
        if input_ids.shape[1] == 0:
            raise ValueError("Tokenizer produced empty input_ids.")

        bsz = input_ids.shape[0]
        image_tokens = torch.full(
            (bsz, self._num_image_tokens),
            self._image_token_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_mask = torch.ones((bsz, self._num_image_tokens), dtype=attention_mask.dtype, device=attention_mask.device)

        bos = input_ids[:, :1]
        rest = input_ids[:, 1:]
        bos_mask = attention_mask[:, :1]
        rest_mask = attention_mask[:, 1:]

        input_ids = torch.cat([bos, image_tokens, rest], dim=1)
        attention_mask = torch.cat([bos_mask, image_mask, rest_mask], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def _encode_text_embedding(self, text: str) -> torch.Tensor:
        encoded = self._tokenizer(
            str(text or ""),
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=256,
        )
        input_ids = encoded["input_ids"].to(self._device)
        attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(self._device)
        embed_module = self._model.get_input_embeddings()
        with torch.no_grad():
            token_embs = embed_module(input_ids).to(torch.float32)
        weights = attention_mask.to(torch.float32).unsqueeze(-1)
        pooled = (token_embs * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return pooled[0].detach().cpu()

    def _extract_visual_embedding(self, pixel_values: torch.Tensor) -> torch.Tensor:
        core_model = getattr(self._model, "model", None)
        if core_model is None and hasattr(self._model, "base_model"):
            base_model = getattr(self._model, "base_model")
            core_model = getattr(base_model, "model", base_model)
        if core_model is None:
            core_model = self._model
        vision_tower = getattr(core_model, "vision_tower", None)
        projector = getattr(core_model, "multi_modal_projector", None)
        if vision_tower is None:
            flat = pixel_values.to(torch.float32).flatten(start_dim=1)
            return flat.mean(dim=1)[0].detach().cpu()

        with torch.no_grad():
            vision_outputs = vision_tower(pixel_values)
            last_hidden_state = (
                vision_outputs[0] if isinstance(vision_outputs, (tuple, list)) else vision_outputs.last_hidden_state
            )
            if projector is not None:
                last_hidden_state = projector(last_hidden_state)
            pooled = last_hidden_state.to(torch.float32).mean(dim=1)
        return pooled[0].detach().cpu()

    def _generate_text(self, *, prompt: str, image: Any, max_new_tokens: int) -> str:
        inputs = self._build_inputs(prompt)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        pixel_values = _to_torch_pixel_values(
            image,
            device=self._device,
            dtype=self._dtype,
            target_image_size=self._target_image_size,
        )

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens),
        }
        if self._temperature > 0:
            gen_kwargs.update({"do_sample": True, "temperature": self._temperature})
        else:
            gen_kwargs.update({"do_sample": False})

        with torch.no_grad():
            out_ids = self._model.generate(**inputs, pixel_values=pixel_values, **gen_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        new_ids = out_ids[0, prompt_len:]
        return self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    def _build_retrieval_block(
        self,
        *,
        retrieved_semantic: str,
        retrieved_visual: str,
    ) -> str:
        blocks: list[str] = []
        if str(retrieved_semantic).strip():
            blocks.append("Retrieved semantic memory:\n" + str(retrieved_semantic).strip())
        if str(retrieved_visual).strip():
            blocks.append("Retrieved visual evidence:\n" + str(retrieved_visual).strip())
        if not blocks:
            return ""
        return "\n".join(blocks) + "\n\n"

    def update(
        self,
        *,
        goal: str,
        prev_memory: str,
        image: Any,
        phase_name: str = "",
        video_metrics: dict[str, Any] | None = None,
        step: int | None = None,
        **_ignored: Any,
    ) -> LongTermMemoryResult:
        goal = str(goal)
        prev_memory = str(prev_memory or "").strip()
        phase_name = str(phase_name or "").strip()
        video_metrics = dict(video_metrics or {})
        step_index = int(self._step_counter if step is None else step)
        pixel_values = _to_torch_pixel_values(
            image,
            device=self._device,
            dtype=self._dtype,
            target_image_size=self._target_image_size,
        )
        current_geometry = _extract_cloth_geometry(image, video_metrics=video_metrics)
        current_visual_embedding = self._extract_visual_embedding(pixel_values)
        current_geometry_vector = geometry_dict_to_vector(current_geometry)
        current_state_guess = _state_from_phase_and_metrics(
            phase_name=phase_name,
            prev_state=self._last_structured_state,
            subtask=phase_name or goal,
            video_metrics=video_metrics,
        )
        query_text = "\n".join(
            [
                f"Goal: {goal}",
                f"Controller context: {phase_name if phase_name else '<none>'}",
                f"Previous memory: {prev_memory if prev_memory else '<empty>'}",
                f"Current state guess: {serialize_structured_state(current_state_guess)}",
                f"Current geometry: {_geometry_to_text(current_geometry)}",
            ]
        ).strip()
        current_cognitive_embedding = self._encode_text_embedding(query_text)
        retrieved = self._pcmb.retrieve(
            cognitive_query=current_cognitive_embedding,
            perceptual_query=current_visual_embedding,
            geometry_query=current_geometry_vector,
        )
        retrieval_block = self._build_retrieval_block(
            retrieved_semantic=retrieved.semantic_prompt,
            retrieved_visual=retrieved.visual_prompt,
        )

        prompt = build_fold_tops_hl_prompt(
            goal=goal,
            prev_memory=prev_memory,
            phase_name=phase_name,
            geometry_text=_geometry_to_text(current_geometry),
            video_metrics=video_metrics,
            retrieved_semantic=retrieved.semantic_prompt,
            retrieved_visual=retrieved.visual_prompt,
        )

        inputs = self._build_inputs(prompt)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self._max_new_tokens,
        }
        if self._temperature > 0:
            gen_kwargs.update({"do_sample": True, "temperature": self._temperature})
        else:
            gen_kwargs.update({"do_sample": False})

        with torch.no_grad():
            out_ids = self._model.generate(**inputs, pixel_values=pixel_values, **gen_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        new_ids = out_ids[0, prompt_len:]
        raw_text = self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        memory, subtask = _parse_memory_and_subtask(raw_text)
        memory = (memory if memory is not None else prev_memory).strip()
        subtask = (subtask if subtask is not None else goal).strip()
        done, completion_score, completion_reason = _parse_done_and_completion(raw_text, subtask=subtask)
        parsed_state = _parse_state_field(raw_text)
        structured_state = _normalize_structured_state(
            parsed_state=parsed_state,
            phase_name=phase_name,
            prev_state=self._last_structured_state,
            subtask=subtask,
            video_metrics=video_metrics,
            completion_score=completion_score,
        )

        if not memory:
            memory = _state_to_memory_text(structured_state, goal)

        # Keep memory succinct by default (rough cap in characters; can be tuned).
        if len(memory) > 4000:
            memory = memory[-4000:]

        cognitive_summary = _truncate_text(
            f"{serialize_structured_state(structured_state)}; memory={memory}; "
            f"completion={completion_score if completion_score is not None else 'unknown'}",
            limit=240,
        )
        perceptual_evidence = _truncate_text(_geometry_to_text(current_geometry), limit=220)
        self._pcmb.add_cognitive(
            step=step_index,
            embedding=self._encode_text_embedding(cognitive_summary),
            state=structured_state,
            summary=cognitive_summary,
            completion_score=completion_score,
        )
        self._pcmb.add_perceptual(
            step=step_index,
            embedding=current_visual_embedding,
            geometry=current_geometry,
            evidence=perceptual_evidence,
        )
        self._last_structured_state = dict(structured_state)
        self._step_counter = int(step_index) + 1

        return LongTermMemoryResult(
            memory=memory,
            subtask=subtask,
            raw_text=raw_text,
            structured_state=structured_state,
            retrieved_semantic_summary=retrieved.semantic_prompt,
            retrieved_visual_summary=retrieved.visual_prompt,
            retrieved_semantic_hint=retrieved.semantic_hint,
            retrieved_visual_hint=retrieved.visual_hint,
            pcmb_debug={
                **self._pcmb.snapshot(),
                "current_step": int(step_index),
                "current_geometry": dict(current_geometry),
                "retrieval": retrieved.as_debug_dict(),
            },
            done=done,
            completion_reason=completion_reason,
            completion_score=completion_score,
        )
