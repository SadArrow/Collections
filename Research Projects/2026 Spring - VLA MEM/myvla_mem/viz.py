from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import os
import pathlib
from typing import Any

import numpy as np


def _now_stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_name(name: str) -> str:
    out = []
    for ch in str(name):
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out).strip("._")
    return s or "item"


def _ensure_uint8_hwc_rgb(image: Any) -> np.ndarray:
    x = np.asarray(image)
    if x.ndim == 3 and x.shape[0] == 3 and x.shape[-1] != 3:
        x = np.transpose(x, (1, 2, 0))  # CHW -> HWC
    if x.ndim != 3 or x.shape[-1] != 3:
        raise ValueError(f"Expected RGB image (HWC), got shape: {x.shape}")
    if np.issubdtype(x.dtype, np.floating):
        x = np.clip(x, 0.0, 1.0)
        x = (x * 255.0).astype(np.uint8)
    elif x.dtype != np.uint8:
        x = x.astype(np.uint8)
    return x


def _save_image_png(image_hwc_uint8: np.ndarray, path: pathlib.Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_hwc_uint8).save(os.fspath(path))


def _save_stack(
    stack: Any,
    *,
    out_dir: pathlib.Path,
    name: str,
    make_gif: bool = True,
    gif_duration_ms: int = 150,
) -> dict[str, Any]:
    x = np.asarray(stack)
    out: dict[str, Any] = {"shape": list(x.shape), "dtype": str(x.dtype)}

    cam_dir = out_dir / _safe_name(name)
    cam_dir.mkdir(parents=True, exist_ok=True)

    if x.ndim == 3:
        frame = _ensure_uint8_hwc_rgb(x)
        _save_image_png(frame, cam_dir / "frame_000.png")
        out["num_frames"] = 1
        out["current_png"] = str((pathlib.Path(_safe_name(name)) / "frame_000.png").as_posix())
        return out

    if x.ndim != 4:
        raise ValueError(f"Expected image stack with 3D or 4D, got shape: {x.shape}")

    frames = []
    for i, f in enumerate(x):
        frame = _ensure_uint8_hwc_rgb(f)
        rel = cam_dir / f"frame_{i:03d}.png"
        _save_image_png(frame, rel)
        frames.append(frame)

    out["num_frames"] = int(x.shape[0])
    out["current_png"] = str((pathlib.Path(_safe_name(name)) / f"frame_{x.shape[0]-1:03d}.png").as_posix())

    if make_gif and len(frames) > 1:
        from PIL import Image

        gif_path = cam_dir / "stack.gif"
        pil_frames = [Image.fromarray(f) for f in frames]
        pil_frames[0].save(
            os.fspath(gif_path),
            save_all=True,
            append_images=pil_frames[1:],
            duration=int(gif_duration_ms),
            loop=0,
        )
        out["gif"] = str((pathlib.Path(_safe_name(name)) / "stack.gif").as_posix())

    return out


def _write_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(text), encoding="utf-8")


def _write_json(path: pathlib.Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _save_actions(actions: Any, path_prefix: pathlib.Path) -> dict[str, Any]:
    x = np.asarray(actions, dtype=np.float32)
    np.save(os.fspath(path_prefix.with_suffix(".npy")), x)
    np.savetxt(os.fspath(path_prefix.with_suffix(".csv")), x, delimiter=",", fmt="%.6f")
    return {"shape": list(x.shape), "dtype": str(x.dtype), "npy": path_prefix.with_suffix(".npy").name, "csv": path_prefix.with_suffix(".csv").name}


@dataclasses.dataclass
class InferenceVizWriter:
    run_dir: pathlib.Path
    _report_lines: list[str] = dataclasses.field(default_factory=list)

    @classmethod
    def create(cls, *, base_dir: pathlib.Path, name: str | None, meta: dict[str, Any]) -> "InferenceVizWriter":
        base_dir.mkdir(parents=True, exist_ok=True)
        run_name = _safe_name(name) if name and str(name).strip() else _now_stamp()
        run_dir = base_dir / run_name
        suffix = 1
        while run_dir.exists():
            run_dir = base_dir / f"{run_name}_{suffix}"
            suffix += 1
        run_dir.mkdir(parents=True, exist_ok=False)

        _write_json(run_dir / "meta.json", meta)
        writer = cls(run_dir=run_dir)
        writer._report_lines.append(f"# myVLA inference viz: `{run_dir.name}`\n")
        return writer

    def add_step(
        self,
        step: int,
        *,
        goal: str,
        low_level_prompt: str,
        prev_memory: str,
        language_memory: str,
        subtask: str,
        hl_raw_text: str | None,
        structured_state: dict[str, Any] | None = None,
        retrieved_semantic_summary: str = "",
        retrieved_visual_summary: str = "",
        pcmb_debug: dict[str, Any] | None = None,
        images: dict[str, Any],
        actions: Any,
    ) -> None:
        step_dir = self.run_dir / f"step_{int(step):03d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        _write_text(step_dir / "goal.txt", goal)
        _write_text(step_dir / "low_level_prompt.txt", low_level_prompt)
        _write_text(step_dir / "prev_memory.txt", prev_memory)
        _write_text(step_dir / "language_memory.txt", language_memory)
        _write_text(step_dir / "subtask.txt", subtask)
        if hl_raw_text is not None:
            _write_text(step_dir / "hl_raw_text.txt", hl_raw_text)
        if structured_state:
            _write_json(step_dir / "structured_state.json", structured_state)
        if str(retrieved_semantic_summary).strip():
            _write_text(step_dir / "retrieved_semantic_summary.txt", retrieved_semantic_summary)
        if str(retrieved_visual_summary).strip():
            _write_text(step_dir / "retrieved_visual_summary.txt", retrieved_visual_summary)
        if pcmb_debug:
            _write_json(step_dir / "pcmb_debug.json", pcmb_debug)

        vision_dir = step_dir / "vision"
        vision_meta: dict[str, Any] = {}
        for k, v in (images or {}).items():
            if v is None:
                continue
            vision_meta[str(k)] = _save_stack(v, out_dir=vision_dir, name=str(k))
        _write_json(step_dir / "vision.json", vision_meta)

        actions_meta = _save_actions(actions, step_dir / "actions")
        _write_json(step_dir / "actions.json", actions_meta)

        # Markdown report entry
        self._report_lines.append(f"## Step {int(step)}\n")
        self._report_lines.append(f"- Goal: `{goal}`\n")
        self._report_lines.append(f"- Low-level prompt: `{low_level_prompt}`\n")
        self._report_lines.append(f"- Subtask: `{subtask}`\n")
        mem_preview = (language_memory or "").replace("\n", " ").strip()
        if len(mem_preview) > 300:
            mem_preview = mem_preview[:300] + "..."
        self._report_lines.append(f"- Language memory: `{mem_preview}`\n")
        if structured_state:
            self._report_lines.append(f"- Structured state: `{json.dumps(structured_state, ensure_ascii=False)}`\n")
        if str(retrieved_semantic_summary).strip():
            sem_preview = str(retrieved_semantic_summary).replace("\n", " ").strip()
            if len(sem_preview) > 240:
                sem_preview = sem_preview[:240] + "..."
            self._report_lines.append(f"- Retrieved semantic summary: `{sem_preview}`\n")
        if str(retrieved_visual_summary).strip():
            vis_preview = str(retrieved_visual_summary).replace("\n", " ").strip()
            if len(vis_preview) > 240:
                vis_preview = vis_preview[:240] + "..."
            self._report_lines.append(f"- Retrieved visual summary: `{vis_preview}`\n")

        if hl_raw_text is not None:
            raw_preview = (hl_raw_text or "").replace("\n", " ").strip()
            if len(raw_preview) > 300:
                raw_preview = raw_preview[:300] + "..."
            self._report_lines.append(f"- HL raw text: `{raw_preview}`\n")

        for cam_key, info in vision_meta.items():
            rel_dir = _safe_name(cam_key)
            if "gif" in info:
                rel = pathlib.Path(f"step_{int(step):03d}") / "vision" / rel_dir / "stack.gif"
                self._report_lines.append(f"\n![{cam_key}]({rel.as_posix()})\n")
            else:
                rel = pathlib.Path(f"step_{int(step):03d}") / "vision" / rel_dir / "frame_000.png"
                self._report_lines.append(f"\n![{cam_key}]({rel.as_posix()})\n")

        self._report_lines.append("\n")

    def finalize(self, *, final_actions: Any | None = None) -> None:
        if final_actions is not None:
            _save_actions(final_actions, self.run_dir / "final_actions")
        _write_text(self.run_dir / "report.md", "".join(self._report_lines))
