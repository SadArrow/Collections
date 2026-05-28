from __future__ import annotations

import collections
from typing import Any

import numpy as np

from myvla_pi05.policy import Pi05DroidPolicy

from .long_term import LongTermMemoryResult, PretrainedVlmLongTermMemoryProcessor


_DROID_IMAGE_KEYS = ("observation/exterior_image_1_left", "observation/wrist_image_left")


def _compact_line(text: str, *, limit: int = 180) -> str:
    compact = " ".join(str(text or "").split()).strip()
    if len(compact) <= int(limit):
        return compact
    return compact[: max(0, int(limit) - 3)].rstrip() + "..."


def _extract_current_frame(image: Any) -> np.ndarray:
    x = np.asarray(image)
    if x.ndim == 4:
        # [T, H, W, C] or [T, C, H, W] -> take current frame
        x = x[-1]
    if x.ndim == 3 and x.shape[0] == 3 and x.shape[-1] != 3:
        x = np.transpose(x, (1, 2, 0))  # CHW -> HWC
    if x.ndim != 3 or x.shape[-1] != 3:
        raise ValueError(f"Expected RGB image frame (HWC), got shape: {x.shape}")
    if np.issubdtype(x.dtype, np.floating):
        x = np.clip(x, 0.0, 1.0)
        x = (x * 255.0).astype(np.uint8)
    return x


def _compose_low_level_prompt(
    *,
    goal: str,
    subtask: str,
    language_memory: str,
    structured_state: dict[str, Any] | None = None,
    retrieved_semantic_hint: str = "",
    retrieved_visual_hint: str = "",
) -> str:
    lines = [
        f"Task: {str(goal).strip()}",
        f"Subtask: {str(subtask).strip()}",
    ]
    if str(language_memory or "").strip():
        lines.append(f"Memory: {_compact_line(language_memory, limit=220)}")
    if structured_state:
        state_summary = "; ".join(f"{k}={v}" for k, v in structured_state.items() if v)
        if state_summary:
            lines.append(f"State: {_compact_line(state_summary, limit=220)}")
    if str(retrieved_semantic_hint or "").strip():
        lines.append(f"Retrieved semantic cue: {_compact_line(retrieved_semantic_hint, limit=180)}")
    if str(retrieved_visual_hint or "").strip():
        lines.append(f"Retrieved visual cue: {_compact_line(retrieved_visual_hint, limit=180)}")
    return "\n".join(lines).strip()


class MemPi05DroidAgent:
    """MEM wrapper around the pi0.5 low-level policy.

    - Short-term memory: maintains a small window of recent frames and passes them as a video stack.
    - Long-term memory: maintains a language memory string and uses a pretrained VLM to update it + emit subtasks.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        *,
        device: str | None = None,
        tokenizer_model: str | None = None,
        video_window: int = 1,
        hl_vlm_dir: str | None = None,
        hl_device: str | None = None,
        hl_dtype: str = "bfloat16",
        hl_revision: str | None = None,
        hl_max_new_tokens: int = 128,
        hl_temperature: float = 0.0,
    ) -> None:
        self.low_level = Pi05DroidPolicy(checkpoint_dir, device=device, tokenizer_model=tokenizer_model)
        self.video_window = int(video_window)
        if self.video_window < 1:
            raise ValueError("video_window must be >= 1")

        self._frame_buffers = {
            k: collections.deque(maxlen=self.video_window) for k in _DROID_IMAGE_KEYS
        }

        self.language_memory = ""

        self._hl: PretrainedVlmLongTermMemoryProcessor | None = None
        if hl_vlm_dir and str(hl_vlm_dir).strip():
            self._hl = PretrainedVlmLongTermMemoryProcessor(
                hl_vlm_dir,
                device=str(hl_device) if hl_device else str(self.low_level.device),
                dtype=hl_dtype,
                revision=hl_revision,
                max_new_tokens=int(hl_max_new_tokens),
                temperature=float(hl_temperature),
            )

    def reset(self, *, language_memory: str = "") -> None:
        self.language_memory = str(language_memory or "")
        for buf in self._frame_buffers.values():
            buf.clear()
        if self._hl is not None and hasattr(self._hl, "reset"):
            self._hl.reset()

    def _with_video_stack(self, example: dict[str, Any]) -> dict[str, Any]:
        if self.video_window <= 1:
            return example

        out = dict(example)
        for key in _DROID_IMAGE_KEYS:
            frame = _extract_current_frame(out[key])
            buf = self._frame_buffers[key]
            buf.append(frame)
            frames = list(buf)
            if len(frames) < self.video_window:
                frames = [frames[0]] * (self.video_window - len(frames)) + frames
            out[key] = np.stack(frames, axis=0)  # [T, H, W, C], oldest -> current
        return out

    def step(self, example: dict[str, Any], *, num_steps: int = 10, debug: bool = False) -> dict[str, Any]:
        goal = str(example.get("prompt", "")).strip()
        prev_memory = self.language_memory

        ex = self._with_video_stack(example)
        low_level_prompt = str(ex.get("prompt", "")).strip()
        structured_state: dict[str, Any] = {}
        retrieved_semantic_summary = ""
        retrieved_visual_summary = ""
        retrieved_semantic_hint = ""
        retrieved_visual_hint = ""
        pcmb_debug: dict[str, Any] = {}

        subtask = goal
        hl_result: LongTermMemoryResult | None = None
        if self._hl is not None:
            # Let the high-level policy consume the recent video stack instead of only the latest frame.
            base_img = ex[_DROID_IMAGE_KEYS[0]]
            hl_result = self._hl.update(goal=goal, prev_memory=self.language_memory, image=base_img)
            self.language_memory = hl_result.memory
            subtask = hl_result.subtask
            structured_state = dict(hl_result.structured_state)
            retrieved_semantic_summary = str(hl_result.retrieved_semantic_summary)
            retrieved_visual_summary = str(hl_result.retrieved_visual_summary)
            retrieved_semantic_hint = str(hl_result.retrieved_semantic_hint)
            retrieved_visual_hint = str(hl_result.retrieved_visual_hint)
            pcmb_debug = dict(hl_result.pcmb_debug)

            ex = dict(ex)
            ex["prompt"] = _compose_low_level_prompt(
                goal=goal,
                subtask=subtask,
                language_memory=self.language_memory,
                structured_state=structured_state,
                retrieved_semantic_hint=retrieved_semantic_hint,
                retrieved_visual_hint=retrieved_visual_hint,
            )
            low_level_prompt = str(ex.get("prompt", "")).strip()

        out = self.low_level.infer(ex, num_steps=int(num_steps))
        out["language_memory"] = self.language_memory
        out["subtask"] = subtask
        if hl_result is not None:
            out["hl_raw_text"] = hl_result.raw_text
            out["structured_state"] = structured_state
            out["retrieved_semantic_summary"] = retrieved_semantic_summary
            out["retrieved_visual_summary"] = retrieved_visual_summary
            out["retrieved_semantic_hint"] = retrieved_semantic_hint
            out["retrieved_visual_hint"] = retrieved_visual_hint
            out["pcmb_debug"] = pcmb_debug
            out["done"] = bool(hl_result.done)
            out["completion_reason"] = str(hl_result.completion_reason)
            out["completion_score"] = hl_result.completion_score
        else:
            out["structured_state"] = {}
            out["retrieved_semantic_summary"] = ""
            out["retrieved_visual_summary"] = ""
            out["retrieved_semantic_hint"] = ""
            out["retrieved_visual_hint"] = ""
            out["pcmb_debug"] = {}
            out["done"] = False
            out["completion_reason"] = ""
            out["completion_score"] = None
        if debug:
            out["_viz_debug"] = {
                "goal": goal,
                "low_level_prompt": low_level_prompt,
                "prev_memory": prev_memory,
                "structured_state": structured_state,
                "retrieved_semantic_summary": retrieved_semantic_summary,
                "retrieved_visual_summary": retrieved_visual_summary,
                "pcmb_debug": pcmb_debug,
                "images": {k: ex.get(k) for k in _DROID_IMAGE_KEYS},
            }
        return out
