from __future__ import annotations

import json
from typing import Any


FOLD_TOPS_CANONICAL_SUBTASKS: tuple[str, ...] = (
    "approach the left sleeve cuff from above",
    "grasp the left sleeve cuff",
    "fold the left sleeve inward toward the center",
    "release the left sleeve and retreat",
    "approach the right sleeve cuff from above",
    "grasp the right sleeve cuff",
    "fold the right sleeve inward toward the center",
    "release the right sleeve and retreat",
    "approach the lower hem corners",
    "grasp the lower hem corners",
    "lift the lower hem slightly",
    "bring the lower hem toward the center seam",
    "lay the lower hem flat near the center seam",
    "release the lower hem and lift clear",
    "flatten and square the folded shirt into a compact block",
    "inspect the square fold and prepare the next adjustment",
    "task complete",
)


def default_fold_tops_goal(goal: str = "") -> str:
    text = str(goal or "").strip()
    if text:
        return text
    return "Fold the garment into a compact square block with two robot arms."


def default_fold_tops_seed_memory(goal: str = "") -> str:
    _ = default_fold_tops_goal(goal)
    return ""


def _compact(text: str, *, limit: int = 320) -> str:
    compact = " ".join(str(text or "").split()).strip()
    if len(compact) <= int(limit):
        return compact
    return compact[: max(0, int(limit) - 3)].rstrip() + "..."


def build_fold_tops_hl_prompt(
    *,
    goal: str,
    prev_memory: str = "",
    phase_name: str = "",
    geometry_text: str = "",
    video_metrics: dict[str, Any] | None = None,
    retrieved_semantic: str = "",
    retrieved_visual: str = "",
) -> str:
    lines: list[str] = [
        "You are the high-level policy of a dual-arm robot for garment folding.",
        "Choose the next atomic subtask for the low-level controller.",
        "The SUBTASK line must copy exactly one item from the allowed subtask list below.",
        "Do not output a full plan or extra commentary.",
        "",
        "Goal g:",
        default_fold_tops_goal(goal),
        "",
    ]

    if str(phase_name).strip():
        lines.extend(
            [
                "Current controller context:",
                str(phase_name).strip(),
                "",
            ]
        )

    metrics = dict(video_metrics or {})
    if metrics:
        lines.extend(
            [
                "Recent video summary:",
                f"- frame_count: {int(metrics.get('frame_count', 1))}",
                f"- cloth_tail_mean_delta: {float(metrics.get('cloth_tail_mean_delta', 0.0)):.3f}",
                f"- cloth_last_delta: {float(metrics.get('cloth_last_delta', 0.0)):.3f}",
                f"- overview_tail_mean_delta: {float(metrics.get('overview_tail_mean_delta', 0.0)):.3f}",
                "",
            ]
        )

    if str(geometry_text).strip():
        lines.extend(
            [
                "Current cloth geometry estimate:",
                str(geometry_text).strip(),
                "",
            ]
        )

    if str(retrieved_semantic).strip():
        lines.extend(
            [
                "Retrieved semantic memory:",
                _compact(retrieved_semantic, limit=320),
                "",
            ]
        )

    if str(retrieved_visual).strip():
        lines.extend(
            [
                "Retrieved visual evidence:",
                _compact(retrieved_visual, limit=320),
                "",
            ]
        )

    lines.extend(
        [
            "Previous language memory m_t:",
            _compact(prev_memory if prev_memory else "<empty>", limit=420),
            "",
            "Allowed SUBTASK values:",
        ]
    )
    lines.extend(f"- {item}" for item in FOLD_TOPS_CANONICAL_SUBTASKS)
    lines.extend(
        [
            "",
            "Rules:",
            "- The SUBTASK line must be exactly one item from the allowed list above.",
            "- Keep MEMORY compact and factual. Mention only persistent cloth state and the next focus.",
            "- DONE=yes only when the garment already looks compact, square, and stable across recent frames.",
            "- If either sleeve is still outside, the lower hem is still down, or the cloth is moving, DONE must be no.",
            "",
            "Output EXACTLY 6 lines in this format:",
            'STATE: {"fold_stage":"...","left_sleeve":"...","right_sleeve":"...","hem":"...","shape":"...","stability":"...","last_effect":"...","next_focus":"..."}',
            "MEMORY: <your updated memory>",
            "SUBTASK: <one allowed subtask copied verbatim>",
            "DONE: <yes or no>",
            "COMPLETION: <0-100 confidence that the garment is already fully folded>",
            "REASON: <short completion rationale>",
            "Do not output anything else.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def render_fold_tops_hl_target(
    *,
    state: dict[str, Any],
    subtask: str,
    memory: str,
    done: bool,
    completion: int | float,
    reason: str,
) -> str:
    return "\n".join(
        [
            f"STATE: {json.dumps(state, ensure_ascii=False)}",
            f"MEMORY: {str(memory).strip()}",
            f"SUBTASK: {str(subtask).strip()}",
            f"DONE: {'yes' if bool(done) else 'no'}",
            f"COMPLETION: {int(round(float(completion)))}",
            f"REASON: {str(reason).strip()}",
        ]
    )

