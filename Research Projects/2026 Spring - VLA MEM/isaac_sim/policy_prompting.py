from __future__ import annotations

import os
from typing import Any


_FOLD_CONTROL_PHASE_LABELS = {
    "approach_left_sleeve": "approach left sleeve",
    "grasp_left_sleeve": "grasp left sleeve cuff",
    "fold_left_sleeve": "fold left sleeve inward",
    "release_left_sleeve": "release left sleeve and retreat",
    "approach_right_sleeve": "approach right sleeve",
    "grasp_right_sleeve": "grasp right sleeve cuff",
    "fold_right_sleeve": "fold right sleeve inward",
    "release_right_sleeve": "release right sleeve and retreat",
    "approach_lower_hem": "approach lower hem corners",
    "grasp_lower_hem": "grasp lower hem corners",
    "lift_lower_hem": "lift lower hem slightly",
    "bring_lower_hem_to_center": "bring lower hem toward center",
    "lay_lower_hem_flat": "push lower hem toward center seam with clearance",
    "lift_clear_before_release_lower_hem": "lift lower hem clear while still grasping",
    "release_lower_hem": "release lower hem and hold still",
    "retreat_after_release_lower_hem": "retreat after lower hem release",
    "flatten_square": "gently sweep side flaps inward and flatten",
    "inspect_finish": "inspect square fold and finish",
}

_DEFAULT_GOAL_ONLY_FOLD_TOPS_PROMPT = (
    "Use two robot arms to fold the shirt into a neat compact square. Start by visually aligning and flattening "
    "the garment on the table. Fold the left sleeve inward toward the center of the shirt, then fold the right "
    "sleeve inward toward the center, while keeping the cloth low and controlled. Next, grasp the lower hem, lift "
    "it only as much as needed, and fold the lower part of the shirt upward toward the center or upper body so the "
    "shirt becomes a compact rectangular or square block. Finish by gently pressing and aligning the folded shirt "
    "so the edges look tidy, symmetric, and stable. Use the current visual observation to decide the next local "
    "motion. If the shirt already appears neatly folded with the sleeves tucked in and the lower hem folded up, "
    "stop making large manipulation motions and only keep the folded shirt stable."
)


def default_goal_only_fold_tops_prompt() -> str:
    return _DEFAULT_GOAL_ONLY_FOLD_TOPS_PROMPT


def _resolve_low_level_prompt_style(prompt_style: str = "") -> str:
    style = str(prompt_style or "").strip().lower()
    if not style:
        style = str(os.environ.get("MYVLA_LL_PROMPT_STYLE", "phase_structured")).strip().lower()
    if style in {"goal_only", "goal", "no_phase", "no_subtask"}:
        return "goal_only"
    return "phase_structured"


def _compact_line(text: str, *, limit: int = 180) -> str:
    compact = " ".join(str(text or "").split()).strip()
    if len(compact) <= int(limit):
        return compact
    return compact[: max(0, int(limit) - 3)].rstrip() + "..."


def fold_phase_name(step: int) -> str:
    step = int(step)
    if step < 8:
        return "approach left sleeve"
    if step < 12:
        return "grasp left sleeve cuff"
    if step < 24:
        return "fold left sleeve inward"
    if step < 28:
        return "release left sleeve and retreat"
    if step < 36:
        return "approach right sleeve"
    if step < 40:
        return "grasp right sleeve cuff"
    if step < 48:
        return "fold right sleeve inward"
    if step < 52:
        return "release right sleeve and retreat"
    if step < 60:
        return "approach lower hem corners"
    if step < 64:
        return "grasp lower hem corners"
    if step < 68:
        return "lift lower hem slightly"
    if step < 74:
        return "bring lower hem toward center"
    if step < 80:
        return "push lower hem toward center seam with clearance"
    if step < 84:
        return "lift lower hem clear while still grasping"
    if step < 92:
        return "release lower hem and hold still"
    if step < 98:
        return "retreat after lower hem release"
    if step < 104:
        return "gently sweep side flaps inward and flatten"
    return "inspect square fold and finish"


def fold_control_phase_name(phase_key: str) -> str:
    return _FOLD_CONTROL_PHASE_LABELS.get(str(phase_key or "").strip(), "inspect square fold and finish")


def _fallback_control_phase_from_hint(phase_name: str) -> str:
    phase_l = str(phase_name or "").strip().lower()
    if "left sleeve" in phase_l and "approach" in phase_l:
        return "approach_left_sleeve"
    if "left sleeve" in phase_l and "grasp" in phase_l:
        return "grasp_left_sleeve"
    if "left sleeve" in phase_l and "fold" in phase_l:
        return "fold_left_sleeve"
    if "left sleeve" in phase_l and "release" in phase_l:
        return "release_left_sleeve"
    if "right sleeve" in phase_l and "approach" in phase_l:
        return "approach_right_sleeve"
    if "right sleeve" in phase_l and "grasp" in phase_l:
        return "grasp_right_sleeve"
    if "right sleeve" in phase_l and "fold" in phase_l:
        return "fold_right_sleeve"
    if "right sleeve" in phase_l and "release" in phase_l:
        return "release_right_sleeve"
    if "lower hem" in phase_l and "approach" in phase_l:
        return "approach_lower_hem"
    if "lower hem" in phase_l and "grasp" in phase_l:
        return "grasp_lower_hem"
    if "lower hem" in phase_l and "clear while still grasping" in phase_l:
        return "lift_clear_before_release_lower_hem"
    if "lower hem" in phase_l and "retreat" in phase_l:
        return "retreat_after_release_lower_hem"
    if "lower hem" in phase_l and "hold still" in phase_l:
        return "release_lower_hem"
    if "lower hem" in phase_l and "lift" in phase_l:
        return "lift_lower_hem"
    if "lower hem" in phase_l and "bring" in phase_l:
        return "bring_lower_hem_to_center"
    if "lower hem" in phase_l and "push" in phase_l:
        return "lay_lower_hem_flat"
    if "lower hem" in phase_l and "lay" in phase_l:
        return "lay_lower_hem_flat"
    if "lower hem" in phase_l and "release" in phase_l:
        return "release_lower_hem"
    if "flatten" in phase_l or "square" in phase_l or "press" in phase_l:
        return "flatten_square"
    if "inspect" in phase_l or "finish" in phase_l:
        return "inspect_finish"
    return "approach_left_sleeve"


def infer_fold_control_phase(
    *,
    subtask: str,
    structured_state: dict[str, Any] | None = None,
    fallback_phase_name: str = "",
) -> str:
    state = dict(structured_state or {})
    next_focus = str(state.get("next_focus", "") or "").strip()
    fold_stage = str(state.get("fold_stage", "") or "").strip()
    shape = str(state.get("shape", "") or "").strip()
    stability = str(state.get("stability", "") or "").strip()

    text = " ".join(
        part
        for part in [
            str(subtask or "").strip(),
            next_focus,
            fold_stage,
            shape,
            stability,
        ]
        if part
    ).lower()

    if not text:
        return _fallback_control_phase_from_hint(fallback_phase_name)

    if any(token in text for token in ("task complete", "done", "finished", "already folded")):
        return "inspect_finish"

    if "left" in text and "sleeve" in text:
        if any(token in text for token in ("approach", "hover", "move above", "move to")):
            return "approach_left_sleeve"
        if any(token in text for token in ("release", "retreat", "lift clear", "back away")):
            return "release_left_sleeve"
        if any(token in text for token in ("grasp", "pinch", "close", "cuff")):
            return "grasp_left_sleeve"
        if any(token in text for token in ("fold", "drag", "bring", "tuck")):
            return "fold_left_sleeve"
        return "approach_left_sleeve"

    if "right" in text and "sleeve" in text:
        if any(token in text for token in ("approach", "hover", "move above", "move to")):
            return "approach_right_sleeve"
        if any(token in text for token in ("release", "retreat", "lift clear", "back away")):
            return "release_right_sleeve"
        if any(token in text for token in ("grasp", "pinch", "close", "cuff")):
            return "grasp_right_sleeve"
        if any(token in text for token in ("fold", "drag", "bring", "tuck")):
            return "fold_right_sleeve"
        return "approach_right_sleeve"

    if any(token in text for token in ("hem", "bottom", "lower edge", "lower shirt", "bottom edge")):
        if any(token in text for token in ("retreat after release", "retreat after lower hem release")):
            return "retreat_after_release_lower_hem"
        if any(token in text for token in ("lift clear while still grasping", "still grasping", "while still grasping")):
            return "lift_clear_before_release_lower_hem"
        if any(token in text for token in ("push", "center seam", "with clearance", "lay flat")):
            return "lay_lower_hem_flat"
        if any(token in text for token in ("release", "lift clear", "open gripper", "retreat")):
            return "release_lower_hem"
        if any(token in text for token in ("bring", "fold", "tuck", "toward center", "to center")):
            return "bring_lower_hem_to_center"
        if "lift" in text:
            return "lift_lower_hem"
        if any(token in text for token in ("grasp", "pinch", "close")):
            return "grasp_lower_hem"
        return "approach_lower_hem"

    if any(token in text for token in ("square", "flatten", "align", "press", "refine")):
        if any(token in text for token in ("inspect", "finish", "stable")):
            return "inspect_finish"
        return "flatten_square"

    return _fallback_control_phase_from_hint(fallback_phase_name)


def merge_subtask_with_phase(*, goal: str, subtask: str, phase_name: str) -> str:
    goal_l = str(goal or "").lower()
    subtask = str(subtask or "").strip()
    phase_name = str(phase_name or "").strip()
    if not subtask:
        return phase_name
    compact_goal = " ".join(str(goal or "").split()).strip().lower()
    compact_subtask = " ".join(subtask.split()).strip().lower()
    if compact_subtask == compact_goal:
        return phase_name
    if phase_name and phase_name.lower() == compact_subtask:
        return phase_name
    if compact_subtask.startswith(compact_goal) and compact_goal:
        return phase_name
    if "fold" in goal_l and "shirt" in goal_l and compact_subtask in {"continue", "keep folding", "next step", "proceed"}:
        return phase_name
    return subtask


def compose_low_level_prompt(
    *,
    goal: str,
    subtask: str,
    language_memory: str,
    arm_side: str,
    phase_name: str,
    structured_state_summary: str = "",
    retrieved_semantic_hint: str = "",
    retrieved_visual_hint: str = "",
    prompt_style: str = "",
) -> str:
    memory_text = str(language_memory or "").strip()
    arm_role = "left arm controls the left half of the shirt" if arm_side == "left" else "right arm controls the right half of the shirt"
    resolved_goal = _compact_line(str(goal or "").strip() or default_goal_only_fold_tops_prompt(), limit=700)
    style = _resolve_low_level_prompt_style(prompt_style)
    if style == "goal_only":
        lines = [
            f"Task: {resolved_goal}",
            f"Role: {arm_role}.",
            "No stage label or subtask label is provided; infer the next local motion directly from the current observation.",
            "Move smoothly, keep the cloth close to the table, and avoid sudden outward pulls or large vertical lifts.",
            "Use the current image to decide where to align the wrist, how low to descend, and when to close or open the hand.",
        ]
        if memory_text:
            lines.append(f"Memory: {_compact_line(memory_text, limit=180)}")
        if structured_state_summary:
            lines.append(f"State: {_compact_line(structured_state_summary, limit=180)}")
        if retrieved_semantic_hint:
            lines.append(f"Retrieved semantic cue: {_compact_line(retrieved_semantic_hint, limit=160)}")
        if retrieved_visual_hint:
            lines.append(f"Retrieved visual cue: {_compact_line(retrieved_visual_hint, limit=160)}")
        return "\n".join(lines).strip()
    lines = [
        f"Task: {resolved_goal}",
        f"Phase: {str(phase_name).strip()}",
        f"Subtask: {str(subtask).strip()}",
        f"Role: {arm_role}.",
        "Move smoothly over the cloth, stay low near the shirt surface, and coordinate toward the table center line.",
        "Goal shape: fold the shirt into a compact square block with aligned left/right edges and a tucked lower hem.",
        "Prefer manipulations that clearly tuck each sleeve toward the center before lifting the lower hem.",
        "During the lower-hem phases, drag and lay the shirt flat near the table; do not lift the whole garment vertically.",
        "After laying the hem down, release and lift clear before pressing inward again; avoid dragging the cloth outward after release.",
        "Finish by pressing the folded shirt until the silhouette looks compact, rectangular, and then square.",
    ]
    if memory_text:
        lines.append(f"Memory: {_compact_line(memory_text, limit=180)}")
    if structured_state_summary:
        lines.append(f"State: {_compact_line(structured_state_summary, limit=180)}")
    if retrieved_semantic_hint:
        lines.append(f"Retrieved semantic cue: {_compact_line(retrieved_semantic_hint, limit=160)}")
    if retrieved_visual_hint:
        lines.append(f"Retrieved visual cue: {_compact_line(retrieved_visual_hint, limit=160)}")
    return "\n".join(lines).strip()


def compose_bimanual_low_level_prompt(
    *,
    goal: str,
    subtask: str,
    language_memory: str,
    phase_name: str,
    structured_state_summary: str = "",
    retrieved_semantic_hint: str = "",
    retrieved_visual_hint: str = "",
    prompt_style: str = "",
) -> str:
    memory_text = str(language_memory or "").strip()
    resolved_goal = _compact_line(str(goal or "").strip() or default_goal_only_fold_tops_prompt(), limit=700)
    style = _resolve_low_level_prompt_style(prompt_style)
    if style == "goal_only":
        lines = [
            f"Task: {resolved_goal}",
            "Control both robot arms together as one coordinated bimanual policy.",
            "No stage label or subtask label is provided; infer the next local motion directly from the current observation.",
            "Output absolute joint targets for both UR10e+Shadow-Hand systems in a single synchronized action.",
            "Use the left and right shirt observations to decide where both wrists should move, how low each hand should descend, and when the fingers should close or open.",
            "Keep the cloth close to the table, avoid sudden outward pulls, and avoid large vertical lifts unless the current image clearly requires a brief lift.",
        ]
        if memory_text:
            lines.append(f"Memory: {_compact_line(memory_text, limit=180)}")
        if structured_state_summary:
            lines.append(f"State: {_compact_line(structured_state_summary, limit=180)}")
        if retrieved_semantic_hint:
            lines.append(f"Retrieved semantic cue: {_compact_line(retrieved_semantic_hint, limit=160)}")
        if retrieved_visual_hint:
            lines.append(f"Retrieved visual cue: {_compact_line(retrieved_visual_hint, limit=160)}")
        return "\n".join(lines).strip()
    lines = [
        f"Task: {resolved_goal}",
        f"Phase: {str(phase_name).strip()}",
        f"Subtask: {str(subtask).strip()}",
        "Control both robot arms together as one coordinated bimanual policy.",
        "Output absolute joint targets for both UR10e+Shadow-Hand systems in a single synchronized action.",
        "Use the left and right shirt observations to align both wrists with the cloth before grasping.",
        "Keep the hands low near the cloth surface, avoid lifting the whole garment vertically, and maintain tension symmetry.",
        "Goal shape: fold the shirt into a compact square block with both sleeves tucked inward and the lower hem folded upward.",
    ]
    if memory_text:
        lines.append(f"Memory: {_compact_line(memory_text, limit=180)}")
    if structured_state_summary:
        lines.append(f"State: {_compact_line(structured_state_summary, limit=180)}")
    if retrieved_semantic_hint:
        lines.append(f"Retrieved semantic cue: {_compact_line(retrieved_semantic_hint, limit=160)}")
    if retrieved_visual_hint:
        lines.append(f"Retrieved visual cue: {_compact_line(retrieved_visual_hint, limit=160)}")
    return "\n".join(lines).strip()
