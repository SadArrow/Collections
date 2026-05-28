from __future__ import annotations

import dataclasses
from typing import Any

import torch


_GEOMETRY_KEYS = (
    "foreground_ratio",
    "bbox_xmin",
    "bbox_ymin",
    "bbox_xmax",
    "bbox_ymax",
    "center_x",
    "center_y",
    "width_ratio",
    "height_ratio",
    "aspect_ratio",
    "left_mass",
    "right_mass",
    "top_mass",
    "bottom_mass",
    "edge_density",
    "motion_cloth",
    "motion_overview",
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if out != out or out in (float("inf"), float("-inf")):
        return float(default)
    return out


def _normalize_cpu_vector(vec: torch.Tensor | Any) -> torch.Tensor:
    if torch.is_tensor(vec):
        out = vec.detach().to(dtype=torch.float32, device="cpu").reshape(-1)
    else:
        out = torch.as_tensor(vec, dtype=torch.float32).reshape(-1)
    norm = torch.linalg.vector_norm(out)
    if float(norm) > 1e-6:
        out = out / norm
    return out


def geometry_dict_to_vector(geometry: dict[str, Any] | None) -> torch.Tensor:
    data = dict(geometry or {})
    return torch.tensor([_safe_float(data.get(key, 0.0)) for key in _GEOMETRY_KEYS], dtype=torch.float32)


def _cosine(a: torch.Tensor | None, b: torch.Tensor | None) -> float:
    if a is None or b is None:
        return 0.0
    if a.numel() == 0 or b.numel() == 0:
        return 0.0
    aa = _normalize_cpu_vector(a)
    bb = _normalize_cpu_vector(b)
    if aa.shape != bb.shape:
        size = min(int(aa.numel()), int(bb.numel()))
        aa = aa[:size]
        bb = bb[:size]
    return _safe_float(torch.dot(aa, bb).item(), default=0.0)


def _truncate(text: str, limit: int = 220) -> str:
    compact = " ".join(str(text or "").split()).strip()
    if len(compact) <= int(limit):
        return compact
    return compact[: max(0, int(limit) - 3)].rstrip() + "..."


def serialize_structured_state(state: dict[str, Any] | None) -> str:
    data = dict(state or {})
    ordered_keys = (
        "fold_stage",
        "left_sleeve",
        "right_sleeve",
        "hem",
        "shape",
        "stability",
        "last_effect",
        "next_focus",
    )
    parts = []
    for key in ordered_keys:
        value = str(data.get(key, "") or "").strip()
        if value:
            parts.append(f"{key}={value}")
    return "; ".join(parts)


@dataclasses.dataclass
class CognitiveMemoryEntry:
    step_start: int
    step_end: int
    embedding: torch.Tensor
    state: dict[str, Any]
    summary: str
    completion_score: float | None = None


@dataclasses.dataclass
class PerceptualMemoryEntry:
    step_start: int
    step_end: int
    embedding: torch.Tensor
    geometry_vector: torch.Tensor
    geometry: dict[str, Any]
    evidence: str


@dataclasses.dataclass
class RetrievedMemoryHit:
    step_start: int
    step_end: int
    score: float
    summary: str
    payload: dict[str, Any]


@dataclasses.dataclass
class RetrievedMemoryContext:
    cognitive_hits: list[RetrievedMemoryHit] = dataclasses.field(default_factory=list)
    perceptual_hits: list[RetrievedMemoryHit] = dataclasses.field(default_factory=list)
    semantic_prompt: str = ""
    visual_prompt: str = ""
    semantic_hint: str = ""
    visual_hint: str = ""

    def as_debug_dict(self) -> dict[str, Any]:
        return {
            "cognitive_hits": [
                {
                    "step_start": int(hit.step_start),
                    "step_end": int(hit.step_end),
                    "score": float(hit.score),
                    "summary": str(hit.summary),
                    "payload": dict(hit.payload),
                }
                for hit in self.cognitive_hits
            ],
            "perceptual_hits": [
                {
                    "step_start": int(hit.step_start),
                    "step_end": int(hit.step_end),
                    "score": float(hit.score),
                    "summary": str(hit.summary),
                    "payload": dict(hit.payload),
                }
                for hit in self.perceptual_hits
            ],
            "semantic_prompt": str(self.semantic_prompt),
            "visual_prompt": str(self.visual_prompt),
            "semantic_hint": str(self.semantic_hint),
            "visual_hint": str(self.visual_hint),
        }


class PerceptualCognitiveMemoryBank:
    def __init__(
        self,
        *,
        cognitive_capacity: int = 24,
        perceptual_capacity: int = 24,
        top_k: int = 3,
        perceptual_geometry_weight: float = 0.25,
    ) -> None:
        self.cognitive_capacity = max(1, int(cognitive_capacity))
        self.perceptual_capacity = max(1, int(perceptual_capacity))
        self.top_k = max(1, int(top_k))
        self.perceptual_geometry_weight = float(min(max(perceptual_geometry_weight, 0.0), 1.0))
        self.reset()

    def reset(self) -> None:
        self._cognitive_entries: list[CognitiveMemoryEntry] = []
        self._perceptual_entries: list[PerceptualMemoryEntry] = []

    @property
    def cognitive_size(self) -> int:
        return len(self._cognitive_entries)

    @property
    def perceptual_size(self) -> int:
        return len(self._perceptual_entries)

    def snapshot(self) -> dict[str, Any]:
        return {
            "cognitive_size": int(self.cognitive_size),
            "perceptual_size": int(self.perceptual_size),
            "cognitive_capacity": int(self.cognitive_capacity),
            "perceptual_capacity": int(self.perceptual_capacity),
        }

    def retrieve(
        self,
        *,
        cognitive_query: torch.Tensor | None,
        perceptual_query: torch.Tensor | None,
        geometry_query: torch.Tensor | None,
    ) -> RetrievedMemoryContext:
        cognitive_hits: list[RetrievedMemoryHit] = []
        perceptual_hits: list[RetrievedMemoryHit] = []

        if cognitive_query is not None:
            ranked_cognitive = sorted(
                (
                    (
                        _cosine(cognitive_query, entry.embedding),
                        entry,
                    )
                    for entry in self._cognitive_entries
                ),
                key=lambda item: item[0],
                reverse=True,
            )[: self.top_k]
            for score, entry in ranked_cognitive:
                cognitive_hits.append(
                    RetrievedMemoryHit(
                        step_start=int(entry.step_start),
                        step_end=int(entry.step_end),
                        score=float(score),
                        summary=str(entry.summary),
                        payload={
                            "structured_state": dict(entry.state),
                            "completion_score": entry.completion_score,
                        },
                    )
                )

        if perceptual_query is not None:
            ranked_perceptual = sorted(
                (
                    (
                        (1.0 - self.perceptual_geometry_weight) * _cosine(perceptual_query, entry.embedding)
                        + self.perceptual_geometry_weight * _cosine(geometry_query, entry.geometry_vector),
                        entry,
                    )
                    for entry in self._perceptual_entries
                ),
                key=lambda item: item[0],
                reverse=True,
            )[: self.top_k]
            for score, entry in ranked_perceptual:
                perceptual_hits.append(
                    RetrievedMemoryHit(
                        step_start=int(entry.step_start),
                        step_end=int(entry.step_end),
                        score=float(score),
                        summary=str(entry.evidence),
                        payload={"geometry": dict(entry.geometry)},
                    )
                )

        semantic_prompt = "\n".join(
            f"- steps {hit.step_start}-{hit.step_end}, score={hit.score:.3f}: {hit.summary}" for hit in cognitive_hits
        )
        visual_prompt = "\n".join(
            f"- steps {hit.step_start}-{hit.step_end}, score={hit.score:.3f}: {hit.summary}" for hit in perceptual_hits
        )

        semantic_hint = _truncate(cognitive_hits[0].summary, limit=180) if cognitive_hits else ""
        visual_hint = _truncate(perceptual_hits[0].summary, limit=180) if perceptual_hits else ""

        return RetrievedMemoryContext(
            cognitive_hits=cognitive_hits,
            perceptual_hits=perceptual_hits,
            semantic_prompt=semantic_prompt,
            visual_prompt=visual_prompt,
            semantic_hint=semantic_hint,
            visual_hint=visual_hint,
        )

    def add_cognitive(
        self,
        *,
        step: int,
        embedding: torch.Tensor,
        state: dict[str, Any],
        summary: str,
        completion_score: float | None,
    ) -> None:
        self._cognitive_entries.append(
            CognitiveMemoryEntry(
                step_start=int(step),
                step_end=int(step),
                embedding=_normalize_cpu_vector(embedding),
                state=dict(state),
                summary=_truncate(summary, limit=240),
                completion_score=_safe_float(completion_score, default=0.0) if completion_score is not None else None,
            )
        )
        while len(self._cognitive_entries) > self.cognitive_capacity:
            self._merge_best_adjacent_cognitive()

    def add_perceptual(
        self,
        *,
        step: int,
        embedding: torch.Tensor,
        geometry: dict[str, Any],
        evidence: str,
    ) -> None:
        self._perceptual_entries.append(
            PerceptualMemoryEntry(
                step_start=int(step),
                step_end=int(step),
                embedding=_normalize_cpu_vector(embedding),
                geometry_vector=_normalize_cpu_vector(geometry_dict_to_vector(geometry)),
                geometry=dict(geometry),
                evidence=_truncate(evidence, limit=240),
            )
        )
        while len(self._perceptual_entries) > self.perceptual_capacity:
            self._merge_best_adjacent_perceptual()

    def _merge_best_adjacent_cognitive(self) -> None:
        if len(self._cognitive_entries) < 2:
            return
        best_index = 0
        best_score = -1.0
        for index in range(len(self._cognitive_entries) - 1):
            score = _cosine(self._cognitive_entries[index].embedding, self._cognitive_entries[index + 1].embedding)
            if score > best_score:
                best_score = score
                best_index = index
        left = self._cognitive_entries[best_index]
        right = self._cognitive_entries[best_index + 1]
        merged_state = dict(left.state)
        for key, value in right.state.items():
            if value not in ("", None, "unknown", "uncertain"):
                merged_state[key] = value
        merged = CognitiveMemoryEntry(
            step_start=int(left.step_start),
            step_end=int(right.step_end),
            embedding=_normalize_cpu_vector(0.5 * (left.embedding + right.embedding)),
            state=merged_state,
            summary=_truncate(f"{left.summary} | {right.summary}", limit=240),
            completion_score=right.completion_score if right.completion_score is not None else left.completion_score,
        )
        self._cognitive_entries[best_index] = merged
        self._cognitive_entries.pop(best_index + 1)

    def _merge_best_adjacent_perceptual(self) -> None:
        if len(self._perceptual_entries) < 2:
            return
        best_index = 0
        best_score = -1.0
        for index in range(len(self._perceptual_entries) - 1):
            emb_score = _cosine(self._perceptual_entries[index].embedding, self._perceptual_entries[index + 1].embedding)
            geom_score = _cosine(
                self._perceptual_entries[index].geometry_vector,
                self._perceptual_entries[index + 1].geometry_vector,
            )
            score = (1.0 - self.perceptual_geometry_weight) * emb_score + self.perceptual_geometry_weight * geom_score
            if score > best_score:
                best_score = score
                best_index = index
        left = self._perceptual_entries[best_index]
        right = self._perceptual_entries[best_index + 1]
        merged_geometry = dict(left.geometry)
        for key in _GEOMETRY_KEYS:
            merged_geometry[key] = 0.5 * (_safe_float(left.geometry.get(key, 0.0)) + _safe_float(right.geometry.get(key, 0.0)))
        merged = PerceptualMemoryEntry(
            step_start=int(left.step_start),
            step_end=int(right.step_end),
            embedding=_normalize_cpu_vector(0.5 * (left.embedding + right.embedding)),
            geometry_vector=_normalize_cpu_vector(0.5 * (left.geometry_vector + right.geometry_vector)),
            geometry=merged_geometry,
            evidence=_truncate(f"{left.evidence} | {right.evidence}", limit=240),
        )
        self._perceptual_entries[best_index] = merged
        self._perceptual_entries.pop(best_index + 1)
