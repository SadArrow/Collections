from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any

import numpy as np


@dataclasses.dataclass(frozen=True)
class NormStats:
    mean: np.ndarray
    std: np.ndarray
    q01: np.ndarray | None = None
    q99: np.ndarray | None = None


def load_norm_stats(path: str | pathlib.Path) -> dict[str, NormStats]:
    path = pathlib.Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if "norm_stats" not in data:
        raise ValueError(f"Invalid norm_stats.json: missing 'norm_stats' key: {path}")

    out: dict[str, NormStats] = {}
    for name, stats in data["norm_stats"].items():
        out[name] = NormStats(
            mean=np.asarray(stats["mean"], dtype=np.float32),
            std=np.asarray(stats["std"], dtype=np.float32),
            q01=np.asarray(stats["q01"], dtype=np.float32) if stats.get("q01") is not None else None,
            q99=np.asarray(stats["q99"], dtype=np.float32) if stats.get("q99") is not None else None,
        )
    return out


def pad_to_dim(x: np.ndarray, target_dim: int, *, axis: int = -1, value: float = 0.0) -> np.ndarray:
    current_dim = x.shape[axis]
    if current_dim >= target_dim:
        return x
    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (0, target_dim - current_dim)
    return np.pad(x, pad_width, constant_values=value)


def normalize(x: np.ndarray, stats: NormStats) -> np.ndarray:
    mean = stats.mean[..., : x.shape[-1]]
    std = stats.std[..., : x.shape[-1]]
    return (x - mean) / (std + 1e-6)


def unnormalize(x: np.ndarray, stats: NormStats) -> np.ndarray:
    mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
    std = pad_to_dim(stats.std, x.shape[-1], axis=-1, value=1.0)
    return x * (std + 1e-6) + mean


def clip_to_quantile_range(
    x: np.ndarray,
    stats: NormStats,
    *,
    margin_ratio: float = 0.0,
) -> np.ndarray:
    if stats.q01 is None or stats.q99 is None:
        return np.asarray(x, dtype=np.float32)
    low = pad_to_dim(np.asarray(stats.q01, dtype=np.float32), x.shape[-1], axis=-1, value=-np.inf)
    high = pad_to_dim(np.asarray(stats.q99, dtype=np.float32), x.shape[-1], axis=-1, value=np.inf)
    span = np.maximum(high - low, 1.0e-6)
    margin = np.asarray(span * float(max(0.0, margin_ratio)), dtype=np.float32)
    return np.clip(np.asarray(x, dtype=np.float32), low - margin, high + margin)


def apply_stats(data: dict[str, Any], stats: dict[str, NormStats], *, fn) -> dict[str, Any]:
    out = dict(data)
    for key, stat in stats.items():
        if key in out:
            out[key] = fn(np.asarray(out[key], dtype=np.float32), stat)
    return out
