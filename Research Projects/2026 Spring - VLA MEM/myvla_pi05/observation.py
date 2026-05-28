from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import torch


@dataclasses.dataclass
class Observation:
    images: dict[str, Any]
    image_masks: dict[str, Any]
    state: Any
    tokenized_prompt: Any | None = None
    tokenized_prompt_mask: Any | None = None
    token_ar_mask: Any | None = None
    token_loss_mask: Any | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "Observation":
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together.")

        # Convert uint8 images -> float32 [-1, 1]. Support both numpy and torch tensors.
        for key in data["image"]:
            image = data["image"][key]
            if isinstance(image, np.ndarray) and image.dtype == np.uint8:
                data["image"][key] = image.astype(np.float32) / 255.0 * 2.0 - 1.0
            elif hasattr(image, "dtype") and image.dtype == torch.uint8:
                # torch: keep [B,C,H,W] (convert from [B,H,W,C] if needed later in preprocessing)
                if image.ndim == 4 and image.shape[-1] == 3:
                    image = image.permute(0, 3, 1, 2)
                elif image.ndim == 5 and image.shape[-1] == 3:
                    # Video: [B, T, H, W, C] -> [B, T, C, H, W]
                    image = image.permute(0, 1, 4, 2, 3)
                data["image"][key] = image.to(torch.float32) / 255.0 * 2.0 - 1.0

        return cls(
            images=data["image"],
            image_masks=data["image_mask"],
            state=data["state"],
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
        )
