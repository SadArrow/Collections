from __future__ import annotations

import dataclasses

import einops
import numpy as np


def make_droid_example() -> dict:
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(7),
        "observation/gripper_position": np.random.rand(1),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if image.ndim == 4 and image.shape[1] == 3:
        # Video, channels-first: [t, c, h, w] -> [t, h, w, c]
        image = einops.rearrange(image, "t c h w -> t h w c")
    return image


@dataclasses.dataclass(frozen=True)
class DroidInputs:
    """Convert a DROID-format example into PI0.5 model input dict."""

    def __call__(self, data: dict) -> dict:
        gripper_pos = np.asarray(data["observation/gripper_position"])
        if gripper_pos.ndim == 0:
            gripper_pos = gripper_pos[np.newaxis]
        state = np.concatenate([np.asarray(data["observation/joint_position"]), gripper_pos]).astype(np.float32)

        base_image = _parse_image(data["observation/exterior_image_1_left"])
        wrist_image = _parse_image(data["observation/wrist_image_left"])

        names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        images = (base_image, wrist_image, np.zeros_like(base_image))
        image_masks = (np.True_, np.True_, np.False_)

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = str(prompt)

        return inputs


@dataclasses.dataclass(frozen=True)
class DroidOutputs:
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :8])}
