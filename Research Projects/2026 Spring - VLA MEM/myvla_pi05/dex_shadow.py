from __future__ import annotations

import dataclasses

import einops
import numpy as np


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if image.ndim == 4 and image.shape[1] == 3:
        image = einops.rearrange(image, "t c h w -> t h w c")
    return image


def _to_float_state(data: object) -> np.ndarray:
    return np.asarray(data, dtype=np.float32).reshape(-1)


def _ensure_rgb_height(image: np.ndarray, *, target_h: int) -> np.ndarray:
    rgb = _parse_image(image)
    if rgb.shape[0] == int(target_h):
        return rgb
    from PIL import Image

    width = max(1, int(round(float(rgb.shape[1]) * float(target_h) / float(max(1, rgb.shape[0])))))
    return np.asarray(Image.fromarray(rgb).resize((width, int(target_h)), Image.Resampling.BILINEAR), dtype=np.uint8)


def _concat_side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_rgb = _parse_image(left)
    right_rgb = _parse_image(right)
    target_h = min(int(left_rgb.shape[0]), int(right_rgb.shape[0]))
    return np.concatenate(
        [
            _ensure_rgb_height(left_rgb, target_h=target_h),
            _ensure_rgb_height(right_rgb, target_h=target_h),
        ],
        axis=1,
    )


@dataclasses.dataclass(frozen=True)
class DexShadowInputs:
    """Convert a DexGarmentLab per-arm example into PI0.5 model inputs."""

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["observation/joint_position"], dtype=np.float32).reshape(-1)
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
class DexShadowOutputs:
    active_action_dim: int = 30

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : int(self.active_action_dim)], dtype=np.float32)}


@dataclasses.dataclass(frozen=True)
class DexBimanualInputs:
    """Convert a DexGarmentLab dual-arm example into PI0.5 model inputs."""

    def __call__(self, data: dict) -> dict:
        left_state = _to_float_state(data["observation/joint_position_left"])
        right_state = _to_float_state(data["observation/joint_position_right"])
        state = np.concatenate([left_state, right_state], axis=0).astype(np.float32)

        left_base = _parse_image(data["observation/exterior_image_1_left"])
        right_base = _parse_image(data.get("observation/exterior_image_1_right", left_base))
        left_wrist = _parse_image(data["observation/wrist_image_left"])
        right_wrist = _parse_image(data["observation/wrist_image_right"])

        names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        images = (_concat_side_by_side(left_base, right_base), left_wrist, right_wrist)
        image_masks = (np.True_, np.True_, np.True_)

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
class DexBimanualOutputs:
    active_action_dim: int = 60

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"], dtype=np.float32)
        return {"actions": actions[:, : int(self.active_action_dim)]}
