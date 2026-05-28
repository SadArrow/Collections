from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812


def resize_with_pad_torch(
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """PyTorch resize_with_pad for tensors in [*b, h, w, c] or [*b, c, h, w].

    If float32, expects range [-1, 1]. If uint8, expects [0, 255].
    """
    original_dim = images.dim()

    # channels-last [..., h, w, c] (c <= 4) vs channels-first [..., c, h, w]
    if images.shape[-1] <= 4:
        channels_last = True
        if original_dim == 3:
            images = images.unsqueeze(0)  # [h,w,c] -> [1,h,w,c]
        leading_shape = images.shape[:-3]
        cur_height, cur_width, channels = images.shape[-3:]
        images = images.reshape(-1, cur_height, cur_width, channels).permute(0, 3, 1, 2)  # -> [b,c,h,w]
    else:
        channels_last = False
        if original_dim == 3:
            images = images.unsqueeze(0)  # [c,h,w] -> [1,c,h,w]
        leading_shape = images.shape[:-3]
        channels, cur_height, cur_width = images.shape[-3:]
        images = images.reshape(-1, channels, cur_height, cur_width)  # -> [b,c,h,w]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    resized = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )

    if images.dtype == torch.uint8:
        resized = torch.round(resized).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized = resized.clamp(-1.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    pad_h0, rem_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + rem_h
    pad_w0, rem_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + rem_w

    constant_value = 0 if images.dtype == torch.uint8 else -1.0
    padded = F.pad(resized, (pad_w0, pad_w1, pad_h0, pad_h1), mode="constant", value=constant_value)

    if channels_last:
        padded = padded.permute(0, 2, 3, 1)  # [b,c,h,w] -> [b,h,w,c]
        padded = padded.reshape(*leading_shape, height, width, channels)
        if original_dim == 3:
            padded = padded.squeeze(0)
    else:
        padded = padded.reshape(*leading_shape, channels, height, width)
        if original_dim == 3:
            padded = padded.squeeze(0)

    return padded
