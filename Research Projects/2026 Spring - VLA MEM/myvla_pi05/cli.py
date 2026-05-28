from __future__ import annotations

import argparse
import os
import pathlib

import numpy as np

from . import droid
from .policy import Pi05DroidPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pi0.5 (pi05_droid) inference with a PyTorch checkpoint.")
    parser.add_argument(
        "--checkpoint_dir",
        default=str(pathlib.Path(__file__).resolve().parents[1] / "pi05_droid_pytorch"),
        help="Directory containing model.safetensors, config.json, and assets/*/norm_stats.json",
    )
    parser.add_argument("--device", default="", help="torch device, e.g. cuda / cuda:0 / cpu (default: auto)")
    parser.add_argument("--num_steps", type=int, default=10, help="Flow matching steps (default: 10)")
    parser.add_argument("--tokenizer_model", default="", help="Path to paligemma_tokenizer.model (optional)")
    parser.add_argument(
        "--video_window",
        type=int,
        default=1,
        help="Short-term video memory window (frames). >1 passes a frame stack to the SigLIP video encoder.",
    )
    args = parser.parse_args()

    device = args.device.strip() or None
    tokenizer_model = args.tokenizer_model.strip() or None

    policy = Pi05DroidPolicy(
        args.checkpoint_dir,
        device=device,
        tokenizer_model=tokenizer_model,
    )

    example = droid.make_droid_example()
    if int(args.video_window) > 1:
        window = int(args.video_window)
        for k in ("observation/exterior_image_1_left", "observation/wrist_image_left"):
            img = np.asarray(example[k])
            example[k] = np.stack([img] * window, axis=0)
    out = policy.infer(example, num_steps=args.num_steps)
    actions = out["actions"]
    print("Inference OK.")
    print(f"actions shape: {actions.shape}")
    print(f"actions dtype: {actions.dtype}")
    print(f"first action sample (first 5 dims): {actions.ravel()[:5]}")


if __name__ == "__main__":
    main()
