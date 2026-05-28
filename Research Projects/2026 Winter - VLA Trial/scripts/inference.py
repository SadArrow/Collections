#!/usr/bin/env python3
"""
Inference script for VLA model.
Usage: python -m scripts.inference --checkpoint path/to/checkpoint.pt
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vla_model import VLAAgent
from vla_model.type import RawVLAData


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--instruction", type=str, default="pick up the object")
    parser.add_argument("--image_front", type=str, default=None)
    parser.add_argument("--image_side", type=str, default=None)
    args = parser.parse_args()

    if args.checkpoint and os.path.exists(args.checkpoint):
        agent = VLAAgent(path=args.checkpoint, device=args.device)
    else:
        # Demo without checkpoint: create agent with default config
        agent = VLAAgent(path=None, device=args.device)

    # Build sample
    if args.image_front and os.path.exists(args.image_front):
        from PIL import Image
        front = np.array(Image.open(args.image_front).convert("RGB"))
    else:
        front = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    if args.image_side and os.path.exists(args.image_side):
        from PIL import Image
        side = np.array(Image.open(args.image_side).convert("RGB"))
    else:
        side = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    raw = RawVLAData(
        instruction=args.instruction,
        images={"front": front, "side": side},
        proprio=np.zeros((2, 7), dtype=np.float32),  # placeholder proprio
    )

    result = agent.sample_action(raw)
    print("Action:", result.get("action", "N/A"))
    return result


if __name__ == "__main__":
    main()
