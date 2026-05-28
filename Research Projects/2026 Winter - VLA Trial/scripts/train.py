#!/usr/bin/env python3
"""
Training script for VLA model.
Usage: python -m scripts.train --config configs/default.yaml
"""
import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vla_model import VLA
from vla_model.config import VLAModelConfig, VLADataConfig, ViTConfig, LLMConfig
from vla_model.data import DataPreprocessor, vla_collator


def get_default_config():
    return {
        "data": {
            "robot": "default",
            "proprio_len": 2,
            "action_len": 16,
            "action_dim": 7,
            "goal_dim": 6,
            "proprio_dim": 7,
            "action_rel_len": 1,
            "dt_steps": 1,
            "action_token_num": 256,
            "img_steps": 1,
            "img_key": ["front", "side"],
            "image_size": 224,
            "tokenizer_type": "ratio_min_max_uniform",
        },
        "model": {
            "backbone_2d": {"name": "vit_base_patch16_224", "image_size": 224, "pretrained": True},
            "llm": {"name": "Qwen/Qwen2-0.5B", "max_len": 2048, "attn_implementation": "eager"},
            "pred": "cot_flow_matching",
            "action_expert": True,
            "action_expert_cfg": {"hidden_size_scale": 2, "intermediate_size_scale": 2},
            "flow_matching_cfg": {
                "beta_alpha": 2.0, "beta_beta": 2.0,
                "time_min": 0.01, "time_max": 1.0,
            },
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    cfg = get_default_config()
    if args.config and os.path.exists(args.config):
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    data_cfg = VLADataConfig.model_validate(cfg["data"])
    model_cfg = VLAModelConfig.model_validate(cfg["model"])
    data_cfg.setup()

    # Build model (without tokenizer - will need to init with a placeholder)
    from transformers import AutoTokenizer
    data_cfg.tokenizer = AutoTokenizer.from_pretrained(model_cfg.llm.name, trust_remote_code=True)
    data_cfg.pred = model_cfg.pred

    model = VLA(model_cfg)
    model.init(train=True)

    # Example: create dummy dataset for structure demo
    # In practice, replace with your Dataset that yields RawVLAData
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 16

        def __getitem__(self, i):
            import numpy as np
            from vla_model.type import RawVLAData
            return RawVLAData(
                instruction="pick up the red block",
                images={"front": np.random.rand(224, 224, 3).astype(np.uint8), "side": np.random.rand(224, 224, 3).astype(np.uint8)},
                proprio=np.random.randn(2, 7).astype(np.float32),
                action=np.random.randn(16, 7).astype(np.float32),
            )

    preprocessor = DataPreprocessor(data_cfg)
    dataset = DummyDataset()

    def collate_fn(batch):
        transformed = [preprocessor.transform(b) for b in batch]
        return vla_collator(data_cfg, transformed)

    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs(args.output_dir, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        for batch in loader:
            # Forward pass would go here - simplified for structure
            optimizer.zero_grad()
            # loss = model(**batch)  # implement training forward
            # loss.backward()
            # optimizer.step()
        print(f"Epoch {epoch+1}/{args.epochs}")

    torch.save({"model": model.state_dict()}, os.path.join(args.output_dir, "checkpoint.pt"))
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
