"""Batch collator for VLA data."""
from typing import List
import torch
from torch.nn.utils.rnn import pad_sequence

from vla_model.type import BatchVLAData
from vla_model.config import VLADataConfig


def vla_collator(config: VLADataConfig, datas: List[BatchVLAData]) -> dict:
    pad_idx = config.tokenizer.pad_token_id if config.tokenizer else 0
    max_len = getattr(config.tokenizer, "model_max_length", 2048) if config.tokenizer else 2048

    input_ids = pad_sequence(
        [d.input_ids[0] for d in datas],
        batch_first=True,
        padding_value=pad_idx,
    )
    robot_input_ids = pad_sequence(
        [d.robot_input_ids[0] for d in datas],
        batch_first=True,
        padding_value=pad_idx,
    )

    kwargs = {
        "input_ids": input_ids[:, :max_len],
        "robot_input_ids": robot_input_ids,
        "attention_mask": input_ids[:, :max_len] != pad_idx,
        "robot_attention_mask": robot_input_ids != pad_idx,
    }

    for k in ["images", "action", "proprio", "goal", "is_action"]:
        if getattr(datas[0], k, None) is not None:
            kwargs[k] = torch.cat([getattr(d, k) for d in datas], dim=0)
        else:
            kwargs[k] = None

    return kwargs
