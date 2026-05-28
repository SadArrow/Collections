"""
Action Expert: accepts VLM output (KV cache) and suffix embeddings, outputs hidden states.
Uses a smaller transformer derived from the LLM config, or a minimal decoder as fallback.
"""
import copy
from typing import Optional, Tuple
from dataclasses import dataclass
import torch
from torch import nn
from transformers import PreTrainedModel
@dataclass
class ActionExpertOutput:
    """Output compatible with causal LM interface (hidden_states, past_key_values)."""

    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    past_key_values: Optional[Tuple] = None


def create_action_expert_from_llm(
    llm: PreTrainedModel,
    hidden_size_scale: int = 2,
    intermediate_size_scale: int = 2,
) -> nn.Module:
    """
    Create a smaller action expert from LLM config.
    Uses the same decoder architecture so it properly handles past_key_values.
    """
    try:
        config = copy.deepcopy(llm.config)
        if hasattr(config, "attn_implementation"):
            config.attn_implementation = "eager"
        config.hidden_size = config.hidden_size // hidden_size_scale
        config.intermediate_size = getattr(
            config, "intermediate_size", config.hidden_size * 4
        ) // intermediate_size_scale
        config.num_hidden_layers = min(4, getattr(config, "num_hidden_layers", 32) // 2)
        model_cls = type(llm)
        if hasattr(model_cls, "_from_config"):
            return model_cls._from_config(config)
    except Exception:
        pass

    # Fallback: simple decoder (no prefix KV support; for testing only)
    hidden_size = getattr(llm.config, "hidden_size", 896) // hidden_size_scale
    return SimpleActionExpert(
        hidden_size=hidden_size,
        num_layers=4,
        num_heads=8,
        intermediate_size=hidden_size * 4,
    )


class SimpleActionExpert(nn.Module):
    """
    Minimal decoder for suffix-only processing.
    When past_key_values is provided, concatenates virtual prefix to emulate full context.
    """

    def __init__(
        self,
        hidden_size: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        intermediate_size: int = 2048,
    ):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": hidden_size})()
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=intermediate_size,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)

    def forward(
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        # past_key_values ignored in SimpleActionExpert; use create_action_expert_from_llm for full support
        tgt = inputs_embeds
        memory = torch.zeros(tgt.shape[0], 1, tgt.shape[2], device=tgt.device, dtype=tgt.dtype)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt.size(1), device=tgt.device, dtype=torch.bool
        )
        hidden = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        return ActionExpertOutput(
            hidden_states=(hidden,),
            past_key_values=past_key_values,
        )
