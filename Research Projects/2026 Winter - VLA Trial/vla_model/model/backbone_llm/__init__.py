"""
VLM (Vision-Language Model) backbone: pre-trained LLM for processing ViT features + tokens.
"""
from typing import List, Optional
import copy
import torch
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.modeling_outputs import CausalLMOutputWithPast

from vla_model.config import LLMConfig


PAD_TOKEN = "<|endoftext|>"


class LLMBackbone(nn.Module):
    """Wrapper around HuggingFace causal LM for VLM processing."""

    def __init__(self, config: LLMConfig, train: bool = False) -> None:
        super().__init__()
        config = copy.deepcopy(config)
        self.config = config

        attn_impl = config.attn_implementation
        if attn_impl == "flex_attention":
            try:
                import torch._dynamo
                attn_impl = "sdpa"  # fallback if flex not available
            except Exception:
                attn_impl = "sdpa"

        self.llm = config.model_cls.from_pretrained(
            config.name,
            attn_implementation=attn_impl if attn_impl != "eager" else None,
            torch_dtype=torch.bfloat16 if not train else torch.float32,
            trust_remote_code=True,
        )
        self.llm.config.use_cache = True

        self.tokenizer = config.token_cls.from_pretrained(
            config.name,
            model_max_length=config.max_len,
            padding_side="right",
            trust_remote_code=True,
        )
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": config.special_tokens}
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
        self.llm.config.pad_token_id = self.tokenizer.pad_token_id
        self.llm.resize_token_embeddings(
            len(self.tokenizer), pad_to_multiple_of=config.pad_multiple_of
        )

    @property
    def input_dim(self) -> int:
        return self.llm.get_input_embeddings().embedding_dim

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> CausalLMOutputWithPast:
        return self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def generate(
        self,
        max_token_num: int,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache: Optional[dict] = None,
    ):
        """Autoregressive generation. Returns (generated_tokens, cache)."""
        assert inputs_embeds is not None and inputs_embeds.shape[0] == 1
        if cache is None:
            cache = {}
        device = inputs_embeds.device
        past_key_values = cache.get("past_key_values")

        full_length = inputs_embeds.shape[1] + max_token_num
        if past_key_values is not None:
            full_length += past_key_values[0][0].shape[2]

        position_ids = torch.arange(full_length, device=device).unsqueeze(0)
        attention_mask = torch.tril(
            torch.ones((full_length, full_length), device=device)
        ).unsqueeze(0).unsqueeze(0)

        PAD_TO = 16
        num_padding = cache.get("num_padding", ((PAD_TO - full_length % PAD_TO) % PAD_TO))
        if num_padding is None:
            num_padding = ((PAD_TO - full_length % PAD_TO) % PAD_TO)
        if num_padding > 0 and past_key_values is None:
            pad_embeds = torch.zeros(
                (inputs_embeds.shape[0], num_padding, inputs_embeds.shape[2]),
                dtype=inputs_embeds.dtype, device=device
            )
            inputs_embeds = torch.cat([pad_embeds, inputs_embeds], dim=1)

        generated_tokens = []
        for _ in range(max_token_num):
            past_len = past_key_values[0][0].shape[2] if past_key_values is not None else 0
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask[:, :, past_len : past_len + inputs_embeds.shape[1], : past_len + inputs_embeds.shape[1]],
                position_ids=position_ids[:, past_len : past_len + inputs_embeds.shape[1]],
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            next_token = outputs.logits[:, -1, :].argmax(dim=-1)
            generated_tokens.append(next_token.item())
            past_key_values = outputs.past_key_values
            inputs_embeds = self.llm.get_input_embeddings()(next_token.unsqueeze(-1))

        return [generated_tokens], {**cache, "past_key_values": past_key_values, "num_padding": num_padding}

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    @property
    def input_embedding(self) -> nn.Module:
        return self.llm.get_input_embeddings()
