"""
VLA: Vision-Language-Action model.
Pipeline: ViT -> Projector -> VLM (+ tokens) -> Action Expert (flow matching) -> Action chunk.
"""
from typing import Optional, List, Tuple, Any
import os
import json
import copy
import torch
from torch import nn
from transformers import PreTrainedTokenizerBase

from vla_model.type import RawVLAData
from vla_model.model.backbone_2d import Backbone2D
from vla_model.model.backbone_llm import LLMBackbone
from vla_model.config import (
    VLAModelConfig,
    VLADataConfig,
    ImageTransform,
    ActionExpertConfig,
)
from vla_model.model.vla.projector import FusedMLPProjector
from vla_model.model.vla.flow_matching import VLAFlowMatchingModule
from vla_model.model.vla.action_expert import create_action_expert_from_llm
from vla_model.utils.constant import IGNORE_INDEX

try:
    from safetensors.torch import load_file as load_safetensors
except ImportError:
    load_safetensors = None


def update_state_dict(state_dict: dict) -> dict:
    if "llm_backbone" in state_dict:
        state_dict["llm"] = state_dict.pop("llm_backbone")
    if "vision_backbone" in state_dict:
        state_dict["backbone_2d"] = {
            k.replace("_featurizer", ".model"): v
            for k, v in state_dict.pop("vision_backbone").items()
        }
    return state_dict


def make_block_attn_mask(input_mask: torch.Tensor, block_mask: torch.Tensor) -> torch.Tensor:
    cumsum = torch.cumsum(block_mask.long(), dim=0)
    causal_num = (cumsum == 0).sum().item()
    n = input_mask.shape[1]
    device = input_mask.device
    causal_mask = torch.tril(torch.ones((n, n), dtype=torch.bool, device=device))
    if causal_num != len(block_mask):
        block_attn_mask = cumsum[None, causal_num:] <= cumsum[causal_num:, None]
        causal_mask[causal_num:, causal_num:] = block_attn_mask
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return torch.logical_and(causal_mask, valid_mask)[:, None]


def load_model(model: nn.Module, ckpt_path: str) -> nn.Module:
    if ckpt_path.endswith(".safetensors") and load_safetensors:
        ckpt = load_safetensors(ckpt_path)
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        ckpt = ckpt.get("model", ckpt)
    ckpt = update_state_dict(ckpt)
    model.load_state_dict(ckpt, strict=False)
    return model


class VLA(nn.Module):
    """VLA: ViT + VLM + Action Expert (flow matching)."""

    def __init__(self, config: VLAModelConfig):
        super().__init__()
        self.config = config

    def init(self, train: bool = False):
        from vla_model.config import ViTConfig
        vit_config = self.config.backbone_2d
        if isinstance(vit_config, dict):
            vit_config = ViTConfig(**vit_config)
        elif hasattr(vit_config, "model_dump"):
            vit_config = ViTConfig(**vit_config.model_dump())
        self.backbone_2d = Backbone2D.init(vit_config)
        self.backbone_2d_dim = self.backbone_2d.feature_dim
        self.image_transform = getattr(self.backbone_2d, "image_transform", None)
        self.is_train = train

        self.llm = LLMBackbone(self.config.llm, train=train)
        self.llm_dim = self.llm.input_dim
        self.tokenizer = self.llm.tokenizer

        cfg = self.config
        cfg.action_len = getattr(cfg, "action_len", None) or 16
        cfg.action_dim = getattr(cfg, "action_dim", None) or 7
        cfg.proprio_dim = getattr(cfg, "proprio_dim", None) or 7

        if cfg.action_expert and cfg.action_expert_cfg:
            ae_cfg = cfg.action_expert_cfg
            self.action_expert = create_action_expert_from_llm(
                self.llm.llm,
                hidden_size_scale=ae_cfg.hidden_size_scale or 2,
                intermediate_size_scale=ae_cfg.intermediate_size_scale or 2,
            )
        else:
            self.action_expert = None

        torch.manual_seed(self.backbone_2d_dim)
        self.projector = FusedMLPProjector(self.backbone_2d_dim, self.llm_dim)

        if cfg.pred == "cot_flow_matching" and cfg.flow_matching_cfg:
            ae_hidden = getattr(self.action_expert.config, "hidden_size", 512) if self.action_expert else 512
            self.flow_module = VLAFlowMatchingModule(
                config=cfg.flow_matching_cfg,
                action_dim=cfg.action_dim,
                llm_dim=ae_hidden,
                action_len=cfg.action_len,
                proprio_dim=cfg.proprio_dim,
            )

    def from_pretrained(self, path: Optional[str] = None) -> "VLA":
        path = path or self.config.ckpt
        if path and os.path.exists(path):
            load_model(self, path)
        return self

    @staticmethod
    def insert_img_info(orig: torch.Tensor, img_info: torch.Tensor) -> torch.Tensor:
        return torch.cat([orig[:, :1], img_info, orig[:, 1:]], dim=1)

    @staticmethod
    def insert_img_info_single(orig: torch.Tensor, img_info: torch.Tensor) -> torch.Tensor:
        return torch.cat([orig[:1], img_info, orig[1:]], dim=0)

    def get_proj_feat_2d(self, images: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(self.is_train):
            feat_2d = self.backbone_2d(images)
        return self.projector(feat_2d)

    def embed_prefix(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        proj_feat_2d: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        b = len(input_ids)
        if proj_feat_2d is None:
            proj_feat_2d = self.get_proj_feat_2d(images)
        n_img_token = proj_feat_2d.shape[1]

        input_embed = self.llm.input_embedding(input_ids)
        mm_input_embed = self.insert_img_info(input_embed, proj_feat_2d).to(input_embed.dtype)

        img_attn_mask = torch.ones((b, n_img_token), dtype=torch.bool, device=attention_mask.device)
        mm_attn_mask = self.insert_img_info(attention_mask, img_attn_mask)

        n_mm_token = mm_attn_mask.shape[1]
        mm_block_mask = torch.zeros((n_mm_token,), dtype=torch.bool, device=attention_mask.device)

        mm_labels = None
        if labels is not None:
            img_labels = torch.full(
                (b, n_img_token), IGNORE_INDEX, dtype=labels.dtype, device=labels.device
            )
            mm_labels = self.insert_img_info(labels, img_labels)

        return mm_input_embed, mm_attn_mask, mm_block_mask, mm_labels

    def generate(
        self,
        input_ids: torch.LongTensor,
        robot_input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        robot_attention_mask: torch.Tensor,
        images: torch.Tensor,
        proprio: torch.Tensor,
        max_token_num: int = 100,
        flow_matching_iter: int = 10,
        inference_kwargs: Optional[List[dict]] = None,
        token_pattern=None,
    ) -> Tuple[Any, Any]:
        proj_feat_2d = self.get_proj_feat_2d(images)
        prefix_embeds, prefix_mask, prefix_block_mask, _ = self.embed_prefix(
            input_ids=input_ids,
            attention_mask=attention_mask,
            proj_feat_2d=proj_feat_2d,
            labels=None,
        )

        if self.config.pred == "cot_flow_matching":
            cot_parse, kv_cache = self._generate_autoregressive(
                input_ids=input_ids,
                robot_input_ids=robot_input_ids,
                proj_feat_2d=proj_feat_2d,
                attention_mask=attention_mask,
                robot_attention_mask=robot_attention_mask,
                max_token_num=max_token_num,
                token_pattern=token_pattern,
                inference_kwargs=inference_kwargs or [{}],
                require_kv_cache=True,
            )

            input_ids_full = torch.tensor(cot_parse.input_ids, device=input_ids.device)[None]
            _, prefix_mask, prefix_block_mask, _ = self.embed_prefix(
                input_ids=input_ids_full,
                attention_mask=torch.ones_like(input_ids_full).bool(),
                proj_feat_2d=proj_feat_2d,
                labels=None,
            )

            padded_prefix_length = kv_cache[0][0].shape[2]
            num_paddings = padded_prefix_length - prefix_mask.shape[1]
            if num_paddings > 0:
                pad_mask = torch.zeros(
                    (prefix_mask.shape[0], num_paddings),
                    dtype=prefix_mask.dtype,
                    device=prefix_mask.device,
                )
                prefix_mask = torch.cat([pad_mask, prefix_mask], dim=1)
                pad_block = torch.zeros(
                    (num_paddings,), dtype=prefix_block_mask.dtype, device=prefix_block_mask.device
                )
                prefix_block_mask = torch.cat([pad_block, prefix_block_mask], dim=0)

            action = self._generate_flow_matching(
                prefix_kv_cache=tuple(kv_cache),
                prefix_mask=prefix_mask,
                prefix_block_mask=prefix_block_mask,
                proprio=proprio,
                flow_matching_iter=flow_matching_iter,
            )
            return cot_parse, action
        raise NotImplementedError(f"pred={self.config.pred}")

    def _generate_flow_matching(
        self,
        prefix_kv_cache,
        prefix_mask: torch.Tensor,
        prefix_block_mask: torch.Tensor,
        proprio: torch.Tensor,
        flow_matching_iter: int,
    ) -> torch.Tensor:
        device = prefix_kv_cache[0][0].device
        dtype = prefix_kv_cache[0][0].dtype
        proprio = proprio.to(dtype)

        noise = self.flow_module.sample_noise(
            batch_size=len(proprio), device=device, dtype=dtype
        )
        proprio_embeds = self.flow_module.proprior_proj(proprio)
        suffix_mask, suffix_block_mask = self.flow_module.get_suffix_masks(proprio_embeds)

        full_input_mask = torch.cat((prefix_mask, suffix_mask), dim=1)
        full_block_mask = torch.cat((prefix_block_mask, suffix_block_mask), dim=0)
        full_attn_mask = make_block_attn_mask(full_input_mask, full_block_mask).to(dtype)
        full_position_ids = torch.cumsum(full_input_mask.long(), dim=1) - 1
        suffix_attn_mask = full_attn_mask[:, :, -suffix_mask.shape[1] :, :]
        suffix_position_ids = full_position_ids[:, -suffix_mask.shape[1] :]

        def compute_v_t(x_t: torch.Tensor, time_vec: torch.Tensor):
            suffix_embeds = self.flow_module.embed_suffix_flow_matching_embeds(
                proprio_embeds, x_t, time_vec
            )
            out = self.action_expert(
                attention_mask=suffix_attn_mask,
                position_ids=suffix_position_ids,
                inputs_embeds=suffix_embeds,
                past_key_values=prefix_kv_cache,
                use_cache=True,
                output_hidden_states=True,
            )
            action_hidden = out.hidden_states[-1][:, -self.config.action_len :]
            return self.flow_module.get_v_t(action_hidden)

        return self.flow_module.denoise(compute_v_t, noise, flow_matching_iter)

    def _generate_autoregressive(
        self,
        input_ids,
        robot_input_ids,
        proj_feat_2d,
        attention_mask,
        robot_attention_mask,
        max_token_num,
        token_pattern,
        inference_kwargs,
        require_kv_cache=False,
    ):
        assert input_ids.shape[0] == 1
        cache = None
        current_input_embeddings = []
        output = []

        for idx, token_info in enumerate(
            [*(token_pattern.infos if token_pattern else []), *(token_pattern.robot_infos if token_pattern else [])]
        ):
            if token_info is None:
                continue
            if token_info.as_input:
                emb = self.llm.input_embedding(
                    torch.tensor(
                        inference_kwargs[0][token_info.key],
                        device=input_ids.device,
                    )
                )
                if idx == 0:
                    emb = self.insert_img_info_single(emb, proj_feat_2d[0])
                current_input_embeddings.append(emb)
                continue

            generated, cache = self.llm.generate(
                max_token_num=token_info.length,
                inputs_embeds=torch.cat(current_input_embeddings, dim=0).unsqueeze(0),
                cache=cache,
            )
            output.extend(generated[0])
            current_input_embeddings = [
                self.llm.input_embedding(
                    torch.tensor([generated[0][-1]], dtype=torch.long, device=input_ids.device)
                )
            ]
            parse_ret = token_pattern.update_tokens(output, **inference_kwargs[0])
            if parse_ret.terminate or len(output) >= max_token_num:
                break

        kv_cache = None
        if require_kv_cache and current_input_embeddings:
            _, cache_with_kv = self.llm.generate(
                max_token_num=1,
                inputs_embeds=torch.cat(current_input_embeddings, dim=0).unsqueeze(0),
                cache=cache,
            )
            kv_cache = cache_with_kv["past_key_values"]
        return parse_ret, kv_cache


# Data preprocessor and collator will be imported from data module
def get_preprocessor_and_collator():
    from vla_model.data.preprocess import DataPreprocessor
    from vla_model.data.vla_data_collator import vla_collator
    return DataPreprocessor, vla_collator


class VLAAgent:
    """High-level agent for inference."""

    def __init__(
        self,
        path: Optional[str] = None,
        device: str = "cuda:0",
        compile_model: bool = False,
    ):
        self.path = path
        self.device = device
        self.model_cfg, self.data_cfg, self.model, self.preprocessor = self._load_vla(
            path, device, compile_model
        )
        self.token_pattern = getattr(self.preprocessor, "pattern", None)

    def _load_vla(self, path, device, compile_model):
        from vla_model.data.preprocess import DataPreprocessor
        from vla_model.data.vla_data_collator import vla_collator

        base = os.path.dirname(os.path.abspath(path or "."))
        cfg_path = os.path.join(base, "config.json")
        if not os.path.exists(cfg_path):
            cfg_path = os.path.join(base, "..", "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            data_cfg = VLADataConfig.model_validate(cfg.get("data", {}))
            model_cfg = VLAModelConfig.model_validate(cfg.get("model", {}))
        else:
            from vla_model.config import ViTConfig, LLMConfig
            data_cfg = VLADataConfig()
            model_cfg = VLAModelConfig(
                backbone_2d=ViTConfig(),
                llm=LLMConfig(),
                pred="cot_flow_matching",
                action_expert=True,
            )
        data_cfg.setup()

        model = VLA(model_cfg)
        model.init(train=False)
        if path and os.path.exists(path):
            load_model(model, path)
        model = model.to(device).eval()

        data_cfg.tokenizer = model.tokenizer
        data_cfg.image_size = getattr(model.config.backbone_2d, "image_size", 224)
        data_cfg.image_transform = getattr(model, "image_transform", None)
        data_cfg.pred = model_cfg.pred

        preprocessor = DataPreprocessor(data_cfg)
        preprocessor_path = os.path.join(base, "preprocessor.npz")
        if not os.path.exists(preprocessor_path):
            preprocessor_path = os.path.join(base, "..", "preprocessor.npz")
        if os.path.exists(preprocessor_path):
            import numpy as np
            preprocessor.load(np.load(preprocessor_path))

        return model_cfg, data_cfg, model, preprocessor

    def sample_action(self, raw: RawVLAData) -> dict:
        from vla_model.data.vla_data_collator import vla_collator
        with torch.no_grad():
            with torch.autocast(device_type="cuda" if "cuda" in self.device else "cpu", dtype=torch.bfloat16):
                x = self.preprocessor.transform(raw, inference=True)
                model_input = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in vla_collator(self.data_cfg, [x]).items()
                }
                token_result, action_result = self.model.generate(
                    input_ids=model_input["input_ids"],
                    robot_input_ids=model_input["robot_input_ids"],
                    attention_mask=model_input["attention_mask"],
                    robot_attention_mask=model_input["robot_attention_mask"],
                    images=model_input["images"],
                    proprio=model_input["proprio"],
                    inference_kwargs=x.inference_kwargs,
                    token_pattern=self.token_pattern,
                    max_token_num=100,
                )
        ret = {"action": None}
        if hasattr(self.preprocessor, "robot_tokenizer"):
            ret["action"] = self.preprocessor.robot_tokenizer.inv_norm_action(
                action_result.float().cpu().numpy()[0]
            )
        else:
            ret["action"] = action_result.float().cpu().numpy()[0]
        return ret
