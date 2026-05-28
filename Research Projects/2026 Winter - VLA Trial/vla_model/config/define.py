"""
VLA model configuration definitions.
Based on GraspVLA-main/vla_network/config/define.py
"""
from typing import Optional, List, Type, Dict
from pydantic import BaseModel, Field, ConfigDict
from PIL import Image
import torch
import importlib
from transformers import PreTrainedModel, PreTrainedTokenizerBase, PreTrainedTokenizerFast


class ImageTransform:
    """Interface for image transformation."""

    def __call__(
        self, img: Image.Image, **kwargs: str
    ) -> torch.Tensor: ...


class BasicDataConfig(BaseModel):
    """Basic data configuration."""

    exp_name: Optional[str] = Field(default=None)
    robot: str = "default"
    proprio_len: int = 2
    action_len: int = 16
    action_dim: int = Field(default=7)  # e.g. xyz rpy gripper
    goal_dim: Optional[int] = Field(default=6)
    action_rel_len: int = 1
    dt_steps: int = 1

    def setup(self):
        pass


class VLADataConfig(BasicDataConfig):
    """VLA data configuration."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tokenizer: Optional[PreTrainedTokenizerBase] = Field(init=False, default=None)
    image_transform: Optional[ImageTransform] = Field(init=False, default=None)
    action_token_num: int = 256
    img_steps: int = 1
    img_key: Optional[List[str]] = Field(default_factory=lambda: ["front", "side"])
    image_size: Optional[int] = 224
    anything_prob: float = 0.0
    robot_rep: str = "xyz_rpy"
    goal_rep: Optional[str] = "xyz_rpy"
    tokenizer_type: str = "uniform"
    tokenizer_ratio_limit: float = 0.1
    count_num: int = 100
    trans_noise: float = 0.0
    rot_noise: float = 0.0
    brightness_img: str = "none"
    brightness_threshold: float = 0.0
    crop_mode: Dict[str, str] = Field(default_factory=dict)
    proprio_dim: Optional[int] = Field(default=7)
    use_bbox: int = 0
    pred: Optional[str] = Field(init=False, default=None)

    def setup(self):
        super().setup()
        if self.action_dim is None:
            if self.robot_rep in ["xyz_rpy", "xyz_rpy_rot"]:
                self.action_dim = 7
        if self.goal_dim is None:
            if self.goal_rep == "xyz_rpy":
                self.goal_dim = 6
            elif self.goal_rep == "xyz_rot":
                self.goal_dim = 12
        if self.proprio_dim is None:
            if self.robot_rep == "xyz_rpy":
                self.proprio_dim = 7
            elif self.robot_rep == "xyz_rpy_rot":
                self.proprio_dim = 13

    @property
    def img_num(self) -> int:
        return len(self.img_key or []) * self.img_steps


# Supported VLM backbones
LLM_CONFIG = {
    "meta-llama/Llama-2-7b-hf": {
        "family": "llama2",
        "model_cls": ("transformers", "LlamaForCausalLM"),
        "token_cls": ("transformers", "AutoTokenizer"),
    },
    "internlm/internlm2-1_8b": {
        "family": "internlm",
        "model_cls": ("transformers", "AutoModelForCausalLM"),
        "token_cls": ("transformers", "AutoTokenizer"),
    },
    "Qwen/Qwen2-0.5B": {
        "family": "qwen",
        "model_cls": ("transformers", "AutoModelForCausalLM"),
        "token_cls": ("transformers", "AutoTokenizer"),
    },
    # SmolLM2-360M: https://huggingface.co/HuggingFaceTB/SmolLM2-360M
    "HuggingFaceTB/SmolLM2-360M": {
        "family": "smollm",
        "model_cls": ("transformers", "AutoModelForCausalLM"),
        "token_cls": ("transformers", "AutoTokenizer"),
    },
}


class BasicModelConfig(BaseModel):
    pass


class LLMConfig(BaseModel):
    """Language/VLM backbone configuration."""

    name: str = "HuggingFaceTB/SmolLM2-360M"
    max_len: int = Field(default=2048)
    special_tokens: List[str] = Field(default_factory=list)
    pad_multiple_of: int = Field(default=64)
    attn_implementation: str = "eager"  # eager | sdpa | flash_attention_2

    @property
    def family(self) -> str:
        return LLM_CONFIG.get(self.name, LLM_CONFIG["Qwen/Qwen2-0.5B"])["family"]

    @staticmethod
    def get_cls(package: str, name: str):
        module = importlib.import_module(package)
        return getattr(module, name)

    @property
    def model_cls(self) -> Type[PreTrainedModel]:
        cfg = LLM_CONFIG.get(self.name, LLM_CONFIG["Qwen/Qwen2-0.5B"])
        cls_package, cls_name = cfg["model_cls"]
        return self.get_cls(cls_package, cls_name)

    @property
    def token_cls(self) -> Type[PreTrainedTokenizerFast]:
        cfg = LLM_CONFIG.get(self.name, LLM_CONFIG["Qwen/Qwen2-0.5B"])
        cls_package, cls_name = cfg["token_cls"]
        return self.get_cls(cls_package, cls_name)


class ViTConfig(BaseModel):
    """Vision Transformer (ViT) backbone configuration."""

    name: str = "vit_base_patch16_224"  # timm model name
    image_size: int = 224
    pretrained: bool = True
    # Which layer to take features from (0-indexed, -1 = last)
    feature_layer: int = -2


class ActionExpertConfig(BaseModel):
    """Action expert network configuration (smaller than LLM)."""

    hidden_size_scale: Optional[int] = Field(default=2)
    intermediate_size_scale: Optional[int] = Field(default=2)
    hidden_size: Optional[int] = Field(init=False, default=None)
    intermediate_size: Optional[int] = Field(init=False, default=None)
    hidden_act: Optional[str] = Field(init=False, default="silu")


class FlowMatchingConfig(BaseModel):
    """Flow matching for action generation."""

    beta_alpha: float = 2.0
    beta_beta: float = 2.0
    time_min: float = 0.01
    time_max: float = 1.0


class VLAModelConfig(BasicModelConfig):
    """Full VLA model configuration."""

    backbone_2d: ViTConfig  # ViT config (named for compatibility)
    llm: LLMConfig
    ckpt: Optional[str] = None
    pred: str = "cot_flow_matching"
    action_len: int = Field(init=False, default=16)
    action_dim: int = Field(init=False, default=7)
    proprio_dim: int = Field(init=False, default=7)
    action_expert: bool = True
    action_expert_cfg: Optional[ActionExpertConfig] = Field(default_factory=ActionExpertConfig)
    flow_matching_cfg: Optional[FlowMatchingConfig] = Field(default_factory=FlowMatchingConfig)

    def to_dict(self):
        return self.model_dump()


class BasicConfig(BaseModel):
    data: BasicDataConfig
    model: BasicModelConfig
    dummy: Optional[str] = None


class VLAConfig(BasicConfig):
    data: VLADataConfig
    model: VLAModelConfig
    dummy: Optional[str] = None
