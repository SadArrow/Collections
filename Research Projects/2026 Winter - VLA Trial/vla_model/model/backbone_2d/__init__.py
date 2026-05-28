"""
2D Vision backbone: ViT (Vision Transformer) for image encoding.
"""
from abc import ABC, abstractmethod
import torch
from torch import nn

from vla_model.config import ViTConfig, ImageTransform


class Backbone2D(nn.Module, ABC):
    """Abstract 2D vision backbone."""

    config: ViTConfig
    image_transform: ImageTransform

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.config = config

    @property
    @abstractmethod
    def feature_dim(self) -> int: ...

    @abstractmethod
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def init(config: ViTConfig) -> "Backbone2D":
        from .vit_backbone import ViTBackbone
        return ViTBackbone(config)
