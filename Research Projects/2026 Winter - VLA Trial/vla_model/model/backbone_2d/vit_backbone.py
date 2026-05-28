"""
Vision Transformer (ViT) backbone for processing image data.
Uses timm VisionTransformer; supports single or multi-view images.
"""
import torch
from torch import nn
from torchvision.transforms import Compose, Resize
from PIL import Image

try:
    import timm
    from timm.models.vision_transformer import VisionTransformer
except ImportError:
    timm = None
    VisionTransformer = None

from . import Backbone2D
from vla_model.config import ViTConfig, ImageTransform


class ViT(nn.Module):
    """Wrapper around timm VisionTransformer that returns intermediate layer features."""

    def __init__(self, model: "VisionTransformer", feature_layer: int = -2) -> None:
        super().__init__()
        self.model = model
        self.feature_layer = feature_layer
        n = len(self.model.blocks) + feature_layer if feature_layer < 0 else feature_layer
        self.n = min(max(0, n), len(self.model.blocks) - 1)

    @property
    def embed_dim(self) -> int:
        return self.model.embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.get_intermediate_layers(x, n=[self.n])[0]


class ViTBackbone(Backbone2D):
    """
    ViT backbone: single Vision Transformer for image encoding.
    Output shape: (B, num_patches, embed_dim).
    """

    def __init__(self, config: ViTConfig) -> None:
        super().__init__(config)
        if timm is None:
            raise ImportError("Please install timm: pip install timm")

        name = config.name
        self.vit = ViT(
            timm.create_model(
                name,
                pretrained=config.pretrained,
                num_classes=0,
                img_size=config.image_size,
            ),
            feature_layer=config.feature_layer,
        )
        self.vit.eval()

        # Build image transform from timm config
        model_cfg = timm.data.resolve_model_data_config(self.vit.model)
        model_cfg["input_size"] = (3, config.image_size, config.image_size)
        transform = timm.data.create_transform(**model_cfg, is_training=False)
        target_size = (config.image_size, config.image_size)
        interp = getattr(transform.transforms[0], "interpolation", 2)  # BILINEAR
        self.image_transform = Compose([
            Resize(target_size, interpolation=interp),
            *transform.transforms[1:],
        ])

    @property
    def feature_dim(self) -> int:
        return self.vit.embed_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: (B, T_img, C, H, W) or (B, C, H, W)
        Returns: (B, num_patches, feature_dim)
        """
        if images.dim() == 5:
            b, t, c, h, w = images.shape
            x = images.reshape(b * t, c, h, w)
            feat = self.vit(x)
            feat = feat.reshape(b, t, -1, feat.shape[-1])
            # Merge time and patch dims: (B, T*num_patches, D)
            feat = feat.reshape(b, -1, feat.shape[-1])
        else:
            feat = self.vit(images)
        return feat
