"""
VLA: Vision-Language-Action model for robot control.
Pipeline: ViT (images) -> Projector -> VLM (+ tokens) -> Action Expert (flow matching) -> Action chunk.
"""
from .model.vla import VLA, VLAAgent
from .config import VLAModelConfig, VLADataConfig

__all__ = ["VLA", "VLAAgent", "VLAModelConfig", "VLADataConfig"]
