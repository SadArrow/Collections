"""Data types for VLA pipeline."""
from dataclasses import dataclass
from typing import Optional, List, Any, Dict
from pydantic import BaseModel, ConfigDict, BeforeValidator, PlainSerializer
from typing_extensions import Annotated
import numpy as np
import torch


def nd_array_custom_before_validator(x):
    return np.array(x)


def nd_array_custom_serializer(x):
    return x.tolist()


NdArray = Annotated[
    np.ndarray,
    BeforeValidator(nd_array_custom_before_validator),
    PlainSerializer(nd_array_custom_serializer, return_type=list, when_used="json"),
]


class RawVLAData(BaseModel):
    """Raw sample: instruction, images, proprio, action, goal."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    instruction: Optional[str] = None
    can_be_anything: bool = False
    images: Optional[Dict[str, NdArray]] = None
    bboxs: Optional[Dict[str, NdArray]] = None
    pcs: Optional[Dict[str, NdArray]] = None
    proprio: NdArray = None
    proprio_flag: Optional[NdArray] = None
    for_rel_proprio: Optional[NdArray] = None
    for_rel_proprio_flag: Optional[NdArray] = None
    action: Optional[NdArray] = None
    action_flag: Optional[NdArray] = None
    goal: Optional[NdArray] = None
    goal_trans: Optional[NdArray] = None
    goal_rot: Optional[NdArray] = None


@dataclass
class BatchVLAData:
    """Batched VLA data for training/inference."""

    debug: List[Any]
    input_ids: torch.Tensor
    robot_input_ids: torch.Tensor
    labels: Optional[torch.Tensor]
    robot_labels: Optional[torch.Tensor]
    attention_mask: torch.Tensor
    robot_attention_mask: torch.Tensor
    action: Optional[torch.Tensor]
    proprio: torch.Tensor
    goal: Optional[torch.Tensor]
    images: torch.Tensor
    is_action: torch.Tensor
    inference_kwargs: Optional[list] = None
