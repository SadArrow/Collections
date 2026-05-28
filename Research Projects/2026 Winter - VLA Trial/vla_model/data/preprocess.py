"""Data preprocessing for VLA."""
from typing import Dict, List, Optional, Tuple
import random
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from vla_model.type import BatchVLAData, RawVLAData
from vla_model.config import VLADataConfig, ImageTransform
from .tokenizer import RobotTokenizer
from .token_pattern import get_token_pattern


def resize_with_bbox(
    image: Image.Image,
    bbox: Optional[np.ndarray],
    target_size: Tuple[int, int],
    random_padding: bool = False,
) -> Tuple[Image.Image, Optional[np.ndarray]]:
    orig_size = image.size
    ratio = min(target_size[0] / orig_size[0], target_size[1] / orig_size[1])
    new_size = (int(orig_size[0] * ratio), int(orig_size[1] * ratio))
    image = image.resize(new_size, Image.LANCZOS)
    new_image = Image.new("RGB", target_size)
    paste_x = random.randint(0, target_size[0] - new_size[0]) if random_padding else (target_size[0] - new_size[0]) // 2
    paste_y = random.randint(0, target_size[1] - new_size[1]) if random_padding else (target_size[1] - new_size[1]) // 2
    new_image.paste(image, (paste_x, paste_y))
    if bbox is not None:
        new_bbox = bbox * ratio
        new_bbox[0] += paste_x
        new_bbox[1] += paste_y
        new_bbox[2] += paste_x
        new_bbox[3] += paste_y
    else:
        new_bbox = None
    return new_image, new_bbox


class DataPreprocessor:
    def __init__(self, config: VLADataConfig):
        self.config = config
        self.tokenizer = config.tokenizer
        config.tokenizer = None
        self.robot_tokenizer = RobotTokenizer.init(config, self.tokenizer.vocab_size if self.tokenizer else 0)
        config.tokenizer = self.tokenizer
        self.image_transform = config.image_transform
        if config.pred == "cot_flow_matching":
            self.pattern = get_token_pattern(config, "cot_action")

    def load(self, data: dict):
        self.robot_tokenizer.load(data)

    def transform_img_bbox(
        self,
        raw_images: Dict[str, np.ndarray],
        raw_bboxs: Optional[Dict[str, np.ndarray]],
    ) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
        pixel_values = []
        bboxs = []
        img_key = self.config.img_key or ["front", "side"]
        for i in range(self.config.img_steps):
            for img_k in img_key:
                if img_k not in raw_images:
                    continue
                img_arr = raw_images[img_k]
                if img_arr.ndim == 3:
                    img_arr = img_arr[None, ...]
                img, bbox = resize_with_bbox(
                    Image.fromarray(img_arr[i] if img_arr.shape[0] > i else img_arr[0]),
                    raw_bboxs[img_k][i] if raw_bboxs and img_k in raw_bboxs else None,
                    (self.config.image_size, self.config.image_size),
                )
                if self.image_transform:
                    pv = self.image_transform(img)
                else:
                    default_tf = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ])
                    pv = default_tf(img)
                if isinstance(pv, np.ndarray):
                    pv = torch.from_numpy(pv).float()
                pixel_values.append(pv)
                bboxs.append(bbox)
        pixel_values = torch.stack(pixel_values)[None]
        bboxs = np.stack(bboxs) if bboxs and bboxs[0] is not None else None
        return pixel_values, bboxs

    def transform(self, raw_data: RawVLAData, inference: bool = False) -> BatchVLAData:
        pixel_values, bboxs = self.transform_img_bbox(raw_data.images or {}, raw_data.bboxs)
        trans_dic = dict(proprio=raw_data.proprio, action=raw_data.action, goal=None)

        text_ids = self.tokenizer(raw_data.instruction or "", add_special_tokens=True).input_ids

        proprio = trans_dic["proprio"]
        if len(proprio) < self.config.proprio_len:
            proprio = np.concatenate([proprio, np.tile(proprio[-1], (self.config.proprio_len - len(proprio), 1))])

        inference_kwargs = [
            dict(
                text_ids=text_ids,
                hist_proprio=self.robot_tokenizer.proprio(proprio[:-1].flatten()),
                cur_proprio=self.robot_tokenizer.proprio(proprio[-1]),
            )
        ]
        token_result = self.pattern.update_tokens(output=[], **inference_kwargs[0])
        input_ids = token_result.input_ids
        robot_input_ids = token_result.robot_input_ids

        normed = self.robot_tokenizer.norm_proprio(proprio) if hasattr(self.robot_tokenizer, "min_proprio") and getattr(self.robot_tokenizer, "min_proprio", None) is not None else proprio
        proprio_tensor = torch.from_numpy(np.asarray(normed, dtype=np.float32)).float()[None]

        return BatchVLAData(
            debug=[None],
            input_ids=torch.tensor(input_ids)[None],
            labels=None,
            attention_mask=torch.ones(len(input_ids))[None].bool(),
            robot_input_ids=torch.tensor(robot_input_ids)[None],
            robot_attention_mask=torch.ones(len(robot_input_ids))[None].bool(),
            robot_labels=None,
            images=pixel_values,
            action=None,
            proprio=proprio_tensor,
            goal=None,
            is_action=torch.ones(1).bool(),
            inference_kwargs=inference_kwargs,
        )
