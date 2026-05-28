"""Token pattern for CoT action generation."""
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union
import torch
import numpy as np
from pydantic import BaseModel, Field

from vla_model.utils.constant import IGNORE_INDEX
from vla_model.config import VLADataConfig


def to_flatten_list(x: Union[torch.Tensor, np.ndarray, list]) -> list:
    if hasattr(torch, "Tensor") and isinstance(x, torch.Tensor):
        return x.reshape(-1).tolist()
    if isinstance(x, np.ndarray):
        return x.reshape(-1).tolist()
    if isinstance(x, list):
        return np.array(x).reshape(-1).tolist()
    raise ValueError(f"Unsupported type {type(x)}")


class TokenInfo(BaseModel):
    key: str
    length: Optional[int] = None
    est: bool = False
    as_input: bool = False
    terminate: Callable = Field(
        default=lambda tinfo, tokens: len(tokens) == tinfo.length if tinfo.length else False
    )


@dataclass
class TokenResult:
    terminate: bool = False
    input_ids: List[int] = None
    robot_input_ids: List[int] = None

    def __post_init__(self):
        if self.input_ids is None:
            self.input_ids = []
        if self.robot_input_ids is None:
            self.robot_input_ids = []


class TokenPattern(BaseModel):
    infos: List[Optional[TokenInfo]]
    robot_infos: List[Optional[TokenInfo]]

    def update_tokens(self, output: List[int], **kwargs) -> TokenResult:
        output = deepcopy(to_flatten_list(output))
        ret = TokenResult(terminate=False)
        for ids, infos in [(ret.input_ids, self.infos), (ret.robot_input_ids, self.robot_infos)]:
            for info in infos:
                if info is None:
                    continue
                if info.as_input:
                    if info.key in kwargs:
                        value = to_flatten_list(kwargs[info.key])
                        ids.extend(value)
                    setattr(ret, info.key, kwargs.get(info.key))
                else:
                    cur = []
                    while True:
                        if info.terminate(info, cur):
                            setattr(ret, info.key, cur)
                            break
                        elif len(output) == 0:
                            return ret
                        else:
                            token_id = output.pop(0)
                            ids.append(token_id)
                            cur.append(token_id)
        ret.terminate = True
        return ret


def get_cot_action_pattern(config: VLADataConfig) -> TokenPattern:
    return TokenPattern(
        infos=[
            TokenInfo(key="text_ids", length=None, est=False, as_input=True),
            TokenInfo(key="hist_proprio", length=(config.proprio_len - 1) * config.proprio_dim, est=False, as_input=True),
            TokenInfo(key="cur_proprio", length=config.proprio_dim, est=False, as_input=True),
            TokenInfo(key="goal", length=config.goal_dim, est=True, as_input=False) if config.goal_dim else None,
            TokenInfo(key="eos", length=1, est=True, as_input=False),
        ],
        robot_infos=[],
    )


def get_token_pattern(config: VLADataConfig, name: str) -> TokenPattern:
    return {"cot_action": get_cot_action_pattern}[name](config)
