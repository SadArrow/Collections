"""Robot tokenizer for proprio, action, goal."""
from copy import deepcopy
from typing import Callable, Dict, List
import numpy as np

from vla_model.config import VLADataConfig

_robot_tokenizer = None


class RobotTokenizer:
    def __init__(self, config: VLADataConfig, vocab_size: int):
        self.config = config
        self.vocab_size = vocab_size

    @staticmethod
    def init(config: VLADataConfig, vocab_size: int) -> "RobotTokenizer":
        global _robot_tokenizer
        if _robot_tokenizer is None:
            config = deepcopy(config)
            if config.tokenizer_type == "uniform":
                _robot_tokenizer = UniformRobotTokenizer(config, vocab_size)
            elif config.tokenizer_type == "ratio_min_max_uniform":
                _robot_tokenizer = RatioMinMaxUniformRobotTokenizer(config, vocab_size)
            else:
                _robot_tokenizer = RatioMinMaxUniformRobotTokenizer(config, vocab_size)
        return _robot_tokenizer

    def bbox(self, bbox: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def proprio(self, proprio: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def action(self, action: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def inv_action(self, action: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def goal(self, goal: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def inv_goal(self, goal: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def save(self) -> dict:
        return {}

    def load(self, data: dict):
        pass


class UniformRobotTokenizer(RobotTokenizer):
    def __init__(self, config: VLADataConfig, vocab_size: int):
        super().__init__(config, vocab_size)
        self.bins = np.linspace(-1.0, 1.0, config.action_token_num)

    def uniform_tokenize(self, x: np.ndarray) -> np.ndarray:
        x = x.flatten()
        d = np.clip(np.digitize(x, self.bins), a_min=1, a_max=self.config.action_token_num)
        return self.vocab_size - d

    def uniform_detokenize(self, x: np.ndarray) -> np.ndarray:
        y = self.vocab_size - np.array(x)
        return (
            self.bins[np.clip(y - 1, 0, self.config.action_token_num - 1)]
            + self.bins[np.clip(y, 0, self.config.action_token_num - 1)]
        ) / 2

    def bbox(self, bbox: np.ndarray) -> np.ndarray:
        return self.uniform_tokenize(bbox)

    def proprio(self, proprio: np.ndarray) -> np.ndarray:
        return self.uniform_tokenize(proprio)

    def action(self, action: np.ndarray) -> np.ndarray:
        return self.uniform_tokenize(action)

    def goal(self, goal: np.ndarray) -> np.ndarray:
        return self.uniform_tokenize(goal)

    def inv_action(self, action: np.ndarray) -> np.ndarray:
        return self.uniform_detokenize(action)

    def inv_goal(self, goal: np.ndarray) -> np.ndarray:
        return self.uniform_detokenize(goal)

    def norm_proprio(self, proprio: np.ndarray) -> np.ndarray:
        """No-op for uniform; assumes input in [-1, 1]."""
        return np.clip(proprio, -1.0, 1.0)

    def inv_norm_action(self, action: np.ndarray) -> np.ndarray:
        return self.inv_action(action)


class RatioMinMaxUniformRobotTokenizer(RobotTokenizer):
    def __init__(self, config: VLADataConfig, vocab_size: int):
        super().__init__(config, vocab_size)
        self.uniform_tokenizer = UniformRobotTokenizer(config, vocab_size)
        self.min_proprio = self.max_proprio = None
        self.min_action = self.max_action = None

    def norm(self, x: np.ndarray, min_v: np.ndarray, max_v: np.ndarray):
        return (x - min_v) / (max_v - min_v + 1e-8) * 2 - 1

    def inv_norm(self, x: np.ndarray, min_v: np.ndarray, max_v: np.ndarray):
        return (x + 1) / 2 * (max_v - min_v) + min_v

    def norm_proprio(self, proprio: np.ndarray):
        if self.min_proprio is None:
            return np.clip(proprio, -1.0, 1.0)
        return self.norm(proprio, self.min_proprio, self.max_proprio)

    def norm_action(self, action: np.ndarray):
        return self.norm(action, self.min_action, self.max_action)

    def norm_goal(self, goal: np.ndarray):
        return self.norm(goal, self.min_proprio[:-1], self.max_proprio[:-1])

    def inv_norm_action(self, action: np.ndarray):
        return self.inv_norm(action, self.min_action, self.max_action)

    def inv_norm_goal(self, goal: np.ndarray):
        return self.inv_norm(goal, self.min_proprio[:-1], self.max_proprio[:-1])

    def bbox(self, bbox: np.ndarray) -> np.ndarray:
        return self.uniform_tokenizer.bbox(bbox)

    def proprio(self, proprio: np.ndarray) -> np.ndarray:
        p = self.norm_proprio(proprio) if self.min_proprio is not None else proprio
        return self.uniform_tokenizer.proprio(p)

    def action(self, action: np.ndarray) -> np.ndarray:
        a = self.norm_action(action) if self.min_action is not None else action
        return self.uniform_tokenizer.action(a)

    def goal(self, goal: np.ndarray) -> np.ndarray:
        g = self.norm_goal(goal) if self.min_proprio is not None else goal
        return self.uniform_tokenizer.goal(g)

    def inv_action(self, action: np.ndarray) -> np.ndarray:
        a = self.uniform_tokenizer.inv_action(action)
        return self.inv_norm_action(a) if self.min_action is not None else a

    def inv_goal(self, goal: np.ndarray) -> np.ndarray:
        g = self.uniform_tokenizer.inv_goal(goal)
        return self.inv_norm_goal(g) if self.min_proprio is not None else g

    def setup(self, get_func: Callable[[], Dict[str, np.ndarray]]):
        try:
            from tqdm import trange
        except ImportError:
            trange = range
        keys = list(get_func().keys())
        results = [[] for _ in keys]
        for _ in trange(self.config.count_num, desc="setup tokenizer"):
            d = get_func()
            for i, k in enumerate(keys):
                results[i].append(d[k])
        for i in range(len(keys)):
            results[i] = np.stack(results[i])

        def set_min_max(data: np.ndarray, eps: float = 1e-7):
            data = data.reshape(-1, data.shape[-1])
            lo = np.percentile(data, self.config.tokenizer_ratio_limit * 100, axis=0) - eps
            hi = np.percentile(data, (1 - self.config.tokenizer_ratio_limit) * 100, axis=0) + eps
            return lo, hi

        if "proprio" in keys:
            self.min_proprio, self.max_proprio = set_min_max(results[keys.index("proprio")])
        if "action" in keys:
            self.min_action, self.max_action = set_min_max(results[keys.index("action")])

    def store_names(self) -> List[str]:
        return ["min_proprio", "max_proprio", "min_action", "max_action"]

    def save(self) -> dict:
        return {n: getattr(self, n) for n in self.store_names() if getattr(self, n) is not None}

    def load(self, data: dict):
        for n in self.store_names():
            setattr(self, n, data.get(n))
