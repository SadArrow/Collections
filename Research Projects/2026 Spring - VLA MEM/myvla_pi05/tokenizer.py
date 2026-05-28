from __future__ import annotations

import os
import pathlib

import numpy as np
import sentencepiece as spm


def _default_tokenizer_model_path() -> pathlib.Path:
    # myVLA/myvla_pi05/tokenizer.py -> myVLA/assets/paligemma_tokenizer.model
    return pathlib.Path(__file__).resolve().parents[1] / "assets" / "paligemma_tokenizer.model"


class PaligemmaTokenizer:
    def __init__(self, max_len: int = 200, *, model_path: str | os.PathLike[str] | None = None):
        self._max_len = int(max_len)
        if model_path is None:
            env_path = os.environ.get("MYVLA_PALIGEMMA_TOKENIZER_MODEL", "").strip()
            model_path = env_path or _default_tokenizer_model_path()
        model_path = pathlib.Path(model_path).expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(
                f"PaliGemma SentencePiece model not found: {model_path}. "
                "Set MYVLA_PALIGEMMA_TOKENIZER_MODEL or put the file at myVLA/assets/paligemma_tokenizer.model."
            )
        self._tokenizer = spm.SentencePieceProcessor(model_file=str(model_path))

    def tokenize(self, prompt: str, state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        if state is not None:
            state = np.asarray(state, dtype=np.float32).reshape(-1)
            discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
            state_str = " ".join(map(str, discretized_state.tolist()))
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            tokens = self._tokenizer.encode(full_prompt, add_bos=True)
        else:
            tokens = self._tokenizer.encode(cleaned_text, add_bos=True) + self._tokenizer.encode("\n")

        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + padding
            tokens = tokens + padding
        else:
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len

        return np.asarray(tokens, dtype=np.int32), np.asarray(mask, dtype=bool)

