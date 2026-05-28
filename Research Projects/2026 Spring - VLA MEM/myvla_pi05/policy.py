from __future__ import annotations

import dataclasses
import json
import os
import pathlib
from typing import Any

import numpy as np
import safetensors.torch
import torch
from safetensors import safe_open

from . import droid
from . import norm_stats as _norm
from .observation import Observation
from .tokenizer import PaligemmaTokenizer
from .transformers_patch import ensure_transformers_replace_installed


@dataclasses.dataclass(frozen=True)
class Pi05ModelConfig:
    action_dim: int = 32
    action_horizon: int = 15
    max_token_len: int = 200
    dtype: str = "bfloat16"
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    pi05: bool = True
    discrete_state_input: bool = True


def _replace_model_config(config: Pi05ModelConfig, **overrides: Any) -> Pi05ModelConfig:
    values = dataclasses.asdict(config)
    for key, value in overrides.items():
        if value is None:
            continue
        if key not in values:
            raise KeyError(f"Unknown Pi05ModelConfig field: {key}")
        values[key] = value
    return Pi05ModelConfig(**values)


def _load_model_config(checkpoint_dir: pathlib.Path) -> Pi05ModelConfig:
    cfg_path = checkpoint_dir / "config.json"
    if not cfg_path.is_file():
        return Pi05ModelConfig()
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    # Historical: config.json uses "precision"; openpi uses "dtype".
    precision = raw.get("precision", raw.get("dtype", "bfloat16"))
    return Pi05ModelConfig(
        action_dim=int(raw.get("action_dim", 32)),
        action_horizon=int(raw.get("action_horizon", 15)),
        max_token_len=int(raw.get("max_token_len", 200)),
        dtype=str(precision),
        paligemma_variant=str(raw.get("paligemma_variant", "gemma_2b")),
        action_expert_variant=str(raw.get("action_expert_variant", "gemma_300m")),
        pi05=True,
        discrete_state_input=True,
    )


def _tree_map(f, x):
    if isinstance(x, dict):
        return {k: _tree_map(f, v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_tree_map(f, v) for v in x)
    return f(x)


def _save_model_config(path: pathlib.Path, config: Pi05ModelConfig) -> None:
    payload = {
        "action_dim": int(config.action_dim),
        "action_horizon": int(config.action_horizon),
        "max_token_len": int(config.max_token_len),
        "paligemma_variant": str(config.paligemma_variant),
        "action_expert_variant": str(config.action_expert_variant),
        "precision": str(config.dtype),
        "dtype": str(config.dtype),
        "pi05": bool(config.pi05),
        "discrete_state_input": bool(config.discrete_state_input),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_model_weights(
    model: torch.nn.Module,
    checkpoint_path: pathlib.Path,
    *,
    allow_partial: bool = False,
) -> dict[str, Any]:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not bool(allow_partial):
        safetensors.torch.load_model(model, os.fspath(checkpoint_path))
        return {
            "mode": "strict",
            "loaded": len(model.state_dict()),
            "skipped_shape_mismatch": [],
            "missing": [],
        }

    state = model.state_dict()
    loaded: list[str] = []
    skipped_shape_mismatch: list[dict[str, Any]] = []
    ignored_extra: list[str] = []
    partially_expanded: list[dict[str, Any]] = []
    with safe_open(os.fspath(checkpoint_path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key not in state:
                ignored_extra.append(str(key))
                continue
            tensor = handle.get_tensor(key)
            target = state[key]
            if tuple(tensor.shape) != tuple(target.shape):
                expandable = (
                    str(key).startswith("action_in_proj")
                    or str(key).startswith("action_out_proj")
                    or str(key).startswith("state_proj")
                )
                if expandable and tensor.ndim == target.ndim and tensor.ndim in (1, 2):
                    slices = tuple(slice(0, min(int(src), int(dst))) for src, dst in zip(tensor.shape, target.shape))
                    with torch.no_grad():
                        target[slices].copy_(tensor[slices].to(dtype=target.dtype))
                    loaded.append(str(key))
                    partially_expanded.append(
                        {
                            "name": str(key),
                            "checkpoint_shape": list(tensor.shape),
                            "model_shape": list(target.shape),
                            "copied_slices": [min(int(src), int(dst)) for src, dst in zip(tensor.shape, target.shape)],
                        }
                    )
                    continue
                skipped_shape_mismatch.append(
                    {
                        "name": str(key),
                        "checkpoint_shape": list(tensor.shape),
                        "model_shape": list(target.shape),
                    }
                )
                continue
            with torch.no_grad():
                target.copy_(tensor.to(dtype=target.dtype))
            loaded.append(str(key))
    missing = [str(key) for key in state.keys() if str(key) not in set(loaded)]
    summary = {
        "mode": "partial",
        "loaded": len(loaded),
        "partially_expanded": partially_expanded,
        "skipped_shape_mismatch": skipped_shape_mismatch,
        "ignored_extra": ignored_extra,
        "missing": missing,
    }
    if skipped_shape_mismatch or partially_expanded:
        print(
            json.dumps(
                {
                    "event": "pi05_partial_checkpoint_load",
                    "checkpoint": os.fspath(checkpoint_path),
                    "loaded_tensors": int(summary["loaded"]),
                    "partially_expanded_names": [item["name"] for item in partially_expanded],
                    "shape_mismatch_count": len(skipped_shape_mismatch),
                    "shape_mismatch_names": [item["name"] for item in skipped_shape_mismatch],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return summary


class Pi05DroidPolicy:
    def __init__(
        self,
        checkpoint_dir: str | os.PathLike[str],
        *,
        device: str | None = None,
        asset_id: str = "droid",
        tokenizer_model: str | os.PathLike[str] | None = None,
        config_overrides: dict[str, Any] | None = None,
        allow_partial_load: bool = False,
    ):
        self._checkpoint_dir = pathlib.Path(checkpoint_dir).expanduser().resolve()
        self.io_mode = "droid"
        print(f"[pi05_policy] checkpoint_dir={self._checkpoint_dir}", flush=True)
        if not (self._checkpoint_dir / "model.safetensors").is_file():
            raise FileNotFoundError(f"model.safetensors not found in: {self._checkpoint_dir}")

        print("[pi05_policy] ensure_transformers_replace_installed:start", flush=True)
        ensure_transformers_replace_installed(verbose=True)
        print("[pi05_policy] ensure_transformers_replace_installed:done", flush=True)

        self._config = _load_model_config(self._checkpoint_dir)
        if config_overrides:
            self._config = _replace_model_config(self._config, **dict(config_overrides))
        print(
            "[pi05_policy] model_config="
            + json.dumps(dataclasses.asdict(self._config), ensure_ascii=False),
            flush=True,
        )

        self._device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[pi05_policy] device={self._device}", flush=True)
        self._action_q_clip_margin_ratio = float(
            str(os.environ.get("MYVLA_ACTION_Q_CLIP_MARGIN_RATIO", "0.05")).strip() or "0.05"
        )
        run_meta_path = self._checkpoint_dir / "run_meta.json"
        self._run_meta: dict[str, Any] = {}
        if run_meta_path.is_file():
            self._run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
        self._action_target_mode = str(self._run_meta.get("action_target_mode", "abs")).strip().lower() or "abs"
        print(
            "[pi05_policy] run_meta="
            + json.dumps(
                {
                    "path": os.fspath(run_meta_path),
                    "action_target_mode": self._action_target_mode,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        norm_path = self._checkpoint_dir / "assets" / asset_id / "norm_stats.json"
        if not norm_path.is_file():
            raise FileNotFoundError(f"norm_stats.json not found: {norm_path}")
        self._norm_stats = _norm.load_norm_stats(norm_path)
        print(f"[pi05_policy] loaded_norm_stats={norm_path}", flush=True)

        print("[pi05_policy] tokenizer_init:start", flush=True)
        self._tokenizer = PaligemmaTokenizer(max_len=self._config.max_token_len, model_path=tokenizer_model)
        print("[pi05_policy] tokenizer_init:done", flush=True)

        # Import after applying transformers_replace so patched Gemma/PaliGemma/SigLIP modules are picked up.
        print("[pi05_policy] import_pi0_pytorch:start", flush=True)
        from .pi0_pytorch import PI0Pytorch
        print("[pi05_policy] import_pi0_pytorch:done", flush=True)

        print("[pi05_policy] model_construct:start", flush=True)
        self._model = PI0Pytorch(self._config)
        print("[pi05_policy] model_construct:done", flush=True)
        print("[pi05_policy] model_weights_load:start", flush=True)
        self._load_summary = _load_model_weights(
            self._model,
            self._checkpoint_dir / "model.safetensors",
            allow_partial=bool(allow_partial_load),
        )
        print(
            "[pi05_policy] model_weights_load:done "
            + json.dumps(self._load_summary, ensure_ascii=False),
            flush=True,
        )
        print("[pi05_policy] model_to_device:start", flush=True)
        self._model.to(self._device)
        print("[pi05_policy] model_to_device:done", flush=True)
        self._model.eval()
        print("[pi05_policy] model_eval:done", flush=True)

        self._inputs = droid.DroidInputs()
        self._outputs = droid.DroidOutputs()

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        return self._checkpoint_dir

    def infer(self, example: dict, *, num_steps: int = 10) -> dict[str, Any]:
        inputs = self._inputs(example)
        prompt = inputs.pop("prompt")

        # Pad + normalize state to model action_dim
        raw_state = np.asarray(inputs["state"], dtype=np.float32)
        padded_raw_state = _norm.pad_to_dim(raw_state, self._config.action_dim, axis=-1, value=0.0)
        state = _norm.normalize(padded_raw_state, self._norm_stats["state"])
        inputs["state"] = state

        # Tokenize prompt, using normalized state for pi05 discrete-state input
        tokens, token_mask = self._tokenizer.tokenize(prompt, state if self._config.discrete_state_input else None)
        inputs["tokenized_prompt"] = tokens
        inputs["tokenized_prompt_mask"] = token_mask

        # Convert to torch, add batch dim, move to device
        torch_inputs = _tree_map(lambda x: torch.from_numpy(np.array(x)).to(self._device)[None, ...], inputs)
        obs = Observation.from_dict(torch_inputs)

        with torch.no_grad():
            actions = self._model.sample_actions(self._device, obs, num_steps=int(num_steps))

        actions_np = np.asarray(actions[0].detach().cpu(), dtype=np.float32)
        actions_np = _norm.unnormalize(actions_np, self._norm_stats["actions"])
        clipped_actions = _norm.clip_to_quantile_range(
            actions_np,
            self._norm_stats["actions"],
            margin_ratio=float(self._action_q_clip_margin_ratio),
        )
        max_clip_delta = float(np.max(np.abs(clipped_actions - actions_np))) if clipped_actions.size else 0.0
        if max_clip_delta > 1.0e-5:
            print(
                "[pi05_policy] clipped_actions_to_quantile_range "
                + json.dumps(
                    {
                        "max_clip_delta": max_clip_delta,
                        "margin_ratio": float(self._action_q_clip_margin_ratio),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        actions_np = clipped_actions
        if self._action_target_mode == "delta":
            actions_np = actions_np + padded_raw_state[None, :]

        outputs = {"actions": actions_np}
        outputs = self._outputs(outputs)
        return outputs


class Pi05DexShadowPolicy(Pi05DroidPolicy):
    def __init__(
        self,
        checkpoint_dir: str | os.PathLike[str],
        *,
        device: str | None = None,
        asset_id: str = "droid",
        tokenizer_model: str | os.PathLike[str] | None = None,
        active_action_dim: int = 60,
        model_action_dim: int = 64,
    ):
        super().__init__(
            checkpoint_dir,
            device=device,
            asset_id=asset_id,
            tokenizer_model=tokenizer_model,
            config_overrides={"action_dim": int(model_action_dim)},
            allow_partial_load=True,
        )
        self.io_mode = "dex_bimanual"
        self.active_action_dim = int(active_action_dim)
        self.model_action_dim = int(model_action_dim)
        from .dex_shadow import DexBimanualInputs, DexBimanualOutputs

        self._inputs = DexBimanualInputs()
        self._outputs = DexBimanualOutputs(active_action_dim=int(active_action_dim))


class Pi05DexBimanualPolicy(Pi05DexShadowPolicy):
    pass
