from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if os.fspath(REPO_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPO_ROOT))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("MYVLA_DISABLE_TORCH_COMPILE", "1")

from myvla_pi05.dex_shadow import DexBimanualInputs
from myvla_pi05.norm_stats import NormStats, normalize, pad_to_dim
from myvla_pi05.pi0_pytorch import PI0Pytorch
from myvla_pi05.policy import (
    _load_model_config,
    _load_model_weights,
    _replace_model_config,
)
from myvla_pi05.tokenizer import PaligemmaTokenizer
from myvla_pi05.transformers_patch import ensure_transformers_replace_installed

_BASE_SCRIPT = Path(__file__).resolve().with_name("train_fold_tops_ll_pi05_sft.py")
_BASE_SPEC = importlib.util.spec_from_file_location("_ll_train_base", os.fspath(_BASE_SCRIPT))
if _BASE_SPEC is None or _BASE_SPEC.loader is None:
    raise RuntimeError(f"Unable to load helper training script: {_BASE_SCRIPT}")
_BASE = importlib.util.module_from_spec(_BASE_SPEC)
sys.modules[_BASE_SPEC.name] = _BASE
_BASE_SPEC.loader.exec_module(_BASE)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune pi0.5 on DexGarmentLab dual-arm 60D full-joint targets."
    )
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--asset_id", default="droid")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_episodes", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--eval_ratio", type=float, default=0.05)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--train_scope", choices=("expert", "full"), default="expert")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--active_action_dim", type=int, default=60)
    parser.add_argument("--model_action_dim", type=int, default=64)
    parser.add_argument("--target_mode", choices=("abs", "delta"), default="delta")
    parser.add_argument("--split_unit", choices=("episode", "garment"), default="episode")
    return parser.parse_args()


def _resolve_dataset_path(record: dict[str, Any]) -> str:
    for key in ("dataset_path", "copied_dataset"):
        value = str(record.get(key, "")).strip()
        if value:
            return value
    raise RuntimeError(f"Record does not contain a usable dataset path: {record}")


def _load_payload(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as handle:
        return {key: np.asarray(handle[key]) for key in handle.files}


def _episode_length(payload: dict[str, np.ndarray]) -> int:
    for key in ("joint_state", "left_state_full", "subtask", "prompt_bimanual", "prompt_left"):
        if key in payload:
            return int(np.asarray(payload[key]).shape[0])
    raise RuntimeError(f"Unable to infer episode length from payload keys: {sorted(payload.keys())}")


def _step_text(arr: np.ndarray, step_index: int) -> str:
    return str(np.asarray(arr, dtype=object)[int(step_index)])


def _extract_joint_state(payload: dict[str, np.ndarray], step_index: int) -> np.ndarray:
    if "joint_state" in payload:
        return np.asarray(payload["joint_state"][step_index], dtype=np.float32).reshape(-1)
    left = np.asarray(payload["left_state_full"][step_index], dtype=np.float32).reshape(-1)
    right = np.asarray(payload["right_state_full"][step_index], dtype=np.float32).reshape(-1)
    return np.concatenate([left, right], axis=0).astype(np.float32)


def _extract_next_joint_state(payload: dict[str, np.ndarray], step_index: int) -> np.ndarray:
    if "next_joint_state" in payload:
        return np.asarray(payload["next_joint_state"][step_index], dtype=np.float32).reshape(-1)
    left = np.asarray(payload["next_left_state_full"][step_index], dtype=np.float32).reshape(-1)
    right = np.asarray(payload["next_right_state_full"][step_index], dtype=np.float32).reshape(-1)
    return np.concatenate([left, right], axis=0).astype(np.float32)


def _extract_prompt(payload: dict[str, np.ndarray], step_index: int) -> str:
    if "prompt_bimanual" in payload:
        return _step_text(payload["prompt_bimanual"], step_index)
    if "prompt" in payload:
        return _step_text(payload["prompt"], step_index)
    left_prompt = _step_text(payload["prompt_left"], step_index) if "prompt_left" in payload else ""
    right_prompt = _step_text(payload["prompt_right"], step_index) if "prompt_right" in payload else ""
    if left_prompt and right_prompt:
        return (
            "Bimanual control.\n"
            f"Left arm instruction:\n{left_prompt}\n"
            f"Right arm instruction:\n{right_prompt}"
        )
    return left_prompt or right_prompt


def _extract_images(payload: dict[str, np.ndarray], step_index: int) -> dict[str, np.ndarray]:
    left_exterior = np.asarray(payload["left_exterior"][step_index], dtype=np.uint8)
    right_exterior = np.asarray(payload["right_exterior"][step_index], dtype=np.uint8)
    left_wrist = np.asarray(payload["left_wrist"][step_index], dtype=np.uint8)
    right_wrist = np.asarray(payload["right_wrist"][step_index], dtype=np.uint8)
    return {
        "left_exterior": left_exterior,
        "right_exterior": right_exterior,
        "left_wrist": left_wrist,
        "right_wrist": right_wrist,
    }


def _record_group_key(record: dict[str, Any], *, split_unit: str) -> str:
    if str(split_unit) == "garment":
        candidates = [
            record.get("usd_path"),
            record.get("source_usd_path"),
            (record.get("source_meta") or {}).get("usd_path") if isinstance(record.get("source_meta"), dict) else None,
            record.get("garment_key"),
        ]
        for candidate in candidates:
            text = str(candidate or "").strip()
            if text:
                return f"garment::{text}"
    dataset_path = _resolve_dataset_path(record)
    return f"episode::{dataset_path}"


def _split_records(
    records: list[dict[str, Any]],
    *,
    eval_ratio: float,
    seed: int,
    split_unit: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        key = _record_group_key(record, split_unit=str(split_unit))
        grouped.setdefault(key, []).append(record)

    items = list(grouped.items())
    rng = random.Random(int(seed))
    rng.shuffle(items)
    eval_group_count = int(round(len(items) * float(eval_ratio)))
    if len(items) >= 20:
        eval_group_count = max(1, eval_group_count)
    else:
        eval_group_count = 0
    eval_keys = {key for key, _ in items[:eval_group_count]}
    train_records = [record for key, bucket in items if key not in eval_keys for record in bucket]
    eval_records = [record for key, bucket in items if key in eval_keys for record in bucket]
    if not train_records:
        train_records = list(records)
        eval_records = []
    return train_records, eval_records


def _record_field_values(records: list[dict[str, Any]], key: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for record in records:
        text = str(record.get(key, "") or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        values.append(text)
    return values


def _build_joint_action_chunk(
    payload: dict[str, np.ndarray],
    *,
    step_index: int,
    horizon: int,
    action_dim: int,
    target_mode: str,
) -> np.ndarray:
    total_steps = int(_episode_length(payload))
    chunk = np.zeros((int(horizon), int(action_dim)), dtype=np.float32)
    current_state = _extract_joint_state(payload, int(step_index))[: int(action_dim)]
    for offset in range(int(horizon)):
        src = min(int(step_index) + int(offset), max(0, total_steps - 1))
        next_state = _extract_next_joint_state(payload, src)
        target = np.asarray(next_state, dtype=np.float32)[: int(action_dim)]
        if str(target_mode) == "delta":
            target = target - np.asarray(current_state, dtype=np.float32)
        chunk[offset, : int(action_dim)] = target
    return chunk


def _build_norm_stats(
    records: list[dict[str, Any]],
    *,
    model_action_dim: int,
    action_horizon: int,
    active_action_dim: int,
    target_mode: str,
) -> dict[str, NormStats]:
    state_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    for record in records:
        payload = _load_payload(_resolve_dataset_path(record))
        total_steps = _episode_length(payload)
        for step_index in range(total_steps):
            state_row = _extract_joint_state(payload, step_index)[: int(active_action_dim)]
            state_rows.append(pad_to_dim(state_row, int(model_action_dim), axis=-1, value=0.0))
            raw_chunk = _build_joint_action_chunk(
                payload,
                step_index=step_index,
                horizon=int(action_horizon),
                action_dim=int(active_action_dim),
                target_mode=str(target_mode),
            )
            action_rows.append(
                pad_to_dim(raw_chunk, int(model_action_dim), axis=-1, value=0.0).reshape(-1, int(model_action_dim))
            )
    if not state_rows or not action_rows:
        raise RuntimeError("Failed to derive state/action statistics from Dex bimanual trajectories.")
    state_matrix = np.stack(state_rows, axis=0).astype(np.float32)
    action_matrix = np.concatenate(action_rows, axis=0).astype(np.float32)
    state_stats = _BASE._stats_dict_from_array(state_matrix, target_dim=int(model_action_dim))
    action_stats = _BASE._stats_dict_from_array(action_matrix, target_dim=int(model_action_dim))
    return {
        "state": NormStats(
            mean=np.asarray(state_stats["mean"], dtype=np.float32),
            std=np.asarray(state_stats["std"], dtype=np.float32),
            q01=np.asarray(state_stats["q01"], dtype=np.float32),
            q99=np.asarray(state_stats["q99"], dtype=np.float32),
        ),
        "actions": NormStats(
            mean=np.asarray(action_stats["mean"], dtype=np.float32),
            std=np.asarray(action_stats["std"], dtype=np.float32),
            q01=np.asarray(action_stats["q01"], dtype=np.float32),
            q99=np.asarray(action_stats["q99"], dtype=np.float32),
        ),
    }


@dataclass(frozen=True)
class SampleRef:
    dataset_path: str
    step_index: int


def _build_sample_index(records: list[dict[str, Any]], *, max_samples: int) -> list[SampleRef]:
    refs: list[SampleRef] = []
    for record in records:
        dataset_path = _resolve_dataset_path(record)
        payload = _load_payload(dataset_path)
        total_steps = _episode_length(payload)
        for step_index in range(total_steps):
            refs.append(SampleRef(dataset_path=str(dataset_path), step_index=int(step_index)))
            if int(max_samples) > 0 and len(refs) >= int(max_samples):
                return refs
    return refs


class DexBimanualDataset(Dataset):
    def __init__(
        self,
        refs: list[SampleRef],
        *,
        checkpoint_dir: Path,
        norm_stats: dict[str, NormStats],
        active_action_dim: int,
        model_action_dim: int,
        target_mode: str,
    ) -> None:
        self.refs = list(refs)
        self.checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
        base_config = _load_model_config(self.checkpoint_dir)
        self.model_config = _replace_model_config(base_config, action_dim=int(model_action_dim))
        self.norm_stats = norm_stats
        self.active_action_dim = int(active_action_dim)
        self.target_mode = str(target_mode)
        self.tokenizer = PaligemmaTokenizer(max_len=int(self.model_config.max_token_len))
        self.inputs = DexBimanualInputs()
        self._cache_path: str = ""
        self._cache_payload: dict[str, np.ndarray] | None = None

    def __len__(self) -> int:
        return len(self.refs)

    def _load_payload(self, path: str) -> dict[str, np.ndarray]:
        if self._cache_payload is not None and self._cache_path == path:
            return self._cache_payload
        self._cache_payload = _load_payload(path)
        self._cache_path = str(path)
        return self._cache_payload

    def __getitem__(self, index: int) -> dict[str, Any]:
        ref = self.refs[int(index)]
        payload = self._load_payload(ref.dataset_path)
        step_index = int(ref.step_index)
        state = _extract_joint_state(payload, step_index)[: int(self.active_action_dim)]
        prompt = _extract_prompt(payload, step_index)
        images = _extract_images(payload, step_index)

        half = int(self.active_action_dim // 2)
        inputs = self.inputs(
            {
                "observation/exterior_image_1_left": images["left_exterior"],
                "observation/exterior_image_1_right": images["right_exterior"],
                "observation/wrist_image_left": images["left_wrist"],
                "observation/wrist_image_right": images["right_wrist"],
                "observation/joint_position_left": state[:half],
                "observation/joint_position_right": state[half : half * 2],
                "prompt": prompt,
            }
        )

        raw_state = np.asarray(inputs["state"], dtype=np.float32)
        padded_state = pad_to_dim(raw_state, int(self.model_config.action_dim), axis=-1, value=0.0)
        norm_state = normalize(padded_state, self.norm_stats["state"]).astype(np.float32)
        tokens, token_mask = self.tokenizer.tokenize(prompt, norm_state if bool(self.model_config.discrete_state_input) else None)

        raw_action_chunk = _build_joint_action_chunk(
            payload,
            step_index=step_index,
            horizon=int(self.model_config.action_horizon),
            action_dim=int(self.active_action_dim),
            target_mode=str(self.target_mode),
        )
        padded_action_chunk = pad_to_dim(raw_action_chunk, int(self.model_config.action_dim), axis=-1, value=0.0)
        norm_action_chunk = normalize(padded_action_chunk, self.norm_stats["actions"]).astype(np.float32)

        action_mask = np.zeros((int(self.model_config.action_horizon), int(self.model_config.action_dim)), dtype=np.float32)
        action_mask[:, : int(self.active_action_dim)] = 1.0

        return {
            "image": inputs["image"],
            "image_mask": inputs["image_mask"],
            "state": norm_state,
            "tokenized_prompt": tokens.astype(np.int32),
            "tokenized_prompt_mask": token_mask.astype(bool),
            "actions": norm_action_chunk,
            "action_mask": action_mask,
        }


def main() -> None:
    args = _parse_args()
    checkpoint_dir = Path(str(args.checkpoint_dir)).expanduser().resolve()
    manifest_path = Path(str(args.manifest)).expanduser().resolve()
    output_dir = Path(str(args.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ensure_transformers_replace_installed(verbose=True)

    records = _BASE._read_manifest(manifest_path, limit_episodes=int(args.max_episodes))
    base_model_config = _load_model_config(checkpoint_dir)
    model_config = _replace_model_config(base_model_config, action_dim=int(args.model_action_dim))

    train_records, eval_records = _split_records(
        records,
        eval_ratio=float(args.eval_ratio),
        seed=int(args.seed),
        split_unit=str(args.split_unit),
    )

    new_norm_stats = _build_norm_stats(
        train_records,
        model_action_dim=int(model_config.action_dim),
        action_horizon=int(model_config.action_horizon),
        active_action_dim=int(args.active_action_dim),
        target_mode=str(args.target_mode),
    )
    _BASE._save_norm_stats(output_dir / "derived_norm_stats.json", new_norm_stats)

    train_refs = _build_sample_index(train_records, max_samples=int(args.max_samples))
    eval_refs = _build_sample_index(eval_records, max_samples=0)
    if not train_refs:
        raise RuntimeError("No low-level training samples were built from the Dex bimanual manifest.")

    train_dataset = DexBimanualDataset(
        train_refs,
        checkpoint_dir=checkpoint_dir,
        norm_stats=new_norm_stats,
        active_action_dim=int(args.active_action_dim),
        model_action_dim=int(model_config.action_dim),
        target_mode=str(args.target_mode),
    )
    eval_dataset = (
        DexBimanualDataset(
            eval_refs,
            checkpoint_dir=checkpoint_dir,
            norm_stats=new_norm_stats,
            active_action_dim=int(args.active_action_dim),
            model_action_dim=int(model_config.action_dim),
            target_mode=str(args.target_mode),
        )
        if eval_refs
        else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.train_batch_size),
        shuffle=True,
        num_workers=0,
        collate_fn=_BASE._collate_ll,
    )
    eval_loader = (
        DataLoader(
            eval_dataset,
            batch_size=int(args.eval_batch_size),
            shuffle=False,
            num_workers=0,
            collate_fn=_BASE._collate_ll,
        )
        if eval_dataset is not None
        else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PI0Pytorch(model_config)
    load_summary = _load_model_weights(
        model,
        checkpoint_dir / "model.safetensors",
        allow_partial=bool(int(model_config.action_dim) != int(base_model_config.action_dim)),
    )
    if bool(args.gradient_checkpointing):
        model.gradient_checkpointing_enable()
    trainable, total = _BASE._set_trainable_params(model, train_scope=str(args.train_scope))
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    total_steps = int(math.ceil(len(train_loader) * float(args.epochs)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))

    autocast_enabled = bool(bool(args.bf16) and device.type == "cuda")
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    best_eval = float("inf")

    for epoch_idx in range(int(math.ceil(float(args.epochs)))):
        if float(epoch_idx) >= float(args.epochs):
            break
        running_loss = 0.0
        running_weight = 0.0
        for batch_idx, batch in enumerate(train_loader):
            global_step += 1
            observation = batch.observation
            observation.state = observation.state.to(device)
            observation.tokenized_prompt = observation.tokenized_prompt.to(device)
            observation.tokenized_prompt_mask = observation.tokenized_prompt_mask.to(device)
            observation.images = {key: value.to(device) for key, value in observation.images.items()}
            observation.image_masks = {key: value.to(device) for key, value in observation.image_masks.items()}
            actions = batch.actions.to(device)
            action_mask = batch.action_mask.to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                loss_map = model(observation, actions)
                weighted = loss_map * action_mask
                loss = weighted.sum() / action_mask.sum().clamp_min(1.0)
                loss = loss / float(max(1, int(args.gradient_accumulation_steps)))

            loss.backward()
            batch_weight = float(action_mask.sum().item())
            running_loss += float(loss.item()) * float(max(1, int(args.gradient_accumulation_steps))) * batch_weight
            running_weight += batch_weight

            if global_step % int(max(1, int(args.gradient_accumulation_steps))) == 0:
                torch.nn.utils.clip_grad_norm_(
                    [param for param in model.parameters() if param.requires_grad],
                    max_norm=float(args.max_grad_norm),
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if global_step % int(max(1, int(args.logging_steps))) == 0:
                avg_loss = running_loss / max(running_weight, 1.0)
                print(
                    json.dumps(
                        {
                            "step": int(global_step),
                            "epoch": float(epoch_idx) + float(batch_idx + 1) / float(max(1, len(train_loader))),
                            "train_loss": float(avg_loss),
                            "lr": float(scheduler.get_last_lr()[0]),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                running_loss = 0.0
                running_weight = 0.0

            if eval_loader is not None and global_step % int(max(1, int(args.eval_steps))) == 0:
                eval_loss = _BASE._evaluate(model, eval_loader, device=device, use_bf16=bool(args.bf16))
                print(json.dumps({"step": int(global_step), "eval_loss": float(eval_loss)}, ensure_ascii=False), flush=True)
                if eval_loss < best_eval:
                    best_eval = float(eval_loss)
                    run_meta = {
                        "ok": True,
                        "checkpoint_dir": os.fspath(checkpoint_dir),
                        "manifest": os.fspath(manifest_path),
                        "output_dir": os.fspath(output_dir),
                        "train_scope": str(args.train_scope),
                        "split_unit": str(args.split_unit),
                        "train_episodes": int(len(train_records)),
                        "eval_episodes": int(len(eval_records)),
                        "action_target_mode": str(args.target_mode),
                        "source_prompt_styles": _record_field_values(records, "prompt_style"),
                        "source_observation_sources": _record_field_values(records, "observation_source"),
                        "trainable_params": int(trainable),
                        "total_params": int(total),
                        "active_action_dim": int(args.active_action_dim),
                        "model_action_dim": int(model_config.action_dim),
                        "train_samples": int(len(train_refs)),
                        "eval_samples": int(len(eval_refs)),
                        "best_eval_loss": float(best_eval),
                        "load_summary": load_summary,
                    }
                    _BASE._save_checkpoint(
                        model=model,
                        checkpoint_dir=checkpoint_dir,
                        output_dir=output_dir,
                        asset_id=str(args.asset_id),
                        norm_stats=new_norm_stats,
                        run_meta=run_meta,
                    )

            if global_step >= total_steps:
                break
        if global_step >= total_steps:
            break

    final_eval = _BASE._evaluate(model, eval_loader, device=device, use_bf16=bool(args.bf16)) if eval_loader is not None else None
    run_meta = {
        "ok": True,
        "checkpoint_dir": os.fspath(checkpoint_dir),
        "manifest": os.fspath(manifest_path),
        "output_dir": os.fspath(output_dir),
        "train_scope": str(args.train_scope),
        "split_unit": str(args.split_unit),
        "train_episodes": int(len(train_records)),
        "eval_episodes": int(len(eval_records)),
        "action_target_mode": str(args.target_mode),
        "source_prompt_styles": _record_field_values(records, "prompt_style"),
        "source_observation_sources": _record_field_values(records, "observation_source"),
        "trainable_params": int(trainable),
        "total_params": int(total),
        "active_action_dim": int(args.active_action_dim),
        "model_action_dim": int(model_config.action_dim),
        "train_samples": int(len(train_refs)),
        "eval_samples": int(len(eval_refs)),
        "final_eval_loss": None if final_eval is None else float(final_eval),
        "load_summary": load_summary,
    }
    _BASE._save_checkpoint(
        model=model,
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        asset_id=str(args.asset_id),
        norm_stats=new_norm_stats,
        run_meta=run_meta,
    )
    (output_dir / "run_meta.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(run_meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
