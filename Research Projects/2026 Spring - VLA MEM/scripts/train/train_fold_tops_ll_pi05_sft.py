from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import safetensors.torch
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

from myvla_pi05.droid import DroidInputs
from myvla_pi05.norm_stats import NormStats, load_norm_stats, normalize, pad_to_dim
from myvla_pi05.observation import Observation
from myvla_pi05.pi0_pytorch import PI0Pytorch
from myvla_pi05.policy import _load_model_config, _save_model_config
from myvla_pi05.tokenizer import PaligemmaTokenizer
from myvla_pi05.transformers_patch import ensure_transformers_replace_installed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune the low-level pi0.5 Fold Tops policy from expert trajectories.")
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--manifest", required=True, help="manifest.jsonl produced by collect_fold_tops_ll_expert_dataset.py")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--asset_id", default="droid")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_episodes", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--eval_ratio", type=float, default=0.05)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--joint_delta_scale", type=float, default=0.0, help="0 = auto-estimate from expert deltas")
    parser.add_argument(
        "--train_scope",
        choices=("expert", "full"),
        default="expert",
        help="expert = train only action expert and action projection layers; full = train all parameters",
    )
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    return parser.parse_args()


def _read_manifest(path: Path, *, limit_episodes: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not bool(item.get("success", False)):
                continue
            dataset_path = str(item.get("dataset_path", "")).strip()
            if not dataset_path:
                continue
            records.append(item)
            if int(limit_episodes) > 0 and len(records) >= int(limit_episodes):
                break
    if not records:
        raise RuntimeError(f"No successful low-level expert episodes found in {path}")
    return records


def _squeeze_gripper(x: Any) -> float:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.size <= 0:
        return 0.0
    return float(arr[0])


def _hand_state_to_action_value(text: Any) -> float:
    state = str(text).strip().lower()
    return 1.0 if state == "open" else -1.0


def _compute_auto_joint_delta_scale(records: list[dict[str, Any]]) -> float:
    magnitudes: list[np.ndarray] = []
    for record in records:
        payload = np.load(str(record["dataset_path"]), allow_pickle=True)
        for state_key, next_key in (("left_state", "next_left_state"), ("right_state", "next_right_state")):
            cur = np.asarray(payload[state_key], dtype=np.float32)[..., :7]
            nxt = np.asarray(payload[next_key], dtype=np.float32)[..., :7]
            if cur.size and nxt.size:
                magnitudes.append(np.abs(nxt - cur).reshape(-1))
    if not magnitudes:
        return 0.05
    merged = np.concatenate(magnitudes, axis=0)
    if merged.size <= 0:
        return 0.05
    scale = float(np.quantile(merged, 0.95))
    return max(scale, 1e-3)


def _pad_stats_vector(values: np.ndarray, target_dim: int, *, fill_mean: float, fill_std: float) -> tuple[np.ndarray, np.ndarray]:
    mean = np.full((target_dim,), fill_mean, dtype=np.float32)
    std = np.full((target_dim,), fill_std, dtype=np.float32)
    width = min(int(target_dim), int(values.shape[-1]))
    mean[:width] = values[:width]
    return mean, std


def _stats_dict_from_array(data: np.ndarray, *, target_dim: int) -> dict[str, list[float]]:
    if data.ndim != 2:
        raise ValueError(f"Expected 2D data, got {data.shape}")
    mean = np.zeros((target_dim,), dtype=np.float32)
    std = np.ones((target_dim,), dtype=np.float32)
    q01 = np.zeros((target_dim,), dtype=np.float32)
    q99 = np.zeros((target_dim,), dtype=np.float32)
    width = min(int(target_dim), int(data.shape[1]))
    if width > 0:
        body = np.asarray(data[:, :width], dtype=np.float32)
        mean[:width] = body.mean(axis=0)
        std[:width] = np.clip(body.std(axis=0), 1e-4, None)
        q01[:width] = np.quantile(body, 0.01, axis=0)
        q99[:width] = np.quantile(body, 0.99, axis=0)
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "q01": q01.tolist(),
        "q99": q99.tolist(),
    }


def _build_raw_action_chunk(
    payload: dict[str, np.ndarray],
    *,
    arm: str,
    step_index: int,
    horizon: int,
    joint_delta_scale: float,
) -> np.ndarray:
    if arm == "left":
        state_arr = np.asarray(payload["left_state"], dtype=np.float32)
        next_arr = np.asarray(payload["next_left_state"], dtype=np.float32)
        hand_arr = np.asarray(payload["left_hand_state"], dtype=object)
    else:
        state_arr = np.asarray(payload["right_state"], dtype=np.float32)
        next_arr = np.asarray(payload["next_right_state"], dtype=np.float32)
        hand_arr = np.asarray(payload["right_hand_state"], dtype=object)

    total_steps = int(state_arr.shape[0])
    chunk = np.zeros((int(horizon), 8), dtype=np.float32)
    for offset in range(int(horizon)):
        src = min(int(step_index) + int(offset), max(0, total_steps - 1))
        joint_delta = (np.asarray(next_arr[src], dtype=np.float32)[:7] - np.asarray(state_arr[src], dtype=np.float32)[:7]) / float(
            joint_delta_scale
        )
        chunk[offset, :7] = np.clip(joint_delta, -1.0, 1.0)
        chunk[offset, 7] = _hand_state_to_action_value(hand_arr[src])
    return chunk


def _build_norm_stats(records: list[dict[str, Any]], *, model_action_dim: int, action_horizon: int, joint_delta_scale: float) -> dict[str, NormStats]:
    state_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    for record in records:
        payload = np.load(str(record["dataset_path"]), allow_pickle=True)
        for arm in ("left", "right"):
            if arm == "left":
                state_arr = np.asarray(payload["left_state"], dtype=np.float32)
                gripper_arr = np.asarray(payload["left_gripper"], dtype=np.float32)
            else:
                state_arr = np.asarray(payload["right_state"], dtype=np.float32)
                gripper_arr = np.asarray(payload["right_gripper"], dtype=np.float32)
            total_steps = int(state_arr.shape[0])
            for step_index in range(total_steps):
                state_row = np.concatenate(
                    [
                        np.asarray(state_arr[step_index], dtype=np.float32)[:7],
                        np.asarray([_squeeze_gripper(gripper_arr[step_index])], dtype=np.float32),
                    ],
                    axis=0,
                )
                state_rows.append(pad_to_dim(state_row, int(model_action_dim), axis=-1, value=0.0))
                raw_chunk = _build_raw_action_chunk(
                    payload,
                    arm=arm,
                    step_index=step_index,
                    horizon=int(action_horizon),
                    joint_delta_scale=float(joint_delta_scale),
                )
                action_rows.append(pad_to_dim(raw_chunk, int(model_action_dim), axis=-1, value=0.0).reshape(-1, int(model_action_dim)))
    if not state_rows or not action_rows:
        raise RuntimeError("Failed to derive state/action statistics from low-level expert trajectories.")
    state_matrix = np.stack(state_rows, axis=0).astype(np.float32)
    action_matrix = np.concatenate(action_rows, axis=0).astype(np.float32)
    state_stats = _stats_dict_from_array(state_matrix, target_dim=int(model_action_dim))
    action_stats = _stats_dict_from_array(action_matrix, target_dim=int(model_action_dim))
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
    arm: str
    step_index: int


def _build_sample_index(records: list[dict[str, Any]], *, max_samples: int) -> list[SampleRef]:
    refs: list[SampleRef] = []
    for record in records:
        payload = np.load(str(record["dataset_path"]), allow_pickle=True)
        total_steps = int(np.asarray(payload["subtask"], dtype=object).shape[0])
        for step_index in range(total_steps):
            refs.append(SampleRef(dataset_path=str(record["dataset_path"]), arm="left", step_index=int(step_index)))
            refs.append(SampleRef(dataset_path=str(record["dataset_path"]), arm="right", step_index=int(step_index)))
            if int(max_samples) > 0 and len(refs) >= int(max_samples):
                return refs
    return refs


def _split_refs(refs: list[SampleRef], *, eval_ratio: float, seed: int) -> tuple[list[SampleRef], list[SampleRef]]:
    items = list(refs)
    rng = random.Random(int(seed))
    rng.shuffle(items)
    eval_count = int(round(len(items) * float(eval_ratio)))
    if len(items) >= 20:
        eval_count = max(1, eval_count)
    else:
        eval_count = 0
    eval_items = items[:eval_count]
    train_items = items[eval_count:]
    if not train_items:
        train_items = items
        eval_items = []
    return train_items, eval_items


class FoldTopsLlDataset(Dataset):
    def __init__(
        self,
        refs: list[SampleRef],
        *,
        checkpoint_dir: Path,
        asset_id: str,
        joint_delta_scale: float,
    ) -> None:
        self.refs = list(refs)
        self.checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
        self.asset_id = str(asset_id)
        self.model_config = _load_model_config(self.checkpoint_dir)
        norm_path = self.checkpoint_dir / "assets" / self.asset_id / "norm_stats.json"
        self.norm_stats = load_norm_stats(norm_path)
        self.tokenizer = PaligemmaTokenizer(max_len=int(self.model_config.max_token_len))
        self.droid_inputs = DroidInputs()
        self.joint_delta_scale = float(joint_delta_scale)
        self._cache_path: str = ""
        self._cache_payload: dict[str, np.ndarray] | None = None

    def __len__(self) -> int:
        return len(self.refs)

    def _load_payload(self, path: str) -> dict[str, np.ndarray]:
        if self._cache_payload is not None and self._cache_path == path:
            return self._cache_payload
        with np.load(path, allow_pickle=True) as handle:
            payload = {key: np.asarray(handle[key]) for key in handle.files}
        self._cache_path = str(path)
        self._cache_payload = payload
        return payload

    def __getitem__(self, index: int) -> dict[str, Any]:
        ref = self.refs[int(index)]
        payload = self._load_payload(ref.dataset_path)
        step_index = int(ref.step_index)
        arm = str(ref.arm)

        if arm == "left":
            base_image = np.asarray(payload["left_exterior"][step_index], dtype=np.uint8)
            wrist_image = np.asarray(payload["left_wrist"][step_index], dtype=np.uint8)
            joint_state = np.asarray(payload["left_state"][step_index], dtype=np.float32)[:7]
            gripper_state = np.asarray([_squeeze_gripper(payload["left_gripper"][step_index])], dtype=np.float32)
            prompt = str(np.asarray(payload["prompt_left"], dtype=object)[step_index])
        else:
            base_image = np.asarray(payload["right_exterior"][step_index], dtype=np.uint8)
            wrist_image = np.asarray(payload["right_wrist"][step_index], dtype=np.uint8)
            joint_state = np.asarray(payload["right_state"][step_index], dtype=np.float32)[:7]
            gripper_state = np.asarray([_squeeze_gripper(payload["right_gripper"][step_index])], dtype=np.float32)
            prompt = str(np.asarray(payload["prompt_right"], dtype=object)[step_index])

        droid_example = {
            "observation/exterior_image_1_left": base_image,
            "observation/wrist_image_left": wrist_image,
            "observation/joint_position": joint_state,
            "observation/gripper_position": gripper_state,
            "prompt": prompt,
        }
        inputs = self.droid_inputs(droid_example)
        raw_state = np.asarray(inputs["state"], dtype=np.float32)
        padded_state = pad_to_dim(raw_state, int(self.model_config.action_dim), axis=-1, value=0.0)
        norm_state = normalize(padded_state, self.norm_stats["state"]).astype(np.float32)
        tokens, token_mask = self.tokenizer.tokenize(prompt, norm_state if bool(self.model_config.discrete_state_input) else None)

        raw_action_chunk = _build_raw_action_chunk(
            payload,
            arm=arm,
            step_index=step_index,
            horizon=int(self.model_config.action_horizon),
            joint_delta_scale=float(self.joint_delta_scale),
        )
        padded_action_chunk = pad_to_dim(raw_action_chunk, int(self.model_config.action_dim), axis=-1, value=0.0)
        norm_action_chunk = normalize(padded_action_chunk, self.norm_stats["actions"]).astype(np.float32)

        action_mask = np.zeros((int(self.model_config.action_horizon), int(self.model_config.action_dim)), dtype=np.float32)
        action_mask[:, :8] = 1.0

        return {
            "image": inputs["image"],
            "image_mask": inputs["image_mask"],
            "state": norm_state,
            "tokenized_prompt": tokens.astype(np.int32),
            "tokenized_prompt_mask": token_mask.astype(bool),
            "actions": norm_action_chunk,
            "action_mask": action_mask,
        }


@dataclass
class LlBatch:
    observation: Observation
    actions: torch.Tensor
    action_mask: torch.Tensor


def _collate_ll(batch: list[dict[str, Any]]) -> LlBatch:
    image_keys = tuple(batch[0]["image"].keys())
    images = {
        key: torch.from_numpy(np.stack([np.asarray(item["image"][key], dtype=np.uint8) for item in batch], axis=0))
        for key in image_keys
    }
    image_masks = {
        key: torch.from_numpy(np.asarray([bool(item["image_mask"][key]) for item in batch], dtype=bool))
        for key in image_keys
    }
    state = torch.from_numpy(np.stack([np.asarray(item["state"], dtype=np.float32) for item in batch], axis=0))
    tokenized_prompt = torch.from_numpy(
        np.stack([np.asarray(item["tokenized_prompt"], dtype=np.int32) for item in batch], axis=0)
    ).to(torch.long)
    tokenized_prompt_mask = torch.from_numpy(
        np.stack([np.asarray(item["tokenized_prompt_mask"], dtype=bool) for item in batch], axis=0)
    )
    actions = torch.from_numpy(np.stack([np.asarray(item["actions"], dtype=np.float32) for item in batch], axis=0))
    action_mask = torch.from_numpy(np.stack([np.asarray(item["action_mask"], dtype=np.float32) for item in batch], axis=0))
    observation = Observation.from_dict(
        {
            "image": images,
            "image_mask": image_masks,
            "state": state,
            "tokenized_prompt": tokenized_prompt,
            "tokenized_prompt_mask": tokenized_prompt_mask,
        }
    )
    return LlBatch(observation=observation, actions=actions, action_mask=action_mask)


def _set_trainable_params(model: PI0Pytorch, *, train_scope: str) -> tuple[int, int]:
    total = 0
    trainable = 0
    for name, param in model.named_parameters():
        total += int(param.numel())
        allow = bool(str(train_scope) == "full")
        if not allow:
            allow = (
                name.startswith("action_in_proj")
                or name.startswith("action_out_proj")
                or name.startswith("time_mlp_")
                or name.startswith("state_proj")
                or name.startswith("action_time_mlp_")
                or "paligemma_with_expert.gemma_expert" in name
            )
        param.requires_grad_(allow)
        if allow:
            trainable += int(param.numel())
    return trainable, total


def _evaluate(model: PI0Pytorch, loader: DataLoader, *, device: torch.device, use_bf16: bool) -> float:
    model.eval()
    total_loss = 0.0
    total_weight = 0.0
    autocast_enabled = bool(use_bf16 and device.type == "cuda")
    with torch.no_grad():
        for batch in loader:
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
            batch_weight = float(action_mask.sum().item())
            total_loss += float(loss.item()) * batch_weight
            total_weight += batch_weight
    return total_loss / max(total_weight, 1.0)


def _save_norm_stats(path: Path, stats: dict[str, NormStats]) -> None:
    payload = {
        "norm_stats": {
            "state": {
                "mean": stats["state"].mean.tolist(),
                "std": stats["state"].std.tolist(),
                "q01": stats["state"].q01.tolist() if stats["state"].q01 is not None else None,
                "q99": stats["state"].q99.tolist() if stats["state"].q99 is not None else None,
            },
            "actions": {
                "mean": stats["actions"].mean.tolist(),
                "std": stats["actions"].std.tolist(),
                "q01": stats["actions"].q01.tolist() if stats["actions"].q01 is not None else None,
                "q99": stats["actions"].q99.tolist() if stats["actions"].q99 is not None else None,
            },
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _save_checkpoint(
    *,
    model: PI0Pytorch,
    checkpoint_dir: Path,
    output_dir: Path,
    asset_id: str,
    norm_stats: dict[str, NormStats],
    run_meta: dict[str, Any],
) -> None:
    final_dir = output_dir / "final"
    if final_dir.exists():
        shutil.rmtree(final_dir)
    shutil.copytree(checkpoint_dir, final_dir)
    safetensors.torch.save_model(model, os.fspath(final_dir / "model.safetensors"))
    _save_model_config(final_dir / "config.json", model.config)
    _save_norm_stats(final_dir / "assets" / str(asset_id) / "norm_stats.json", norm_stats)
    (final_dir / "run_meta.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    checkpoint_dir = Path(str(args.checkpoint_dir)).expanduser().resolve()
    manifest_path = Path(str(args.manifest)).expanduser().resolve()
    output_dir = Path(str(args.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ensure_transformers_replace_installed(verbose=True)

    records = _read_manifest(manifest_path, limit_episodes=int(args.max_episodes))
    model_config = _load_model_config(checkpoint_dir)

    joint_delta_scale = float(args.joint_delta_scale)
    if joint_delta_scale <= 0.0:
        joint_delta_scale = _compute_auto_joint_delta_scale(records)

    new_norm_stats = _build_norm_stats(
        records,
        model_action_dim=int(model_config.action_dim),
        action_horizon=int(model_config.action_horizon),
        joint_delta_scale=float(joint_delta_scale),
    )
    norm_stats_out = output_dir / "derived_norm_stats.json"
    _save_norm_stats(norm_stats_out, new_norm_stats)

    refs = _build_sample_index(records, max_samples=int(args.max_samples))
    if not refs:
        raise RuntimeError("No low-level training samples were built from the expert manifest.")
    train_refs, eval_refs = _split_refs(refs, eval_ratio=float(args.eval_ratio), seed=int(args.seed))

    train_dataset = FoldTopsLlDataset(
        train_refs,
        checkpoint_dir=checkpoint_dir,
        asset_id=str(args.asset_id),
        joint_delta_scale=float(joint_delta_scale),
    )
    eval_dataset = (
        FoldTopsLlDataset(
            eval_refs,
            checkpoint_dir=checkpoint_dir,
            asset_id=str(args.asset_id),
            joint_delta_scale=float(joint_delta_scale),
        )
        if eval_refs
        else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.train_batch_size),
        shuffle=True,
        num_workers=0,
        collate_fn=_collate_ll,
    )
    eval_loader = (
        DataLoader(
            eval_dataset,
            batch_size=int(args.eval_batch_size),
            shuffle=False,
            num_workers=0,
            collate_fn=_collate_ll,
        )
        if eval_dataset is not None
        else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PI0Pytorch(model_config)
    safetensors.torch.load_model(model, os.fspath(checkpoint_dir / "model.safetensors"))
    if bool(args.gradient_checkpointing):
        model.gradient_checkpointing_enable()
    trainable, total = _set_trainable_params(model, train_scope=str(args.train_scope))
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
        epoch_progress = float(epoch_idx)
        if epoch_progress >= float(args.epochs):
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
                eval_loss = _evaluate(model, eval_loader, device=device, use_bf16=bool(args.bf16))
                print(json.dumps({"step": int(global_step), "eval_loss": float(eval_loss)}, ensure_ascii=False), flush=True)
                if eval_loss < best_eval:
                    best_eval = float(eval_loss)
                    run_meta = {
                        "ok": True,
                        "checkpoint_dir": os.fspath(checkpoint_dir),
                        "manifest": os.fspath(manifest_path),
                        "output_dir": os.fspath(output_dir),
                        "train_scope": str(args.train_scope),
                        "trainable_params": int(trainable),
                        "total_params": int(total),
                        "joint_delta_scale": float(joint_delta_scale),
                        "train_samples": int(len(train_refs)),
                        "eval_samples": int(len(eval_refs)),
                        "best_eval_loss": float(best_eval),
                    }
                    _save_checkpoint(
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

    final_eval = _evaluate(model, eval_loader, device=device, use_bf16=bool(args.bf16)) if eval_loader is not None else None
    run_meta = {
        "ok": True,
        "checkpoint_dir": os.fspath(checkpoint_dir),
        "manifest": os.fspath(manifest_path),
        "output_dir": os.fspath(output_dir),
        "train_scope": str(args.train_scope),
        "trainable_params": int(trainable),
        "total_params": int(total),
        "joint_delta_scale": float(joint_delta_scale),
        "train_samples": int(len(train_refs)),
        "eval_samples": int(len(eval_refs)),
        "final_eval_loss": None if final_eval is None else float(final_eval),
    }
    _save_checkpoint(
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
