from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EXTRA_SITE_DIR = os.environ.get("MYVLA_EXTRA_SITE_DIR", "").strip()
if EXTRA_SITE_DIR and EXTRA_SITE_DIR not in sys.path:
    # Append extra site-packages after the environment defaults so Isaac's bundled
    # torch/transformers stay authoritative while optional packages (e.g. peft)
    # can still be discovered from a workspace-local directory.
    sys.path.append(EXTRA_SITE_DIR)

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PaliGemmaForConditionalGeneration, Trainer, TrainingArguments


REPO_ROOT = Path(__file__).resolve().parents[2]
if os.fspath(REPO_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPO_ROOT))

from myvla_mem.long_term import _safe_dtype_for_device, _to_torch_pixel_values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA/SFT fine-tuning for the high-level Fold Tops VLM.")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--manifest", required=True, help="manifest.jsonl produced by prepare_fold_tops_hl_sft_dataset.py")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--eval_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)

    parser.add_argument("--max_prompt_tokens", type=int, default=768)
    parser.add_argument("--max_target_tokens", type=int, default=192)
    parser.add_argument("--max_seq_len", type=int, default=1536)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--use_4bit", action="store_true")

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    return parser.parse_args()


def _trim_terminal_eos(token_ids: list[int], eos_token_id: int | None) -> list[int]:
    if eos_token_id is None:
        return list(token_ids)
    trimmed = list(token_ids)
    while trimmed and trimmed[-1] == int(eos_token_id):
        trimmed.pop()
    return trimmed


def _load_manifest(path: Path, limit: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not str(record.get("prompt_text", "")).strip():
                continue
            if not str(record.get("target_text", "")).strip():
                continue
            if not str(record.get("image_path", "")).strip():
                continue
            records.append(record)
            if int(limit) > 0 and len(records) >= int(limit):
                break
    if not records:
        raise RuntimeError(f"No usable records found in {path}")
    return records


def _split_records(records: list[dict[str, Any]], *, eval_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    items = list(records)
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


class FoldTopsHlSftDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return dict(self.records[int(index)])


@dataclass
class FoldTopsHlCollator:
    tokenizer: Any
    image_token_id: int
    num_image_tokens: int
    target_image_size: int
    dtype: torch.dtype
    max_prompt_tokens: int
    max_target_tokens: int
    max_seq_len: int

    def _encode_text_pair(self, prompt: str, target: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prompt_ids = self.tokenizer(
            str(prompt),
            add_special_tokens=True,
            truncation=True,
            max_length=int(self.max_prompt_tokens),
        )["input_ids"]
        prompt_ids = _trim_terminal_eos(prompt_ids, self.tokenizer.eos_token_id)
        if not prompt_ids:
            raise ValueError("Prompt tokenization produced an empty sequence.")

        target_ids = self.tokenizer(
            str(target),
            add_special_tokens=False,
            truncation=True,
            max_length=int(self.max_target_tokens),
        )["input_ids"]
        if self.tokenizer.eos_token_id is not None:
            target_ids = list(target_ids) + [int(self.tokenizer.eos_token_id)]

        text_input_ids = torch.tensor(prompt_ids + list(target_ids), dtype=torch.long)
        text_labels = torch.tensor(
            ([-100] * len(prompt_ids)) + list(target_ids),
            dtype=torch.long,
        )

        bos = text_input_ids[:1]
        rest = text_input_ids[1:]
        bos_labels = text_labels[:1]
        rest_labels = text_labels[1:]
        image_tokens = torch.full((int(self.num_image_tokens),), int(self.image_token_id), dtype=torch.long)
        image_labels = torch.full((int(self.num_image_tokens),), -100, dtype=torch.long)

        input_ids = torch.cat([bos, image_tokens, rest], dim=0)
        labels = torch.cat([bos_labels, image_labels, rest_labels], dim=0)
        attention_mask = torch.ones_like(input_ids)

        if input_ids.numel() > int(self.max_seq_len):
            input_ids = input_ids[: int(self.max_seq_len)]
            labels = labels[: int(self.max_seq_len)]
            attention_mask = attention_mask[: int(self.max_seq_len)]
        return input_ids, attention_mask, labels

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        pad_token_id = int(self.tokenizer.pad_token_id)
        input_ids_list: list[torch.Tensor] = []
        attention_masks: list[torch.Tensor] = []
        labels_list: list[torch.Tensor] = []
        pixel_values: list[torch.Tensor] = []

        for item in batch:
            image = Image.open(str(item["image_path"])).convert("RGB")
            sample_input_ids, sample_attention_mask, sample_labels = self._encode_text_pair(
                str(item["prompt_text"]),
                str(item["target_text"]),
            )
            input_ids_list.append(sample_input_ids)
            attention_masks.append(sample_attention_mask)
            labels_list.append(sample_labels)
            pixel_values.append(
                _to_torch_pixel_values(
                    image,
                    device=torch.device("cpu"),
                    dtype=self.dtype,
                    target_image_size=int(self.target_image_size),
                )[0].to(torch.float32)
            )

        batch_input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_id)
        batch_attention_mask = pad_sequence(attention_masks, batch_first=True, padding_value=0)
        batch_labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)
        batch_pixel_values = torch.stack(pixel_values, dim=0).to(self.dtype)

        batch_labels = batch_labels.masked_fill(batch_attention_mask.eq(0), -100)
        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "labels": batch_labels,
            "pixel_values": batch_pixel_values,
        }


def _resolve_lora_target_modules(model: torch.nn.Module) -> list[str]:
    suffixes = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    names: list[str] = []
    for name, module in model.named_modules():
        if "vision_tower" in name:
            continue
        if name == "model.multi_modal_projector.linear":
            names.append(name)
            continue
        if any(name.endswith(f".{suffix}") or name == suffix for suffix in suffixes):
            if "language_model" in name:
                names.append(name)
    if not names:
        raise RuntimeError("Failed to find LoRA target modules for the high-level VLM.")
    return sorted(set(names))


def _build_model(args: argparse.Namespace) -> tuple[Any, Any, torch.dtype]:
    dtype_t = torch.bfloat16 if str(args.dtype) == "bfloat16" else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_t = _safe_dtype_for_device(dtype_t, device)

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token

    model_kwargs: dict[str, Any] = {}
    if bool(args.use_4bit):
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype_t,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = dtype_t

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        str(args.model_dir),
        **model_kwargs,
    )
    model.config.use_cache = False
    if bool(args.gradient_checkpointing):
        model.gradient_checkpointing_enable()

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if bool(args.use_4bit):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=bool(args.gradient_checkpointing))

    lora_config = LoraConfig(
        r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=_resolve_lora_target_modules(model),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer, dtype_t


def main() -> None:
    args = _parse_args()
    output_dir = Path(str(args.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(str(args.manifest)).expanduser().resolve()
    records = _load_manifest(manifest_path, int(args.max_samples))
    train_records, eval_records = _split_records(records, eval_ratio=float(args.eval_ratio), seed=int(args.seed))

    model, tokenizer, dtype_t = _build_model(args)
    target_image_size = int(model.config.vision_config.image_size)
    patch_size = int(model.config.vision_config.patch_size)
    image_token_id = int(model.config.image_token_id)
    num_image_tokens = (target_image_size // patch_size) ** 2

    collator = FoldTopsHlCollator(
        tokenizer=tokenizer,
        image_token_id=image_token_id,
        num_image_tokens=num_image_tokens,
        target_image_size=target_image_size,
        dtype=dtype_t,
        max_prompt_tokens=int(args.max_prompt_tokens),
        max_target_tokens=int(args.max_target_tokens),
        max_seq_len=int(args.max_seq_len),
    )

    training_args = TrainingArguments(
        output_dir=os.fspath(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=float(args.epochs),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        warmup_ratio=float(args.warmup_ratio),
        per_device_train_batch_size=int(args.train_batch_size),
        per_device_eval_batch_size=int(args.eval_batch_size),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        gradient_checkpointing=bool(args.gradient_checkpointing),
        logging_steps=int(args.logging_steps),
        save_steps=int(args.save_steps),
        eval_steps=int(args.eval_steps),
        save_total_limit=int(args.save_total_limit),
        eval_strategy="steps" if eval_records else "no",
        save_strategy="steps",
        bf16=bool(str(args.dtype) == "bfloat16" and torch.cuda.is_available()),
        fp16=False,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=FoldTopsHlSftDataset(train_records),
        eval_dataset=FoldTopsHlSftDataset(eval_records) if eval_records else None,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(os.fspath(output_dir / "final"))
    tokenizer.save_pretrained(os.fspath(output_dir / "final"))

    meta = {
        "ok": True,
        "model_dir": str(args.model_dir),
        "manifest": os.fspath(manifest_path),
        "output_dir": os.fspath(output_dir),
        "train_samples": int(len(train_records)),
        "eval_samples": int(len(eval_records)),
        "dtype": str(args.dtype),
        "use_4bit": bool(args.use_4bit),
        "target_image_size": int(target_image_size),
        "num_image_tokens": int(num_image_tokens),
    }
    (output_dir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
