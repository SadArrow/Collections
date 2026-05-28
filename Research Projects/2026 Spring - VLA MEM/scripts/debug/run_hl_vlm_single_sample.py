#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if os.fspath(REPO_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Fold Tops high-level VLM sample through base or LoRA model.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--output_json", default="")
    parser.add_argument("--extra_site_dir", default="")
    return parser.parse_args()


def _load_record(path: Path, sample_index: int) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx != int(sample_index):
                continue
            return json.loads(line)
    raise IndexError(f"sample_index {sample_index} is out of range for {path}")


def main() -> int:
    args = _parse_args()
    if str(args.extra_site_dir).strip():
        extra = str(Path(args.extra_site_dir).expanduser().resolve())
        if extra not in sys.path:
            sys.path.append(extra)

    from PIL import Image

    from myvla_mem.long_term import PretrainedVlmLongTermMemoryProcessor

    manifest_path = Path(str(args.manifest)).expanduser().resolve()
    record = _load_record(manifest_path, int(args.sample_index))
    image_path = Path(str(record["image_path"])).expanduser().resolve()
    image = Image.open(image_path).convert("RGB")

    model = PretrainedVlmLongTermMemoryProcessor(
        model_dir=str(Path(args.model_dir).expanduser().resolve()),
        device=str(args.device),
        max_new_tokens=int(args.max_new_tokens),
        temperature=0.0,
    )
    raw_text = model._generate_text(
        prompt=str(record["prompt_text"]),
        image=image,
        max_new_tokens=int(args.max_new_tokens),
    )

    payload = {
        "manifest": os.fspath(manifest_path),
        "sample_index": int(args.sample_index),
        "model_dir": os.fspath(Path(args.model_dir).expanduser().resolve()),
        "image_path": os.fspath(image_path),
        "target_subtask": str(record.get("target_subtask", "")),
        "target_text": str(record.get("target_text", "")),
        "prompt_text": str(record.get("prompt_text", "")),
        "raw_text": raw_text,
    }

    output_json = str(args.output_json).strip()
    if output_json:
        out_path = Path(output_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
