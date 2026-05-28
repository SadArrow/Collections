from __future__ import annotations

import argparse
import os
import pathlib
import re
import sys


def _default_out_dir(*, repo_id: str, revision: str | None) -> pathlib.Path:
    # myVLA/scripts/download_pretrained_vlm.py -> myVLA
    root = pathlib.Path(__file__).resolve().parents[1]
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", repo_id.strip())
    if revision:
        safe = f"{safe}-{revision}"
    return root / "pretrained_vlm" / safe


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a pretrained VLM (SigLIP+Gemma, e.g. PaliGemma) for MEM.")
    parser.add_argument(
        "--repo_id",
        default="google/paligemma-3b-mix-224",
        help="Hugging Face repo id (default: google/paligemma-3b-mix-224)",
    )
    parser.add_argument(
        "--revision",
        default="bfloat16",
        help="HF revision to download (e.g. bfloat16/float16/main). Default: bfloat16",
    )
    parser.add_argument(
        "--out_dir",
        default="",
        help="Output directory (default: myVLA/pretrained_vlm/<repo_id>-<revision>)",
    )
    parser.add_argument(
        "--token",
        default="",
        help="HF token (optional). If omitted, uses HF_TOKEN/HUGGINGFACE_HUB_TOKEN env var or cached login.",
    )
    args = parser.parse_args()

    repo_id = str(args.repo_id).strip()
    if not repo_id:
        raise SystemExit("--repo_id must be non-empty")

    revision = str(args.revision).strip() or None

    out_dir = str(args.out_dir).strip()
    dst = pathlib.Path(out_dir).expanduser().resolve() if out_dir else _default_out_dir(repo_id=repo_id, revision=revision)
    dst.mkdir(parents=True, exist_ok=True)

    token = str(args.token).strip() or None
    if token is None:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    from huggingface_hub import snapshot_download

    # Keep it simple: download the files needed for `from_pretrained`.
    allow_patterns = [
        "*.json",
        "*.safetensors",
        "*.bin",
        "*.model",
        "*.txt",
        "tokenizer.*",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "generation_config.json",
    ]

    try:
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=os.fspath(dst),
            local_dir_use_symlinks=False,
            allow_patterns=allow_patterns,
            token=token,
        )
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        print("Download failed.", file=sys.stderr)
        print(msg, file=sys.stderr)
        print(
            "\nIf this is a gated model, you must accept the model license on Hugging Face and provide a token:",
            file=sys.stderr,
        )
        print("  export HF_TOKEN=...  (or pass --token)", file=sys.stderr)
        raise SystemExit(1) from e

    print(os.fspath(dst))


if __name__ == "__main__":
    main()

