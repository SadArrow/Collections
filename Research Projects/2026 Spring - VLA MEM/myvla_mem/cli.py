from __future__ import annotations

import argparse
import datetime as _dt
import pathlib

from myvla_pi05 import droid

from .policy import MemPi05DroidAgent
from .viz import InferenceVizWriter


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pi0.5 + MEM (long/short-term memory) inference demo.")
    parser.add_argument(
        "--checkpoint_dir",
        default=str(pathlib.Path(__file__).resolve().parents[1] / "pi05_droid_pytorch"),
        help="Directory containing model.safetensors, config.json, and assets/*/norm_stats.json",
    )
    parser.add_argument("--device", default="", help="torch device, e.g. cuda / cuda:0 / cpu (default: auto)")
    parser.add_argument("--num_steps", type=int, default=10, help="Flow matching steps (default: 10)")
    parser.add_argument("--tokenizer_model", default="", help="Path to paligemma_tokenizer.model (optional)")
    parser.add_argument("--video_window", type=int, default=1, help="Short-term video memory window (frames).")
    parser.add_argument(
        "--hl_vlm_dir",
        default="",
        help="Pretrained VLM directory/id for long-term memory processor (optional).",
    )
    parser.add_argument("--hl_device", default="", help="HL VLM device (optional, e.g. cpu/cuda:0). Default: policy device")
    parser.add_argument("--hl_dtype", default="bfloat16", help="HL VLM dtype: bfloat16 or float32")
    parser.add_argument("--hl_revision", default="", help="HF revision for HL VLM (e.g. bfloat16, float16).")
    parser.add_argument("--hl_max_new_tokens", type=int, default=128, help="HL generation tokens")
    parser.add_argument("--hl_temperature", type=float, default=0.0, help="HL generation temperature")
    parser.add_argument("--mem_steps", type=int, default=1, help="How many MEM steps to run (default: 1)")
    parser.add_argument("--goal", default="do something", help="Goal instruction (natural language)")
    parser.add_argument("--viz", action="store_true", help="Dump a visualization folder under myVLA/inference_viz/")
    parser.add_argument(
        "--viz_dir",
        default="",
        help="Base directory for viz dumps (optional). If omitted, uses myVLA/inference_viz/ when --viz is set.",
    )
    parser.add_argument("--viz_name", default="", help="Run folder name (optional, default: timestamp)")
    args = parser.parse_args()

    device = args.device.strip() or None
    tokenizer_model = args.tokenizer_model.strip() or None
    hl_vlm_dir = args.hl_vlm_dir.strip() or None
    hl_device = args.hl_device.strip() or None
    hl_revision = args.hl_revision.strip() or None

    agent = MemPi05DroidAgent(
        args.checkpoint_dir,
        device=device,
        tokenizer_model=tokenizer_model,
        video_window=int(args.video_window),
        hl_vlm_dir=hl_vlm_dir,
        hl_device=hl_device,
        hl_dtype=str(args.hl_dtype),
        hl_revision=hl_revision,
        hl_max_new_tokens=int(args.hl_max_new_tokens),
        hl_temperature=float(args.hl_temperature),
    )
    agent.reset(language_memory="")

    viz_enabled = bool(args.viz) or bool(str(args.viz_dir).strip())
    viz_writer: InferenceVizWriter | None = None
    if viz_enabled:
        base_dir = (
            pathlib.Path(str(args.viz_dir)).expanduser().resolve()
            if str(args.viz_dir).strip()
            else pathlib.Path(__file__).resolve().parents[1] / "inference_viz"
        )
        meta = {
            "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "args": vars(args),
        }
        viz_writer = InferenceVizWriter.create(base_dir=base_dir, name=str(args.viz_name).strip() or None, meta=meta)
        print(f"[viz] writing to: {viz_writer.run_dir}")

    last_out: dict | None = None
    for step in range(int(args.mem_steps)):
        example = droid.make_droid_example()
        example["prompt"] = str(args.goal)

        out = agent.step(example, num_steps=int(args.num_steps), debug=viz_enabled)
        last_out = out
        actions = out["actions"]
        memory_preview = (out["language_memory"] or "").replace("\n", " ")
        if len(memory_preview) > 200:
            memory_preview = memory_preview[:200] + "..."

        print(f"[step {step}] subtask: {out['subtask']}")
        print(f"[step {step}] language memory: {memory_preview}")
        print(f"[step {step}] actions shape: {actions.shape}")
        if "hl_raw_text" in out:
            raw = str(out["hl_raw_text"] or "").replace("\n", " ").strip()
            if len(raw) > 400:
                raw = raw[:400] + "..."
            print(f"[step {step}] hl_raw_text: {raw}")

        if viz_writer is not None:
            dbg = out.get("_viz_debug") or {}
            images = dbg.get("images") if isinstance(dbg, dict) else {}
            viz_writer.add_step(
                step,
                goal=str(dbg.get("goal", args.goal)),
                low_level_prompt=str(dbg.get("low_level_prompt", "")),
                prev_memory=str(dbg.get("prev_memory", "")),
                language_memory=str(out.get("language_memory", "")),
                subtask=str(out.get("subtask", "")),
                hl_raw_text=str(out.get("hl_raw_text")) if "hl_raw_text" in out else None,
                structured_state=dict(out.get("structured_state") or {}),
                retrieved_semantic_summary=str(out.get("retrieved_semantic_summary", "")),
                retrieved_visual_summary=str(out.get("retrieved_visual_summary", "")),
                pcmb_debug=dict(out.get("pcmb_debug") or {}),
                images=dict(images) if isinstance(images, dict) else {},
                actions=actions,
            )

    if viz_writer is not None:
        viz_writer.finalize(final_actions=(last_out or {}).get("actions") if isinstance(last_out, dict) else None)


if __name__ == "__main__":
    main()
