#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two Fold Tops HALO-style eval summaries.")
    parser.add_argument("--baseline_summary", required=True)
    parser.add_argument("--candidate_summary", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def _load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _episode_key(item: dict[str, Any]) -> tuple[int, str, float, float]:
    plan = dict(item.get("plan") or {})
    return (
        int(item.get("episode_index", plan.get("episode_index", -1))),
        str(plan.get("garment_usd", "")),
        round(float(plan.get("garment_pos_x", 0.0)), 6),
        round(float(plan.get("garment_pos_y", 0.0)), 6),
    )


def main() -> int:
    args = _parse_args()
    baseline_summary_path = Path(args.baseline_summary).expanduser().resolve()
    candidate_summary_path = Path(args.candidate_summary).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = _load_summary(baseline_summary_path)
    candidate = _load_summary(candidate_summary_path)
    baseline_map = {_episode_key(item): item for item in baseline.get("results", [])}
    candidate_map = {_episode_key(item): item for item in candidate.get("results", [])}
    all_keys = sorted(set(baseline_map) | set(candidate_map))

    rows: list[dict[str, Any]] = []
    transition_counter = Counter()
    for key in all_keys:
        old_item = baseline_map.get(key, {})
        new_item = candidate_map.get(key, {})
        episode_index, garment_usd, pos_x, pos_y = key
        old_success = bool(old_item.get("success"))
        new_success = bool(new_item.get("success"))
        if old_success and new_success:
            transition_counter["both_success"] += 1
        elif (not old_success) and new_success:
            transition_counter["improved"] += 1
        elif old_success and (not new_success):
            transition_counter["regressed"] += 1
        else:
            transition_counter["both_fail"] += 1
        rows.append(
            {
                "episode_index": int(episode_index),
                "garment_usd": garment_usd,
                "garment_pos_x": float(pos_x),
                "garment_pos_y": float(pos_y),
                "baseline_success": old_success,
                "candidate_success": new_success,
                "baseline_stop_reason": str(old_item.get("stop_reason", "")),
                "candidate_stop_reason": str(new_item.get("stop_reason", "")),
                "baseline_run_dir": str(old_item.get("run_dir", "")),
                "candidate_run_dir": str(new_item.get("run_dir", "")),
                "baseline_final_state_png": str(old_item.get("final_state_png", "")),
                "candidate_final_state_png": str(new_item.get("final_state_png", "")),
            }
        )

    comparison = {
        "baseline_summary": os.fspath(baseline_summary_path),
        "candidate_summary": os.fspath(candidate_summary_path),
        "baseline_model_label": baseline.get("model_label", ""),
        "candidate_model_label": candidate.get("model_label", ""),
        "baseline_success_rate": baseline.get("success_rate", 0.0),
        "candidate_success_rate": candidate.get("success_rate", 0.0),
        "delta_success_rate": float(candidate.get("success_rate", 0.0)) - float(baseline.get("success_rate", 0.0)),
        "transitions": dict(transition_counter),
        "rows": rows,
    }

    (output_dir / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "episode_index",
                "garment_usd",
                "garment_pos_x",
                "garment_pos_y",
                "baseline_success",
                "candidate_success",
                "baseline_stop_reason",
                "candidate_stop_reason",
                "baseline_run_dir",
                "candidate_run_dir",
                "baseline_final_state_png",
                "candidate_final_state_png",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    markdown_lines = [
        f"# Fold Tops Eval Comparison",
        "",
        f"- baseline: `{comparison['baseline_model_label']}`",
        f"- candidate: `{comparison['candidate_model_label']}`",
        f"- baseline success rate: `{comparison['baseline_success_rate']}`",
        f"- candidate success rate: `{comparison['candidate_success_rate']}`",
        f"- delta success rate: `{comparison['delta_success_rate']}`",
        f"- transitions: `{json.dumps(comparison['transitions'], ensure_ascii=False)}`",
        "",
        "| episode | garment | pos_x | pos_y | baseline | candidate | baseline reason | candidate reason |",
        "| --- | --- | ---: | ---: | --- | --- | --- | --- |",
    ]
    for row in rows:
        markdown_lines.append(
            "| {episode_index} | {garment_usd} | {garment_pos_x:.4f} | {garment_pos_y:.4f} | {baseline_success} | {candidate_success} | {baseline_stop_reason} | {candidate_stop_reason} |".format(
                **row
            )
        )
    (output_dir / "comparison.md").write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": os.fspath(output_dir), "episodes": len(rows), "transitions": dict(transition_counter)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
