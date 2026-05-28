#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover a Fold Tops HALO eval summary from an existing rollout dir.")
    parser.add_argument("--source_eval_dir", required=True, help="Existing eval dir with config/plan/episode files.")
    parser.add_argument("--run_dir", required=True, help="Completed rollout run dir under server_viz.")
    parser.add_argument("--results_root", default="", help="Where to place the recovered eval dir. Defaults to source parent.")
    parser.add_argument("--model_label", default="", help="Override model label. Defaults to source config model_label.")
    parser.add_argument("--output_dir", default="", help="Optional explicit recovered eval dir.")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    return _load_json(path) if path.is_file() else None


def _load_plan(source_eval_dir: Path) -> dict[str, Any]:
    summary_path = source_eval_dir / "summary.json"
    if summary_path.is_file():
        summary = _load_json(summary_path)
        results = list(summary.get("results") or [])
        if results:
            plan = dict(results[0].get("plan") or {})
            if plan:
                return plan

    config = _load_json_if_exists(source_eval_dir / "config.json") or {}
    plan_path = Path(str(config.get("plan_path", "")).strip())
    if plan_path.is_file():
        payload = _load_json(plan_path)
        episodes = list(payload.get("episodes") or payload if isinstance(payload, list) else [])
        if episodes:
            return dict(episodes[0])

    raise RuntimeError(f"Could not resolve episode plan from {source_eval_dir}")


def _resolve_output_dir(*, args: argparse.Namespace, source_eval_dir: Path, model_label: str) -> Path:
    if str(args.output_dir).strip():
        return Path(args.output_dir).expanduser().resolve()
    results_root = Path(args.results_root).expanduser().resolve() if str(args.results_root).strip() else source_eval_dir.parent
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return results_root / f"{model_label}_recovered_{timestamp}"


def main() -> int:
    args = _parse_args()
    source_eval_dir = Path(args.source_eval_dir).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve()
    config = _load_json_if_exists(source_eval_dir / "config.json") or {}

    model_label = str(args.model_label).strip() or str(config.get("model_label", "")).strip()
    if not model_label:
        raise RuntimeError("Could not resolve model_label from args or source config.")

    output_dir = _resolve_output_dir(args=args, source_eval_dir=source_eval_dir, model_label=model_label)
    output_dir.mkdir(parents=True, exist_ok=False)
    episode_dir = output_dir / "episode_000"
    episode_dir.mkdir(parents=True, exist_ok=False)

    plan = _load_plan(source_eval_dir)
    run_summary = _load_json_if_exists(run_dir / "summary.json")
    run_status = _load_json_if_exists(run_dir / "run_status.json")
    error_json = _load_json_if_exists(run_dir / "error.json")
    validation = _load_json_if_exists(run_dir / "validation_result.json")

    if validation is None:
        raise RuntimeError(f"Missing validation_result.json in {run_dir}")

    stop_reason = ""
    for payload in (run_summary, run_status, error_json, validation):
        if payload:
            stop_reason = str(payload.get("stop_reason", "")).strip()
            if stop_reason:
                break

    result = {
        "episode_index": int(plan.get("episode_index", 0)),
        "model_label": model_label,
        "return_code": 0,
        "elapsed_s": None,
        "success": bool(validation.get("success")),
        "stop_reason": stop_reason,
        "run_dir": os.fspath(run_dir),
        "validation_result": validation,
        "summary_json": run_summary,
        "run_status_json": run_status,
        "error_json": error_json,
        "plan": plan,
        "launcher_output_log": os.fspath(source_eval_dir / "episode_000" / "launcher_output.log"),
        "validation_log": os.fspath(source_eval_dir / "episode_000" / "validation.jsonl"),
        "final_state_png": str(validation.get("final_state_png", os.fspath(source_eval_dir / "episode_000" / "final_state.png"))),
        "recovered_from_eval_dir": os.fspath(source_eval_dir),
    }

    summary = {
        "model_label": model_label,
        "episodes": 1,
        "successes": 1 if result["success"] else 0,
        "success_rate": 1.0 if result["success"] else 0.0,
        "stop_reasons": {stop_reason: 1} if stop_reason else {},
        "failure_reasons": {} if result["success"] else {stop_reason or "unspecified_failure": 1},
        "results": [result],
        "plan_path": str(config.get("plan_path", "")),
        "eval_dir": os.fspath(output_dir),
        "myvla_root": str(config.get("myvla_root", "")),
        "checkpoint_dir": str(config.get("checkpoint_dir", "")),
        "tokenizer_model": str(config.get("tokenizer_model", "")),
        "hl_vlm_dir": str(config.get("hl_vlm_dir", "")),
        "runtime_dir": str(config.get("runtime_dir", "")),
        "dex_root": str(config.get("dex_root", "")),
        "viz_dir": str(config.get("viz_dir", "")),
        "seed": config.get("seed", 0),
        "recovered_from_eval_dir": os.fspath(source_eval_dir),
        "recovered_from_run_dir": os.fspath(run_dir),
    }

    recovered_config = dict(config)
    recovered_config["recovered_from_eval_dir"] = os.fspath(source_eval_dir)
    recovered_config["recovered_from_run_dir"] = os.fspath(run_dir)
    recovered_config["recovered_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    (output_dir / "config.json").write_text(json.dumps(recovered_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (episode_dir / "episode_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": os.fspath(output_dir), "success": bool(result["success"]), "stop_reason": stop_reason}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
