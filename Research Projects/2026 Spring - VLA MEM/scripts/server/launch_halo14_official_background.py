#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
from pathlib import Path


DEFAULT_BASE = Path(os.environ.get("MYVLA_SERVER_BASE", "/home/nvme04/qianyupeng"))
DEFAULT_RESULTS_ROOT = DEFAULT_BASE / "eval_results" / "halo14_official"
DEFAULT_RUNNER = DEFAULT_BASE / "myVLA" / "scripts" / "server" / "run_halo14_official_eval.py"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the official HALO14 runner in the background on a specific GPU.")
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--episodes_per_task", type=int, default=1)
    parser.add_argument("--results_root", default=os.fspath(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--runner", default=os.fspath(DEFAULT_RUNNER))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    results_root = Path(args.results_root).expanduser().resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    log_path = results_root / f"{args.label}.out"

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    log_fh = log_path.open("a", encoding="utf-8")
    proc = subprocess.Popen(
        [
            "python3",
            os.fspath(Path(args.runner).expanduser().resolve()),
            "--label",
            str(args.label),
            "--tasks",
            str(args.tasks),
            "--episodes_per_task",
            str(int(args.episodes_per_task)),
            "--results_root",
            os.fspath(results_root),
        ],
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        env=env,
        preexec_fn=os.setsid,
    )
    payload = {
        "pid": int(proc.pid),
        "gpu": str(args.gpu),
        "label": str(args.label),
        "tasks": str(args.tasks),
        "episodes_per_task": int(args.episodes_per_task),
        "results_root": os.fspath(results_root),
        "log_path": os.fspath(log_path),
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
