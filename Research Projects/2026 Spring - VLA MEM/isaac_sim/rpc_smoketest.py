from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from rpc_pickle import RpcClient


def _pick_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for myVLA pi0.5 RPC policy server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0, help="0 = auto-pick a free port")
    parser.add_argument("--timeout_s", type=float, default=30.0)

    parser.add_argument("--spawn_server", action="store_true", help="Spawn policy_rpc_server.py as a subprocess")
    parser.add_argument("--server_start_timeout_s", type=float, default=600.0)
    parser.add_argument("--checkpoint_dir", default=str(Path(__file__).resolve().parents[1] / "pi05_droid_pytorch"))
    parser.add_argument("--tokenizer_model", default="")
    parser.add_argument("--device", default="", help="Torch device for pi0.5 (e.g. cuda:0/cpu). Default: auto")
    parser.add_argument("--video_window", type=int, default=4)

    parser.add_argument("--goal", default="fold the shirt")
    parser.add_argument("--num_steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=224)
    args = parser.parse_args()

    host = str(args.host)
    port = int(args.port) if int(args.port) != 0 else _pick_free_port(host)

    server_proc: subprocess.Popen[str] | None = None
    if bool(args.spawn_server):
        myvla_root = Path(__file__).resolve().parents[1]
        server_py = myvla_root / "isaac_sim" / "policy_rpc_server.py"
        cmd = [
            sys.executable,
            os.fspath(server_py),
            "--host",
            host,
            "--port",
            str(port),
            "--checkpoint_dir",
            str(args.checkpoint_dir),
            "--video_window",
            str(int(args.video_window)),
        ]
        if str(args.tokenizer_model).strip():
            cmd += ["--tokenizer_model", str(args.tokenizer_model)]
        if str(args.device).strip():
            cmd += ["--device", str(args.device)]
        server_proc = subprocess.Popen(cmd)  # noqa: S603

    rng = np.random.default_rng(int(args.seed))
    h = int(args.image_size)
    w = int(args.image_size)
    base_rgb = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    left_rgb = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    right_rgb = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    j = rng.standard_normal(size=(7,)).astype(np.float32)
    g = rng.standard_normal(size=(1,)).astype(np.float32)

    try:
        rpc = RpcClient(host=host, port=port, timeout_s=float(args.timeout_s))
        t0 = time.time()
        while True:
            try:
                rpc.call({"cmd": "reset", "goal": str(args.goal), "language_memory": ""})
                break
            except Exception:  # noqa: BLE001
                if not bool(args.spawn_server):
                    raise
                if time.time() - t0 > float(args.server_start_timeout_s):
                    raise TimeoutError("Timed out waiting for RPC server to start") from None
                time.sleep(1.0)

        resp = rpc.call(
            {
                "cmd": "step",
                "step": 0,
                "base_rgb": base_rgb,
                "left_wrist_rgb": left_rgb,
                "right_wrist_rgb": right_rgb,
                "jL": j,
                "gL": g,
                "jR": j,
                "gR": g,
                "num_steps": int(args.num_steps),
            }
        )
        actions_left = np.asarray(resp["actions_left"])
        actions_right = np.asarray(resp["actions_right"])
        assert actions_left.ndim == 2 and actions_left.shape[1] == 8, actions_left.shape
        assert actions_right.ndim == 2 and actions_right.shape[1] == 8, actions_right.shape
        print(f"[ok] actions_left: {actions_left.shape} actions_right: {actions_right.shape}")
        print(f"[ok] viz_run_dir: {resp.get('viz_run_dir')}")

        rpc.call({"cmd": "close"})
        rpc.close()
    finally:
        if server_proc is not None:
            try:
                server_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server_proc.terminate()
                try:
                    server_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    server_proc.kill()


if __name__ == "__main__":
    main()
