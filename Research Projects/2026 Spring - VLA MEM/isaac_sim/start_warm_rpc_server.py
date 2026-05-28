from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from rpc_pickle import RpcClient

_RUNTIME_PYDEPS_MARKERS = (
    "WorldModelDiffusionVlaRuntime/pydeps",
    "WorldModelDiffusionVlaRuntime\\pydeps",
)


def _pick_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def _load_state(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _ping(host: str, port: int, timeout_s: float) -> dict[str, Any] | None:
    rpc = RpcClient(host=host, port=port, timeout_s=timeout_s)
    try:
        return rpc.call({"cmd": "ping"})
    except Exception:
        return None
    finally:
        rpc.close()


def _ps_quote(text: str) -> str:
    return "'" + str(text).replace("'", "''") + "'"


def _sanitized_child_env() -> dict[str, str]:
    env = dict(os.environ)
    raw_pythonpath = str(env.get("PYTHONPATH", "") or "")
    if raw_pythonpath:
        kept = [
            token
            for token in raw_pythonpath.split(os.pathsep)
            if token and not any(marker in token for marker in _RUNTIME_PYDEPS_MARKERS)
        ]
        if kept:
            env["PYTHONPATH"] = os.pathsep.join(kept)
        else:
            env.pop("PYTHONPATH", None)
    env.setdefault("PYTHONNOUSERSITE", "1")
    return env


def _spawn_detached_windows(cmd: list[str], *, cwd: Path, log_path: str | None = None) -> None:
    file_path = _ps_quote(str(cmd[0]))
    arg_list = ",".join(_ps_quote(str(x)) for x in cmd[1:])
    launcher = [
        "powershell.exe",
        "-NoProfile",
        "-Command",
        f"Start-Process -FilePath {file_path} -WorkingDirectory {_ps_quote(os.fspath(cwd))} "
        f"-WindowStyle Hidden -ArgumentList {arg_list}",
    ]
    subprocess.run(launcher, check=True, cwd=os.fspath(cwd), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # noqa: S603


def main() -> None:
    parser = argparse.ArgumentParser(description="Start or reuse a persistent warm myVLA RPC server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0, help="0 = auto/free port if starting a new server")
    parser.add_argument("--timeout_s", type=float, default=10.0)
    parser.add_argument("--start_timeout_s", type=float, default=600.0)
    parser.add_argument("--python_exe", default=sys.executable)
    parser.add_argument("--state_file", default=str(Path(__file__).resolve().parents[1] / "isaac_sim_runtime" / "rpc_server_state.json"))
    parser.add_argument("--checkpoint_dir", default=str(Path(__file__).resolve().parents[1] / "pi05_droid_pytorch"))
    parser.add_argument("--tokenizer_model", default="")
    parser.add_argument("--device", default="")
    parser.add_argument("--video_window", type=int, default=4)
    parser.add_argument("--hl_vlm_dir", default="")
    parser.add_argument("--hl_device", default="")
    parser.add_argument("--hl_dtype", default="bfloat16")
    parser.add_argument("--hl_revision", default="")
    parser.add_argument("--hl_max_new_tokens", type=int, default=64)
    parser.add_argument("--hl_temperature", type=float, default=0.0)
    parser.add_argument("--viz_dir", default="")
    parser.add_argument("--viz_name", default="")
    parser.add_argument("--log_path", default="", help="Optional log file for the detached warm server.")
    parser.add_argument("--force_restart", action="store_true")
    args = parser.parse_args()

    state_path = Path(str(args.state_file)).expanduser().resolve()
    if not bool(args.force_restart):
        state = _load_state(state_path)
        if state and state.get("ready"):
            host = str(state.get("host", args.host))
            port = int(state.get("port", 0) or 0)
            state_cuda_visible_devices = str(state.get("cuda_visible_devices", "") or "").strip()
            desired_cuda_visible_devices = str(os.environ.get("CUDA_VISIBLE_DEVICES", "") or "").strip()
            if port > 0:
                pong = _ping(host, port, float(args.timeout_s))
                if pong and pong.get("ready"):
                    if state_cuda_visible_devices == desired_cuda_visible_devices:
                        print(
                            json.dumps(
                                {
                                    "reused": True,
                                    "host": host,
                                    "port": port,
                                    "state_file": os.fspath(state_path),
                                }
                            )
                        )
                        return
    else:
        try:
            state_path.unlink()
        except FileNotFoundError:
            pass

    myvla_root = Path(__file__).resolve().parents[1]
    server_py = myvla_root / "isaac_sim" / "policy_rpc_server.py"
    port = int(args.port) if int(args.port) > 0 else _pick_free_port(str(args.host))
    cmd = [
        str(args.python_exe),
        "-u",
        os.fspath(server_py),
        "--host",
        str(args.host),
        "--port",
        str(port),
        "--checkpoint_dir",
        str(args.checkpoint_dir),
        "--video_window",
        str(int(args.video_window)),
        "--state_file",
        os.fspath(state_path),
    ]
    if str(args.tokenizer_model).strip():
        cmd += ["--tokenizer_model", str(args.tokenizer_model)]
    if str(args.device).strip():
        cmd += ["--device", str(args.device)]
    if str(args.hl_vlm_dir).strip():
        cmd += ["--hl_vlm_dir", str(args.hl_vlm_dir)]
        if str(args.hl_device).strip():
            cmd += ["--hl_device", str(args.hl_device)]
        if str(args.hl_dtype).strip():
            cmd += ["--hl_dtype", str(args.hl_dtype)]
        if str(args.hl_revision).strip():
            cmd += ["--hl_revision", str(args.hl_revision)]
        cmd += ["--hl_max_new_tokens", str(int(args.hl_max_new_tokens))]
        cmd += ["--hl_temperature", str(float(args.hl_temperature))]
    if str(args.viz_dir).strip():
        cmd += ["--viz_dir", str(args.viz_dir)]
    if str(args.viz_name).strip():
        cmd += ["--viz_name", str(args.viz_name)]
    log_path_str = ""
    if str(args.log_path).strip():
        log_path = Path(str(args.log_path)).expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path_str = os.fspath(log_path)
    if os.name == "nt":
        _spawn_detached_windows(cmd, cwd=myvla_root, log_path=log_path_str or None)
        spawned_proc = None
    else:
        popen_kwargs: dict[str, Any] = {"cwd": os.fspath(myvla_root), "env": _sanitized_child_env()}
        if log_path_str:
            stdout_handle = open(log_path_str, "a", encoding="utf-8")
            popen_kwargs["stdout"] = stdout_handle
            popen_kwargs["stderr"] = subprocess.STDOUT
        spawned_proc = subprocess.Popen(cmd, **popen_kwargs)  # noqa: S603

    deadline = time.time() + float(args.start_timeout_s)
    while time.time() < deadline:
        if spawned_proc is not None:
            return_code = spawned_proc.poll()
            if return_code is not None:
                raise RuntimeError(f"Warm RPC server process exited before ready (returncode={return_code})")
        pong = _ping(str(args.host), int(port), float(args.timeout_s))
        if pong and pong.get("ready"):
            print(json.dumps({"reused": False, "host": str(args.host), "port": int(port), "state_file": os.fspath(state_path)}))
            return
        time.sleep(1.0)
    raise TimeoutError(f"Timed out waiting for warm RPC server at {args.host}:{port}")


if __name__ == "__main__":
    main()
