from __future__ import annotations

import pickle
import socket
import struct
from typing import Any, Callable

import numpy as np


class RpcError(RuntimeError):
    pass


_NDARRAY_MARKER = "__ndarray__"


def _encode_for_wire(obj: Any) -> Any:
    # Avoid pickling numpy arrays directly: numpy 2.x pickles reference internal modules
    # (e.g. numpy._core) that may not exist in numpy 1.x, which breaks unpickling across
    # mismatched numpy versions (system python <-> Isaac Sim kit python).
    if isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)
        return {
            _NDARRAY_MARKER: True,
            "dtype": str(arr.dtype),
            "shape": tuple(int(x) for x in arr.shape),
            "data": arr.tobytes(order="C"),
        }
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _encode_for_wire(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        items = [_encode_for_wire(v) for v in obj]
        return tuple(items) if isinstance(obj, tuple) else items
    return obj


def _decode_from_wire(obj: Any) -> Any:
    if isinstance(obj, dict) and obj.get(_NDARRAY_MARKER) is True:
        dtype = np.dtype(obj["dtype"])
        shape = tuple(int(x) for x in obj["shape"])
        data = obj["data"]
        return np.frombuffer(data, dtype=dtype).reshape(shape)
    if isinstance(obj, dict):
        return {k: _decode_from_wire(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode_from_wire(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_decode_from_wire(v) for v in obj)
    return obj


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise EOFError("socket closed")
        buf.extend(chunk)
    return bytes(buf)


def send_obj(sock: socket.socket, obj: Any) -> None:
    data = pickle.dumps(_encode_for_wire(obj), protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack("!I", len(data)))
    sock.sendall(data)


def recv_obj(sock: socket.socket) -> Any:
    header = _recv_exact(sock, 4)
    (n,) = struct.unpack("!I", header)
    data = _recv_exact(sock, int(n))
    return _decode_from_wire(pickle.loads(data))


class RpcClient:
    def __init__(self, *, host: str, port: int, timeout_s: float = 30.0) -> None:
        self.host = str(host)
        self.port = int(port)
        self.timeout_s = float(timeout_s)
        self._sock: socket.socket | None = None

    def connect(self) -> None:
        if self._sock is not None:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.timeout_s)
        sock.connect((self.host, self.port))
        self._sock = sock

    def call(self, msg: dict[str, Any]) -> dict[str, Any]:
        if self._sock is None:
            self.connect()
        assert self._sock is not None
        try:
            send_obj(self._sock, msg)
            resp = recv_obj(self._sock)
        except (EOFError, OSError, socket.timeout) as e:
            self.close()
            raise RpcError(f"RPC transport error: {type(e).__name__}: {e}") from e
        if not isinstance(resp, dict):
            raise RpcError(f"Expected dict response, got {type(resp)}")
        if resp.get("ok") is False:
            raise RpcError(str(resp.get("error", "rpc error")))
        return resp

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None


def serve_forever(
    *,
    host: str,
    port: int,
    handler: Callable[[dict[str, Any]], dict[str, Any]],
    timeout_s: float = 30.0,
    backlog: int = 16,
) -> None:
    """Serve sequential TCP connections until a client sends cmd=close/quit/exit.

    This is intentionally single-threaded and simple. It is resilient to clients
    that connect and immediately disconnect (e.g. port probes).
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((str(host), int(port)))
        server.listen(int(backlog))
        while True:
            conn, _addr = server.accept()
            with conn:
                conn.settimeout(float(timeout_s))
                while True:
                    try:
                        msg = recv_obj(conn)
                    except (EOFError, socket.timeout, OSError):
                        break
                    if not isinstance(msg, dict):
                        try:
                            send_obj(conn, {"ok": False, "error": f"expected dict, got {type(msg)}"})
                        except Exception:  # noqa: BLE001
                            break
                        continue

                    cmd = str(msg.get("cmd", "")).lower()
                    try:
                        resp = handler(msg)
                    except Exception as e:  # noqa: BLE001
                        resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                    if not isinstance(resp, dict):
                        resp = {"ok": False, "error": "handler must return dict"}
                    try:
                        send_obj(conn, resp)
                    except Exception:  # noqa: BLE001
                        break
                    if cmd in ("close", "quit", "exit"):
                        return


def serve_once(
    *, host: str, port: int, handler: Callable[[dict[str, Any]], dict[str, Any]], timeout_s: float = 30.0
) -> None:
    """Backwards-compat shim: serve exactly one connection and exit."""
    served = False
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((str(host), int(port)))
        server.listen(1)
        conn, _addr = server.accept()
        served = True
        with conn:
            conn.settimeout(float(timeout_s))
            while True:
                try:
                    msg = recv_obj(conn)
                except (EOFError, socket.timeout, OSError):
                    break
                if not isinstance(msg, dict):
                    send_obj(conn, {"ok": False, "error": f"expected dict, got {type(msg)}"})
                    continue
                try:
                    resp = handler(msg)
                except Exception as e:  # noqa: BLE001
                    send_obj(conn, {"ok": False, "error": f"{type(e).__name__}: {e}"})
                    continue
                send_obj(conn, resp if isinstance(resp, dict) else {"ok": False, "error": "handler must return dict"})
                if str(msg.get("cmd", "")).lower() in ("close", "quit", "exit"):
                    break
    if not served:
        raise RuntimeError("serve_once() returned without serving a client connection")
