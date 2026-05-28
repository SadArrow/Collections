from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import posixpath
import re
import sys
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class ServerConfig:
    host: str
    user: str
    password: str
    my_folder: str


def _parse_server_config(text: str) -> ServerConfig:
    def pick(pattern: str) -> str | None:
        m = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        return (m.group(1).strip() if m else None)  # type: ignore[return-value]

    host = pick(r"^\s*(?:HostName|Host)\s+([^\s#]+)\s*$") or ""
    user = pick(r"^\s*User\s+([^\s#]+)\s*$") or ""
    password = pick(r"^\s*Password\s*:\s*([^\r\n#]+)\s*$") or ""
    my_folder = pick(r"^\s*My\s*Folder\s*:\s*([^\r\n#]+)\s*$") or ""

    if not host or not user or not password or not my_folder:
        raise ValueError("Failed to parse server_config.txt (need Host/HostName, User, Password, My Folder).")

    my_folder = my_folder.strip()
    if not my_folder.startswith("/"):
        my_folder = "/" + my_folder
    my_folder = posixpath.normpath(my_folder)

    return ServerConfig(host=host, user=user, password=password, my_folder=my_folder)


def _safe_join(root: str, rel: str) -> str:
    root = posixpath.normpath(root)
    rel = rel.lstrip("/").strip()
    if not rel:
        raise ValueError("remote_rel_path must be non-empty")
    out = posixpath.normpath(posixpath.join(root, rel))
    if out != root and not out.startswith(root.rstrip("/") + "/"):
        raise ValueError(f"Refusing to write outside root. root={root!r}, rel={rel!r}, out={out!r}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload a small smoke-test file to the remote server (ONLY under 'My Folder')."
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[2] / "server_config.txt"),
        help="Path to server_config.txt",
    )
    parser.add_argument(
        "--remote_rel_path",
        default="codex_smoketest.txt",
        help="Remote relative path under 'My Folder' (default: codex_smoketest.txt)",
    )
    parser.add_argument(
        "--content",
        default="",
        help="File content (optional). If omitted, uses a timestamped message.",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    text = cfg_path.read_text(encoding="utf-8", errors="replace")
    cfg = _parse_server_config(text)

    try:
        import paramiko  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            "Missing dependency: paramiko.\n"
            "Install it first:\n"
            "  python -m pip install paramiko\n"
        ) from e

    remote_path = _safe_join(cfg.my_folder, str(args.remote_rel_path))
    content = str(args.content)
    if not content.strip():
        now = _dt.datetime.now().isoformat(timespec="seconds")
        content = f"codex smoketest upload ok @ {now}\n"

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(cfg.host, username=cfg.user, password=cfg.password, look_for_keys=False, allow_agent=False, timeout=20)
    try:
        sftp = ssh.open_sftp()
        try:
            # Ensure target directory exists (do not create it implicitly).
            parent = posixpath.dirname(remote_path)
            sftp.stat(parent)

            with sftp.open(remote_path, "w") as f:
                f.write(content)
            st = sftp.stat(remote_path)
        finally:
            sftp.close()
    finally:
        ssh.close()

    print(remote_path)
    print(f"bytes={int(getattr(st, 'st_size', -1))}")


if __name__ == "__main__":
    main()

