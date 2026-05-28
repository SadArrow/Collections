#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import os
import posixpath
import stat
import sys
import time
from pathlib import Path

import paramiko


DEFAULT_SAFE_DEST_ROOT = Path(os.environ.get("MYVLA_SERVER_BASE", "/root/workspace/qianyupeng"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy files from a source SSH server into the local workspace.")
    parser.add_argument("--src_host", required=True)
    parser.add_argument("--src_port", type=int, default=22)
    parser.add_argument("--src_user", required=True)
    parser.add_argument("--src_password", required=True)
    parser.add_argument("--src_path", required=True)
    parser.add_argument("--dest_path", required=True)
    parser.add_argument("--safe_dest_root", default=os.fspath(DEFAULT_SAFE_DEST_ROOT))
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Source-relative glob or directory prefix to exclude. Can be passed multiple times.",
    )
    parser.add_argument("--print_every", type=int, default=100)
    return parser.parse_args()


def _ensure_dest_allowed(dest_path: Path, safe_root: Path) -> None:
    dest_resolved = dest_path.resolve()
    safe_resolved = safe_root.resolve()
    if dest_resolved != safe_resolved and safe_resolved not in dest_resolved.parents:
        raise SystemExit(
            f"Refusing to write outside safe root: dest={dest_resolved} safe_root={safe_resolved}"
        )


def _should_exclude(rel_path: str, patterns: list[str]) -> bool:
    rel_norm = rel_path.replace("\\", "/").lstrip("./")
    for pattern in patterns:
        norm_pattern = str(pattern).replace("\\", "/").strip().rstrip("/")
        if not norm_pattern:
            continue
        if fnmatch.fnmatch(rel_norm, norm_pattern):
            return True
        if rel_norm.startswith(norm_pattern + "/"):
            return True
    return False


def _connect(args: argparse.Namespace) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=args.src_host,
        port=int(args.src_port),
        username=args.src_user,
        password=args.src_password,
        timeout=30,
        banner_timeout=30,
        auth_timeout=30,
    )
    return client


def _iter_remote_files(
    sftp: paramiko.SFTPClient,
    remote_root: str,
    *,
    exclude_patterns: list[str],
):
    stack: list[tuple[str, str]] = [(posixpath.normpath(remote_root), "")]
    while stack:
        remote_path, rel_path = stack.pop()
        attrs = sftp.stat(remote_path)
        if stat.S_ISDIR(attrs.st_mode):
            for entry in reversed(sftp.listdir_attr(remote_path)):
                child_remote = posixpath.join(remote_path, entry.filename)
                child_rel = f"{rel_path}/{entry.filename}".lstrip("/")
                if _should_exclude(child_rel, exclude_patterns):
                    continue
                stack.append((child_remote, child_rel))
            continue
        yield remote_path, rel_path, int(attrs.st_size)


def main() -> int:
    args = _parse_args()
    safe_root = Path(args.safe_dest_root).expanduser()
    dest_root = Path(args.dest_path).expanduser()
    _ensure_dest_allowed(dest_root, safe_root)
    dest_root.mkdir(parents=True, exist_ok=True)

    client = _connect(args)
    started_at = time.time()
    copied_files = 0
    copied_bytes = 0
    skipped_files = 0
    skipped_bytes = 0
    try:
        with client.open_sftp() as sftp:
            for index, (remote_file, rel_path, file_size) in enumerate(
                _iter_remote_files(
                    sftp,
                    posixpath.normpath(args.src_path),
                    exclude_patterns=list(args.exclude),
                ),
                start=1,
            ):
                target_path = dest_root / rel_path
                target_path.parent.mkdir(parents=True, exist_ok=True)
                if target_path.is_file() and int(target_path.stat().st_size) == int(file_size):
                    skipped_files += 1
                    skipped_bytes += int(file_size)
                else:
                    sftp.get(remote_file, os.fspath(target_path))
                    copied_files += 1
                    copied_bytes += int(file_size)
                if index % max(1, int(args.print_every)) == 0:
                    elapsed = max(1e-6, time.time() - started_at)
                    done_bytes = copied_bytes + skipped_bytes
                    rate_mib = (copied_bytes / elapsed) / (1024**2)
                    print(
                        f"[sync] files_seen={index} copied={copied_files} skipped={skipped_files} "
                        f"done_gb={done_bytes / (1024**3):.2f} copied_gb={copied_bytes / (1024**3):.2f} "
                        f"rate_mib_s={rate_mib:.2f}",
                        flush=True,
                    )
    finally:
        client.close()

    elapsed = max(1e-6, time.time() - started_at)
    print(
        f"[sync] done copied_files={copied_files} skipped_files={skipped_files} "
        f"copied_gb={copied_bytes / (1024**3):.2f} skipped_gb={skipped_bytes / (1024**3):.2f} "
        f"elapsed_s={elapsed:.1f} rate_mib_s={(copied_bytes / elapsed) / (1024**2):.2f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
