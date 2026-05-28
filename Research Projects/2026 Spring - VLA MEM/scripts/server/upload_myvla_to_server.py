from __future__ import annotations

import argparse
import fnmatch
import os
import posixpath
from pathlib import Path

import paramiko


def should_exclude(rel_path: str, patterns: list[str]) -> bool:
    rel_norm = rel_path.replace("\\", "/")
    for pattern in patterns:
        if fnmatch.fnmatch(rel_norm, pattern):
            return True
        if rel_norm.startswith(pattern.rstrip("/") + "/"):
            return True
    return False


def ensure_remote_dir(sftp: paramiko.SFTPClient, remote_dir: str, cache: set[str]) -> None:
    parts: list[str] = []
    current = "/"
    for part in remote_dir.strip("/").split("/"):
        parts.append(part)
        current = "/" + "/".join(parts)
        if current in cache:
            continue
        try:
            sftp.stat(current)
        except FileNotFoundError:
            sftp.mkdir(current)
        cache.add(current)


def iter_files(root: Path, exclude_patterns: list[str]):
    for path in root.rglob("*"):
        rel = path.relative_to(root).as_posix()
        if should_exclude(rel, exclude_patterns):
            continue
        if path.is_dir():
            continue
        yield path, rel


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable uploader for myVLA to the remote server.")
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--local_dir", default="myVLA")
    parser.add_argument(
        "--remote_dir",
        default=os.path.join(os.environ.get("MYVLA_SERVER_BASE", "/home/nvme04/qianyupeng"), "myVLA").replace("\\", "/"),
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[
            "isaac_sim_viz",
            "inference_viz",
            "_tmp",
            "__pycache__",
            "isaac_sim_runtime",
            "isaac_sim_assets_cache",
            "*.pyc",
        ],
    )
    parser.add_argument("--print_every", type=int, default=25)
    args = parser.parse_args()

    local_root = Path(args.local_dir).resolve()
    if not local_root.is_dir():
        raise FileNotFoundError(local_root)

    files = list(iter_files(local_root, list(args.exclude)))
    total_bytes = sum(path.stat().st_size for path, _ in files)
    print(f"[upload] files={len(files)} total_gb={total_bytes / (1024**3):.2f}")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=args.host, port=int(args.port), username=args.user, password=args.password, timeout=30)
    sftp = client.open_sftp()
    dir_cache: set[str] = set()
    ensure_remote_dir(sftp, args.remote_dir, dir_cache)

    uploaded = 0
    skipped = 0
    sent_bytes = 0
    for idx, (local_path, rel) in enumerate(files, start=1):
        remote_path = posixpath.join(args.remote_dir, rel.replace("\\", "/"))
        ensure_remote_dir(sftp, posixpath.dirname(remote_path), dir_cache)
        local_size = local_path.stat().st_size
        try:
            remote_stat = sftp.stat(remote_path)
            if int(remote_stat.st_size) == int(local_size):
                skipped += 1
                sent_bytes += int(local_size)
                if idx % max(1, int(args.print_every)) == 0:
                    print(
                        f"[upload] {idx}/{len(files)} skipped={skipped} uploaded={uploaded} "
                        f"progress_gb={sent_bytes / (1024**3):.2f}/{total_bytes / (1024**3):.2f}"
                    )
                continue
        except FileNotFoundError:
            pass
        sftp.put(os.fspath(local_path), remote_path)
        uploaded += 1
        sent_bytes += int(local_size)
        if idx % max(1, int(args.print_every)) == 0 or idx == len(files):
            print(
                f"[upload] {idx}/{len(files)} skipped={skipped} uploaded={uploaded} "
                f"progress_gb={sent_bytes / (1024**3):.2f}/{total_bytes / (1024**3):.2f}"
            )

    sftp.close()
    client.close()
    print(f"[upload] done uploaded={uploaded} skipped={skipped} total_gb={total_bytes / (1024**3):.2f}")


if __name__ == "__main__":
    main()
