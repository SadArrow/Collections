from __future__ import annotations

import argparse
import csv
import subprocess
from typing import Any


def query_gpus() -> list[dict[str, Any]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True)
    rows: list[dict[str, Any]] = []
    for raw in csv.reader(out.strip().splitlines()):
        if len(raw) < 4:
            continue
        rows.append(
            {
                "index": int(raw[0].strip()),
                "name": raw[1].strip(),
                "memory_used": int(raw[2].strip()),
                "utilization": int(raw[3].strip()),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Pick low-utilization GPUs for myVLA server runs.")
    parser.add_argument("--count", type=int, default=3, help="How many GPUs to return.")
    parser.add_argument("--max_memory_used", type=int, default=2048, help="Max used memory (MiB) to consider free.")
    parser.add_argument("--max_utilization", type=int, default=20, help="Max utilization (%%) to consider free.")
    args = parser.parse_args()

    rows = query_gpus()
    free = [
        row
        for row in rows
        if int(row["memory_used"]) <= int(args.max_memory_used) and int(row["utilization"]) <= int(args.max_utilization)
    ]
    free.sort(key=lambda row: (int(row["memory_used"]), int(row["utilization"]), int(row["index"])))
    chosen = [str(row["index"]) for row in free[: max(0, int(args.count))]]
    print(",".join(chosen))


if __name__ == "__main__":
    main()
