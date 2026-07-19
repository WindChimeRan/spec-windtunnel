#!/usr/bin/env python3
"""Sample per-GPU memory via nvidia-smi until killed; append jsonl samples.

One line per poll: {"t": epoch_seconds, "mem": {gpu_index: used_mib}}. The
collector reduces this to a peak per GPU and keeps the series for the dashboard.

    python mem_sampler.py --gpus 6,7 --interval 2 --out <lane>/mem.jsonl
"""

import argparse
import json
import subprocess
import time


def sample(gpus: list[int]) -> dict[int, int]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    used = {}
    for line in out.strip().splitlines():
        idx, mem = (p.strip() for p in line.split(","))
        if int(idx) in gpus:
            used[int(idx)] = int(mem)
    return used


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpus", required=True, help="comma-separated gpu indices")
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]
    with open(args.out, "a", buffering=1, encoding="utf-8") as f:
        while True:
            try:
                rec = {"t": time.time(), "mem": sample(gpus)}
                f.write(json.dumps(rec) + "\n")
            except Exception as e:  # nvidia-smi hiccup shouldn't kill the sampler
                f.write(json.dumps({"t": time.time(), "error": str(e)[:100]}) + "\n")
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
