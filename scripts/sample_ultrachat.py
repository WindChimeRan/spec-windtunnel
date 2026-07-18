#!/usr/bin/env python3
"""Sample multi-turn UltraChat conversations for the WT-1 windtunnel data.

Outputs (in --out):
    train.jsonl / val.jsonl / probe.jsonl   one {"prompt_id", "conversations"} per line
    manifest.json                           recipe + split ids + sha256 per file

``probe`` is a held-IN subset of train (the train-fit metric, CONTRACT.md §4):
its conversations also appear in train.jsonl. ``val`` is held out.

The jsonl schema matches speculators' conversations format, so the files feed
``scripts/prepare_data.py --data <path>`` directly (no normalize step).

Stats mode (``--stats``) reports the multi-turn distribution instead of sampling.

Usage:
    export HF_HOME=/mnt/nvme-data/engine/ranran/download/hf
    python scripts/sample_ultrachat.py --stats
    python scripts/sample_ultrachat.py --n 20000 --out <data-dir>/wt1
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

from datasets import load_dataset

DATASET = "HuggingFaceH4/ultrachat_200k"
SPLIT = "train_sft"
# Snapshot on disk; pinned in the manifest so the sample is reproducible.
DEFAULT_REVISION = "8049631c405ae6576f93f445c6b8166f76f5505a"


def n_assistant_turns(messages: list[dict]) -> int:
    return sum(1 for m in messages if m.get("role") == "assistant")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def print_stats(ds) -> None:
    turn_hist: dict[int, int] = {}
    char_lens = []
    for messages in ds["messages"]:
        turns = n_assistant_turns(messages)
        turn_hist[turns] = turn_hist.get(turns, 0) + 1
        char_lens.append(sum(len(m["content"]) for m in messages))
    total = len(ds)
    multi = sum(c for t, c in turn_hist.items() if t >= 2)
    print(f"rows: {total}")
    print(f"rows with >=2 assistant turns: {multi} ({100 * multi / total:.1f}%)")
    print("assistant-turn histogram:")
    for t in sorted(turn_hist):
        print(f"  {t:>2} turns: {turn_hist[t]:>7} ({100 * turn_hist[t] / total:.1f}%)")
    char_lens.sort()
    pct = lambda p: char_lens[int(p / 100 * (len(char_lens) - 1))]  # noqa: E731
    print("conversation char length percentiles:")
    for p in (50, 90, 95, 99):
        print(f"  p{p}: {pct(p):,}")


def write_jsonl(path: Path, rows) -> int:
    n = 0
    with path.open("w") as f:
        for row in rows:
            f.write(
                json.dumps(
                    {"prompt_id": row["prompt_id"], "conversations": row["messages"]},
                    ensure_ascii=False,
                )
                + "\n"
            )
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--revision", default=DEFAULT_REVISION)
    ap.add_argument("--min-assistant-turns", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, help="train conversations (excl. val)")
    ap.add_argument("--val", type=int, default=512)
    ap.add_argument("--probe", type=int, default=256, help="held-IN subset of train")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--stats", action="store_true", help="report distribution only")
    args = ap.parse_args()

    ds = load_dataset(DATASET, split=SPLIT, revision=args.revision)

    if args.stats:
        print_stats(ds)
        return
    if args.n is None or args.out is None:
        ap.error("--n and --out are required unless --stats")

    ds = ds.filter(
        lambda ex: n_assistant_turns(ex["messages"]) >= args.min_assistant_turns,
        num_proc=8,
    )
    need = args.n + args.val
    if need > len(ds):
        raise SystemExit(f"need {need} conversations, only {len(ds)} pass the filter")

    rng = random.Random(args.seed)
    picked = rng.sample(range(len(ds)), need)
    train_idx, val_idx = picked[: args.n], picked[args.n :]
    probe_idx = train_idx[: args.probe]

    args.out.mkdir(parents=True, exist_ok=True)
    splits = {"train": train_idx, "val": val_idx, "probe": probe_idx}
    sizes, hashes, ids = {}, {}, {}
    for name, idx in splits.items():
        rows = ds.select(idx)
        path = args.out / f"{name}.jsonl"
        sizes[name] = write_jsonl(path, rows)
        hashes[name] = sha256_file(path)
        ids[name] = list(rows["prompt_id"])
        print(f"{name}: {sizes[name]} conversations -> {path}")

    manifest = {
        "dataset": DATASET,
        "split": SPLIT,
        "revision": args.revision,
        "min_assistant_turns": args.min_assistant_turns,
        "seed": args.seed,
        "sizes": sizes,
        "sha256": hashes,
        "ids": ids,
    }
    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
