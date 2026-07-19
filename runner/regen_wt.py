#!/usr/bin/env python3
"""On-policy regeneration of a *custom* jsonl via speculators' regen script.

speculators' ``scripts/response_regeneration/script.py`` only regenerates
registered presets (it streams ``load_dataset(preset)``), so it can't take our
sampled ``train.jsonl`` directly. This wrapper loads that script as a module,
feeds it our rows (monkeypatching ``load_dataset``), and drives its async main
unchanged — so the regen logic, retries, and output schema stay exactly
speculators', while the *input* is our frozen sample.

Run under the speculators (base) venv, against a live vLLM server:

    python regen_wt.py \
        --regen-script <spec>/scripts/response_regeneration/script.py \
        --data-file <wt1>/train.jsonl \
        --endpoint http://127.0.0.1:8100/v1/chat/completions \
        --outfile <wt1>/regen/regen_train.jsonl \
        --sampling-params '{"temperature":0.6,"top_p":0.95,"seed":0}' \
        --max-tokens 2048 --concurrency 64 --resume
"""

import argparse
import asyncio
import importlib.util
import json
import sys
from pathlib import Path

CUSTOM_NAME = "wt1_custom"


def load_rows(data_file: str) -> list[dict]:
    """Load our jsonl and give each row a stable ``id`` (= prompt_id).

    The regen script derives its resume key from ``id``/``uuid`` (else a content
    hash); pinning ``id`` to prompt_id keeps ``--resume`` stable across restarts.
    """
    rows = []
    with open(data_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row.setdefault("id", row.get("prompt_id"))
            rows.append(row)
    return rows


def load_regen_module(script_path: str):
    spec = importlib.util.spec_from_file_location("wt_regen_script", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def register_custom(module, name: str) -> None:
    """Register a synthetic preset, matching whatever config shape this
    speculators ref uses.

    Older refs (e.g. #734) keep ``DATASET_CONFIGS`` as plain dicts and gate
    ``--dataset`` on ``choices=list(DATASET_CONFIGS.keys())``. Newer refs use a
    ``DatasetConfig`` dataclass plus a module-level ``REGEN_DATASETS`` list. We
    detect the shape from an existing entry so the wrapper is ref-portable.
    """
    existing = next(iter(module.DATASET_CONFIGS.values()), None)
    if isinstance(existing, dict) or existing is None:
        module.DATASET_CONFIGS[name] = {
            "id": name, "prompt_field": "prompt", "default_split": "train",
        }
    else:  # dataclass DatasetConfig on newer refs
        from speculators.data_generation.configs import DatasetConfig
        module.DATASET_CONFIGS[name] = DatasetConfig(
            name=name, hf_path=name, split="train"
        )
    if hasattr(module, "REGEN_DATASETS"):
        module.REGEN_DATASETS.append(name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regen-script", required=True, help="path to regen script.py")
    ap.add_argument("--data-file", required=True, help="our sampled jsonl")
    ap.add_argument("--endpoint", default="http://127.0.0.1:8100/v1/chat/completions")
    ap.add_argument("--outfile", required=True)
    ap.add_argument("--sampling-params", default='{"temperature":0.6,"top_p":0.95,"seed":0}')
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    rows = load_rows(args.data_file)
    print(f"[regen_wt] loaded {len(rows)} conversations from {args.data_file}", flush=True)

    module = load_regen_module(args.regen_script)

    # Feed our rows in place of the streamed preset. Our conversations are
    # already role/content, so extract_turns needs no normalize/filter.
    module.load_dataset = lambda *a, **k: rows  # noqa: ARG005
    register_custom(module, CUSTOM_NAME)

    Path(args.outfile).parent.mkdir(parents=True, exist_ok=True)

    sys.argv = [
        "regen",
        "--dataset", CUSTOM_NAME,
        "--endpoint", args.endpoint,
        "--outfile", args.outfile,
        "--sampling-params", args.sampling_params,
        "--max-tokens", str(args.max_tokens),
        "--concurrency", str(args.concurrency),
        "--limit", str(len(rows)),
    ]
    if args.resume:
        sys.argv.append("--resume")

    asyncio.run(module.main())


if __name__ == "__main__":
    main()
