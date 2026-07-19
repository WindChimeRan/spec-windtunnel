#!/usr/bin/env python3
"""Collect a calibration run into dashboard-ready metrics.

For each lane dir under <run>:
  - per-epoch val metrics   <lane>/checkpoints/<epoch>/val_metrics.json
  - train loss curve        <lane>/tb/**/events.out.tfevents.*   (all scalars)
  - peak GPU memory         <lane>/mem.jsonl
  - timing / config         <run>/timing.json, <lane>/config.json

Writes, per lane: metrics.jsonl (tidy long-form) + summary.json.
Writes, per run:  board.json + SUMMARY.md.

Tolerant by design: a lane that died mid-run still yields whatever landed, and
the summary says what's missing. Run under the base venv (needs tensorboard).
"""

import argparse
import glob
import json
import os
import re
from pathlib import Path


def read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def eal_from_val(vm: dict):
    """EAL: eal_epoch if written (dflash/dspark), else derive (eagle3)."""
    if vm is None:
        return None
    if "eal_epoch" in vm:
        return vm["eal_epoch"]
    ks = sorted(
        int(m.group(1))
        for k in vm
        if (m := re.fullmatch(r"cond_acc_(\d+)_epoch", k))
    )
    if not ks:
        return None
    eal, cum = 0.0, 1.0
    for k in ks:
        cum *= vm[f"cond_acc_{k}_epoch"]
        eal += cum
    return eal


def position_profile(vm: dict) -> dict:
    """Per-position / per-depth acceptance profile, whichever the lane writes."""
    if vm is None:
        return {}
    prof = {}
    for k, v in vm.items():
        if re.fullmatch(r"position_\d+_acc_epoch", k) or re.fullmatch(
            r"cond_acc_\d+_epoch", k
        ):
            prof[k] = v
    return prof


def collect_val(lane_dir: Path):
    """Return {epoch:int -> val_metrics dict} from checkpoints/<epoch>/."""
    by_epoch = {}
    for p in glob.glob(str(lane_dir / "checkpoints" / "*" / "val_metrics.json")):
        epoch_name = Path(p).parent.name
        if not epoch_name.isdigit():
            continue
        vm = read_json(p)
        if vm is not None:
            by_epoch[int(epoch_name)] = vm
    return dict(sorted(by_epoch.items()))


def collect_tb(lane_dir: Path):
    """All scalar events from tfevents under <lane>/tb. [] if none/unparseable."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except Exception:
        return []
    event_files = glob.glob(str(lane_dir / "tb" / "**" / "events.out.tfevents.*"),
                            recursive=True)
    run_dirs = sorted({str(Path(p).parent) for p in event_files})
    rows = []
    for rd in run_dirs:
        try:
            ea = EventAccumulator(rd, size_guidance={"scalars": 0})
            ea.Reload()
            for tag in ea.Tags().get("scalars", []):
                for ev in ea.Scalars(tag):
                    rows.append(
                        {"source": "tb", "tag": tag, "step": ev.step,
                         "value": ev.value, "wall": ev.wall_time}
                    )
        except Exception:
            continue
    return rows


def _train_eal(vals: dict):
    """Train EAL from end-of-epoch train metrics: direct tag or derive from cond_acc."""
    if "train/eal" in vals:
        return vals["train/eal"]
    ks = sorted(
        int(m.group(1)) for t in vals if (m := re.fullmatch(r"train/cond_acc_(\d+)", t))
    )
    if not ks:
        return None
    eal, cum = 0.0, 1.0
    for k in ks:
        cum *= vals[f"train/cond_acc_{k}"]
        eal += cum
    return eal


def train_by_epoch(tb: list, epochs: int):
    """Per-epoch TRAIN eal/loss/profile from tb, using the end-of-epoch step.

    The trainer logs train metrics per step with no epoch tag, so bin steps into
    epochs by (max_step+1)/epochs and take each epoch's last observed value.
    This is the running train fit — the train-vs-val gap is the generalization
    signal (contract §4).
    """
    train = [r for r in tb if r["tag"].startswith("train/")]
    if not train or not epochs:
        return []
    max_step = max(r["step"] for r in train)
    spe = max(1, round((max_step + 1) / epochs))
    by_ep: dict[int, dict] = {}
    for r in train:
        ep = min(r["step"] // spe, epochs - 1)
        slot = by_ep.setdefault(ep, {})
        prev = slot.get(r["tag"])
        if prev is None or r["step"] >= prev[0]:
            slot[r["tag"]] = (r["step"], r["value"])
    out = []
    for ep in sorted(by_ep):
        vals = {t: v for t, (s, v) in by_ep[ep].items()}
        prof = {
            t.replace("train/", "") + "_epoch": v
            for t, v in vals.items()
            if re.fullmatch(r"train/(cond_acc_\d+|position_\d+_acc)", t)
        }
        out.append({
            "epoch": ep, "eal": _train_eal(vals),
            "loss": vals.get("train/loss"), "profile": prof,
        })
    return out


def collect_mem(lane_dir: Path):
    """Peak used-MiB per gpu + a downsampled series."""
    path = lane_dir / "mem.jsonl"
    peak, series = {}, []
    if not path.exists():
        return {"peak_mib": peak, "series": series}
    with open(path, encoding="utf-8") as f:
        lines = [json.loads(x) for x in f if x.strip()]
    for rec in lines:
        for g, m in rec.get("mem", {}).items():
            peak[str(g)] = max(peak.get(str(g), 0), int(m))
    step = max(1, len(lines) // 400)  # cap series ~400 points
    for rec in lines[::step]:
        if "mem" in rec:
            series.append({"t": rec["t"], "mem": rec["mem"]})
    return {"peak_mib": peak, "series": series}


def summarize_lane(run: Path, lane: str, timing: dict) -> dict:
    lane_dir = run / lane
    cfg = read_json(lane_dir / "config.json") or {}
    val = collect_val(lane_dir)
    tb = collect_tb(lane_dir)
    mem = collect_mem(lane_dir)

    # tidy metrics.jsonl: train (tb) + val (per-epoch, flattened)
    tidy = list(tb)
    for epoch, vm in val.items():
        flat = _flatten(vm)
        for k, v in flat.items():
            tidy.append({"source": "val", "tag": k, "epoch": epoch, "value": v})
        tidy.append({"source": "val", "tag": "eal_derived",
                     "epoch": epoch, "value": eal_from_val(vm)})
    with open(lane_dir / "metrics.jsonl", "w", encoding="utf-8") as f:
        for row in tidy:
            f.write(json.dumps(row) + "\n")

    val_by_epoch = []
    for epoch, vm in val.items():
        val_by_epoch.append({
            "epoch": epoch,
            "eal": eal_from_val(vm),
            "loss": vm.get("loss_epoch"),
            "profile": position_profile(vm),
        })
    # best epoch = lowest val loss (matches trainer's checkpoint_best)
    best = None
    losses = [(e["loss"], e) for e in val_by_epoch if e["loss"] is not None]
    if losses:
        best = min(losses, key=lambda x: x[0])[1]
    final = val_by_epoch[-1] if val_by_epoch else None

    # train-set fit (running, from tb) — the train-vs-val gap is generalization
    train_ep = train_by_epoch(tb, cfg.get("epochs") or 0)

    return {
        "lane": lane,
        "config": cfg,
        "timing_s": {
            "regen_shared": timing.get("regen_s"),
            "prepare_shared": timing.get("prepare_s"),
            "train": timing.get(f"{lane}_train_s"),
        },
        "peak_mem_mib": mem["peak_mib"],
        "val_best": best,
        "val_final": final,
        "val_by_epoch": val_by_epoch,
        "train_by_epoch": train_ep,
        "train_final": train_ep[-1] if train_ep else None,
        "n_train_points": sum(1 for r in tb if r["tag"].endswith("loss")),
        "status": "ok" if val_by_epoch else "no_val_metrics",
    }


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        elif isinstance(v, (int, float)):
            out[key] = v
    return out


def fmt(x, nd=4):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def write_markdown(run: Path, board: dict) -> None:
    lines = ["# Calibration run — summary", ""]
    meta = board.get("meta", {})
    lines += [
        f"- **run**: `{run.name}`",
        f"- **speculators ref**: `{meta.get('spec_sha', '?')}`  ·  "
        f"**windtunnel ref**: `{meta.get('wt_sha', '?')}`",
        f"- **data**: {meta.get('n_conversations', '?')} convs → "
        f"regen `{meta.get('regen_rows', '?')}` rows, "
        f"train max_samples `{meta.get('max_samples', '?')}`, "
        f"epochs `{meta.get('epochs', '?')}`",
        f"- **loss**: `{meta.get('loss_fn', '?')}`  ·  GPUs `{meta.get('gpus', '?')}`",
        "",
        "## Timing (seconds)",
        "",
        "| phase | seconds |",
        "|---|--:|",
        f"| regen (shared) | {fmt(board.get('timing', {}).get('regen_s'), 0)} |",
        f"| prepare (shared) | {fmt(board.get('timing', {}).get('prepare_s'), 0)} |",
    ]
    for lane in board.get("lanes", []):
        t = lane["timing_s"]["train"]
        lines.append(f"| train · {lane['lane']} | {fmt(t, 0)} |")
    lines += ["", "## Results", "",
              "| lane | status | EAL (best) | val loss (best) | best ep | "
              "peak mem (MiB) | train s |", "|---|---|--:|--:|--:|--:|--:|"]
    for lane in board.get("lanes", []):
        best = lane.get("val_best") or {}
        peak = lane.get("peak_mem_mib") or {}
        peak_str = ", ".join(f"{k}:{v}" for k, v in sorted(peak.items())) or "—"
        lines.append(
            f"| {lane['lane']} | {lane['status']} | {fmt(best.get('eal'), 3)} | "
            f"{fmt(best.get('loss'), 4)} | {fmt(best.get('epoch'), 0)} | "
            f"{peak_str} | {fmt(lane['timing_s']['train'], 0)} |"
        )
    lines += ["", "## Per-epoch EAL / loss", ""]
    for lane in board.get("lanes", []):
        lines.append(f"### {lane['lane']}")
        lines.append("")
        lines.append("| epoch | EAL | loss |")
        lines.append("|--:|--:|--:|")
        for e in lane.get("val_by_epoch", []):
            lines.append(f"| {e['epoch']} | {fmt(e.get('eal'), 3)} | "
                         f"{fmt(e.get('loss'), 4)} |")
        lines.append("")
    (run / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--lanes", default="eagle3,dflash,dspark")
    args = ap.parse_args()

    run = Path(args.run)
    timing = read_json(run / "timing.json") or {}
    meta = read_json(run / "env.json") or {}

    lanes = []
    for lane in args.lanes.split(","):
        lane = lane.strip()
        if (run / lane).is_dir():
            lanes.append(summarize_lane(run, lane, timing))

    board = {"meta": meta, "timing": timing, "lanes": lanes}
    (run / "board.json").write_text(json.dumps(board, indent=1))
    write_markdown(run, board)
    print(f"[collect] wrote {run/'board.json'} and {run/'SUMMARY.md'} "
          f"({len(lanes)} lanes)")
    for lane in lanes:
        b = lane.get("val_best") or {}
        print(f"  {lane['lane']:8s} status={lane['status']:16s} "
              f"EAL={fmt(b.get('eal'),3)} loss={fmt(b.get('loss'),4)}")


if __name__ == "__main__":
    main()
