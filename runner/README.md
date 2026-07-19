# runner/ — WT-1 calibration harness

Drives the frozen pipeline (CONTRACT.md) for all three lanes and collects
dashboard-ready metrics. No training code of its own — it checks out a
speculators ref and runs *its* scripts.

## Pieces

| file | role |
|---|---|
| `calibrate.sh` | orchestrator: regen → prepare → train ×3 → collect |
| `regen_wt.py` | drives speculators' regen script over our custom jsonl (ref-portable) |
| `mem_sampler.py` | polls nvidia-smi → per-GPU peak + series |
| `collect.py` | val_metrics + tfevents + mem → `metrics.jsonl` / `board.json` / `SUMMARY.md` |

## Run / resume

```bash
runner/calibrate.sh                 # default run dir runs/<date>-calib
runner/calibrate.sh runs/my-run     # explicit run dir (resumable)
EPOCHS=5 MAX_SAMPLES=10000 runner/calibrate.sh   # override knobs
```

Idempotent: `.regen.done` / `.prepare.done` / `<lane>/.train.done` markers let a
re-run skip finished phases. Lanes are isolated — one lane failing still runs the
rest and collects. Live progress in `<run>/STATUS.md`.

## Environment (this box)

- **speculators code + torch + tensorboard**: `abtest/venvs/base_venv` → checkout
  `abtest/wt-base`. Pinned to **#734** (`9b74129`) — the newest ref that imports
  cleanly here; current main (#771) needs an `hs_connectors` that isn't installed.
  This is the calibration ref; recorded per-run in `env.json`.
- **vLLM server**: `abtest/venvs/vllm_venv` (vllm 0.25.1). Separate process, talks
  to the trainer over HTTP, so the torch-version gap doesn't matter.
- Two footguns the smoke test surfaced, handled in `calibrate.sh`:
  - vLLM's worker JIT-compiles a custom op and needs **`ninja` on PATH** →
    `PATH` gets `vllm_venv/bin` prepended at launch.
  - `torchrun` isn't on PATH → invoked as `python -m torch.distributed.run`.
- GPUs: regen on `2,3,6,7` (DP=4), then serve on `2,3` (DP=2) + train on `6,7`.
- Model cache + data live under `download/` (`HF_HOME`, `download/windtunnel/wt1`).

## Output layout

```
runs/<date>-calib/
  env.json  timing.json  STATUS.md  SUMMARY.md  board.json
  vllm_regen.log  regen.log  prepare.log  vllm_train.log
  <lane>/  config.json  train.log  mem.jsonl  metrics.jsonl
           checkpoints/<epoch>/val_metrics.json   tb/<run>/events.out.tfevents.*
```
