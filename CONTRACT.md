# spec-windtunnel — WT-1 contract (DRAFT)

An internal, unofficial leaderboard for `vllm-project/speculators` training changes.
Small frozen task, a few hours per row, so any PR's effect on drafter quality is a
table lookup instead of a multi-day ad-hoc A/B — and so we know when a change is
worth re-training the large checkpoints.

**Non-goals**: not CI (nothing auto-triggers), not a public benchmark, not competing
with anyone's reported numbers. Rows are added deliberately, one at a time.

Values marked ⏳ are frozen by the calibration procedure (§8) before the `wt-1` tag.
Until that tag exists, nothing here is binding.

---

## 1. Lanes

Three independent boards. Each lane has its own frozen config, its own baseline,
its own noise floor, and updates independently.

| lane | verifier | drafter family |
|---|---|---|
| `eagle3` | `Qwen/Qwen3-8B` | Eagle-3, online |
| `dflash` | `Qwen/Qwen3-8B` | DFlash, online |
| `dspark` | `Qwen/Qwen3-8B` | DSpark (Markov + confidence heads), online |

Each lane is scored independently, but all three share the verifier and the data,
so cross-lane reads stay honest — in particular dspark-vs-dflash isolates exactly
what the Markov + confidence heads buy.

## 2. Data (shared recipe, per-verifier artifacts)

- **Source**: `HuggingFaceH4/ultrachat_200k`, HF revision pinned ⏳.
- **Filter**: multi-turn only — ≥ 2 assistant turns (≥ 4 messages).
- **Sample**: seed-pinned uniform sample of N conversations, `N` ⏳ (sized by §8 so a
  Lane-A row fits the §5 budget). Sampling script lives in this repo; the output
  **manifest** (conversation ids + sha256) is committed. The data itself lives on
  disk, never in git.
- **On-policy regeneration**: every row trains on regenerated data — assistant turns
  re-written by the lane's verifier. Frozen regen params: `temperature 0.6,
  top_p 0.95, seed 0, max_tokens 4096`, and **`enable_thinking: true`** — Qwen3-8B
  is a reasoning model; if the target serves with thinking on, the drafter must
  learn to draft `<think>` tokens too, so WT-1 regenerates *with* thinking. This is
  the heavier choice (long generations), which is why `max_tokens`, `seq_length`,
  and the regen budget are all larger. Regen output is cached on disk keyed by
  `(verifier revision, manifest hash, regen params, regen-code fingerprint)`;
  a training-only PR reuses the cache, a data-pipeline PR misses it and pays.
  All three lanes share one verifier, so one regen artifact serves the whole board.
- **Splits** (fixed at sample time, in the manifest):
  - `train` — everything else
  - `val` — 512 held-out conversations → generalization metrics
  - `probe` — 256 held-in training conversations → train-fit metrics (§4)

## 3. Training config (pinned, fully explicit)

Every flag is explicit — repo defaults are never relied on, because repo defaults
are exactly the kind of thing a measured PR may change.

Common to all lanes:

```
--loss-fn '{"ce": 0.1, "tv": 0.9}'
--optimizer muon
--muon-momentum 0.95 --muon-weight-decay 0.1 --muon-ns-steps 5
--muon-adjust-lr-fn match_rms_adamw
--draft-vocab-size 32000
--epochs ⏳          # multi-epoch on purpose: the drafter should approach
                     # convergence on the small train set (see §4)
seed: 0 (baseline uses 0/1/2, see §6)
```

Per-lane:

| | eagle3 | dflash | dspark |
|---|---|---|---|
| lr / muon-lr | 1e-4 / 1e-3 | 3e-4 / 3e-3 | 3e-4 / 3e-3 |
| seq length | 16384 | 16384 | 16384 |
| block size | — | 8 | 8 |
| num layers | (example default) | 5 | 5 |
| target layer ids | (example default) | 2 18 33 | 2 18 33 |
| markov head | — | — | rank 256, vanilla |
| confidence α | — | — | 1.0 |

The only deliberate non-default is the **loss combination** — everything else
mirrors the `examples/train/*` defaults (muon-lr = 10×lr, the repo's derivation),
pinned explicitly so later default-changing PRs can't silently move the benchmark.
One adapted lane: no dspark-on-8B example exists, so dspark takes the dflash-8B
scaffold (5 layers, target layers `2 18 33`) plus its Markov/confidence heads —
which is what makes the dspark-vs-dflash read clean (§1).

## 4. Metrics — every row reports

Core (all lanes):

| column | meaning |
|---|---|
| `val_EAL` | expected accepted length on `val` — the headline score |
| `train_EAL` | same, on the `probe` split — **train-fit** |
| `val_loss`, `train_loss` | final-epoch losses |
| `wall_time` | regen / train / eval breakdown + total |
| `peak_mem` | max reserved GPU memory during training |

Per-lane profile columns:

- `eagle3`: `cond_acc_1..k` (per-depth conditional acceptance)
- `dflash` / `dspark`: `pos_1`, `pos_4`, `pos_7` (per-position acceptance,
  head/mid/tail of the 8-slot block); dspark adds confidence-head calibration
  (`confidence_abs_error`, cumprod bias)

Why train-fit is first-class: if the drafter can't fit the small training set,
generalization numbers are noise about the wrong question. `train_EAL` low →
capacity/optimization problem; `train_EAL` high but `val_EAL` low → generalization
problem. Different diagnoses, different fixes.

## 5. Environment

- 4× GPU (same box class for every row; exact GPU model recorded per row —
  time/memory columns are only comparable within one GPU type).
- GPU split (vLLM vs. train) pinned per lane at freeze ⏳.
- Budget: one row ≤ **2h wall** including regen-cache-miss; ≤ ~1h on cache hit.

## 6. Noise floor (row zero)

Before any comparison row: the baseline config runs **3× (seeds 0/1/2)** per lane.
The board stores mean ± std for every metric and renders each later row's delta as
**significant** (outside the band) or **within-noise** — never raw greens/reds.
Rationale: speculators#788's from-scratch deltas were pure noise; a board without a
noise floor would have called them signal.

## 7. Rows

A row = `(speculators ref, config delta)`. The runner:

1. checks the ref out into a worktree,
2. runs the pinned pipeline (regen from cache if keyed identical),
3. writes `runs/<lane>/<date>-<slug>/{result.json, config, env, log-tail}` — committed.

Row metadata (all in `result.json`): speculators SHA, windtunnel SHA, contract
version, config hash, manifest hash, regen cache key, GPU model/count, seeds, date.

The leaderboard (`LEADERBOARD.md`, one table per lane) is regenerated from
`runs/*` by a render script — never hand-edited.

## 8. Freeze procedure (path to `wt-1`)

1. Write the sampling script; produce + commit the manifest (pins HF revision, N ⏳).
2. **Calibration**: one baseline run per lane; adjust `N` × `epochs` so the row
   fits §5's budget while the loss curve visibly flattens (train-fit regime).
3. Run the 3-seed noise floor (§6).
4. Fill every ⏳, tag `wt-1`, open the board.

Reference points for sizing (from `examples/train/*` headers, 4×H100, 5K×5ep):
eagle3 ≈ 17 min, dflash ≈ 25 min — so the 2h budget has ~4–6× headroom for
more conversations and more epochs.

## 9. Deferred (explicitly out of WT-1)

- **Lane B — fine-tune from a fixed z-lab checkpoint**: the near-convergence
  regime where loss-shape changes (e.g. #788) are visible and from-scratch runs
  are blind. Add as `wt-1.1` once the board is running.
- **SpecForge anchor row**: run their pipeline on our data, eval with **our**
  harness (reported numbers from different harnesses are never comparable).
- **Real-decode eval** (SpecBench/MT-Bench-style acceptance on a fixed prompt
  set) as an extra column.
- Contract changes of any kind → `wt-2`; rows never compare across contract
  versions.
