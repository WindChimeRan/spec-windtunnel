#!/usr/bin/env bash
# WT-1 calibration orchestrator (unattended).
#
# Phases (idempotent; re-run skips completed phases via .done markers):
#   0  on-policy regen of train.jsonl        (vLLM DP=4 on 2,3,6,7)
#   1  prepare_data -> shared Arrow          (CPU)
#   2  launch training vLLM                  (DP=2 on 2,3)
#   3  train+instrument eagle3/dflash/dspark (torchrun on 6,7)
#   4  collect metrics -> board.json/SUMMARY.md
#
# Lanes are isolated: one lane failing does not stop the others. Regen/prepare/
# server-launch failures are fatal (everything downstream depends on them).
set -uo pipefail

# ---- paths / params ----
BASE_PY=/mnt/nvme-data/engine/ranran/abtest/venvs/base_venv/bin/python
VLLM_PY=/mnt/nvme-data/engine/ranran/abtest/venvs/vllm_venv/bin/python
SPEC=/mnt/nvme-data/engine/ranran/abtest/wt-base
WT=/mnt/nvme-data/engine/ranran/spec-windtunnel
DATA=/mnt/nvme-data/engine/ranran/download/windtunnel/wt1
REGEN_OUT=$DATA/regen/regen_train.jsonl
PREPARED=$DATA/prepared
MODEL="Qwen/Qwen3-8B"
PORT=${PORT:-8100}
EPOCHS=${EPOCHS:-4}
MAX_SAMPLES=${MAX_SAMPLES:-6000}
SEQLEN=${SEQLEN:-16384}          # full multi-turn conversation cap (train + prepare)
MAX_TOKENS=${MAX_TOKENS:-4096}   # regen: per-response generation cap
REGEN_LIMIT=${REGEN_LIMIT:-2000} # regen: first N convs of the manifest (0 = all)
REGEN_GPUS=${REGEN_GPUS:-2,3,6,7}
SERVE_GPUS=${SERVE_GPUS:-2,3}
TRAIN_GPUS=${TRAIN_GPUS:-6,7}
# enable_thinking:true -> Qwen3 emits <think> traces then the answer; the drafter
# must learn to draft thinking tokens if the target serves with thinking on. Heavier
# (long generations), so seq/max_tokens are larger and N is smaller for calibration.
SAMPLING=${SAMPLING:-'{"temperature":0.6,"top_p":0.95,"seed":0,"chat_template_kwargs":{"enable_thinking":true}}'}
VLLM_BIN=$(dirname "$VLLM_PY")  # worker JIT needs `ninja` from the venv bin on PATH

RUN=${1:-$WT/runs/$(date +%Y%m%d)-calib}
mkdir -p "$RUN"
export HF_HOME=/mnt/nvme-data/engine/ranran/download/hf
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
MEM_PID=""

status() { echo "[$(date +%H:%M:%S)] $*"; echo "- \`$(date +%H:%M:%S)\` $*" >> "$RUN/STATUS.md"; }
die()    { status "FATAL: $*"; exit 1; }

jset() {  # jset FILE KEY VALUE [int|str]
  "$BASE_PY" - "$1" "$2" "$3" "${4:-str}" <<'PY'
import json,sys
p,k,v,t=sys.argv[1:5]
try: d=json.load(open(p))
except Exception: d={}
d[k]=int(v) if t=="int" else v
json.dump(d,open(p,"w"),indent=1)
PY
}

# ---- vLLM lifecycle ----
launch_server() {  # launch_server GPUS DP LOGFILE
  local gpus=$1 dp=$2 logf=$3
  status "launch vLLM: gpus=$gpus dp=$dp port=$PORT -> $(basename "$logf")"
  env -u VLLM_PORT CUDA_VISIBLE_DEVICES="$gpus" HF_HUB_OFFLINE=1 \
    PATH="$VLLM_BIN:$PATH" \
    setsid "$VLLM_PY" "$SPEC/scripts/launch_vllm.py" "$MODEL" \
      --target-layer-ids 2 18 33 -- \
      --data-parallel-size "$dp" --port "$PORT" --gpu-memory-utilization 0.85 \
      > "$logf" 2>&1 < /dev/null &
  echo $! > "$RUN/.server.pid"
}
kill_server() {
  [ -f "$RUN/.server.pid" ] || return 0
  local pid; pid=$(cat "$RUN/.server.pid")
  kill -TERM -"$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -KILL -"$pid" 2>/dev/null || true
  rm -f "$RUN/.server.pid"
}
wait_health() {  # wait_health TIMEOUT_S
  local t=${1:-900} pid=""
  [ -f "$RUN/.server.pid" ] && pid=$(cat "$RUN/.server.pid")
  for _ in $(seq 1 "$t"); do
    curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && { status "server healthy"; return 0; }
    if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
      status "server process exited before healthy (see log)"; return 1
    fi
    sleep 1
  done
  status "server health timeout after ${t}s"; return 1
}
wait_gpu_free() {  # wait_gpu_free "2,3,6,7" TIMEOUT_S
  local gpus=$1 t=${2:-180}
  for _ in $(seq 1 "$((t/2))"); do
    local busy=0 g m
    for g in ${gpus//,/ }; do
      m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
      [ -n "$m" ] && [ "$m" -gt 2000 ] && busy=1
    done
    [ "$busy" -eq 0 ] && return 0
    sleep 2
  done
  return 0
}
cleanup() { kill_server; [ -n "$MEM_PID" ] && kill "$MEM_PID" 2>/dev/null || true; }
trap cleanup EXIT

# ---- meta ----
: > "$RUN/STATUS.md"
status "=== WT-1 calibration start (run=$(basename "$RUN")) ==="
jset "$RUN/env.json" spec_sha "$(git -C "$SPEC" rev-parse --short HEAD 2>/dev/null)"
jset "$RUN/env.json" wt_sha   "$(git -C "$WT"   rev-parse --short HEAD 2>/dev/null)"
TOTAL_CONVS=$(wc -l < "$DATA/train.jsonl")
jset "$RUN/env.json" n_conversations "$([ "$REGEN_LIMIT" -gt 0 ] && echo "$REGEN_LIMIT" || echo "$TOTAL_CONVS")" int
jset "$RUN/env.json" max_samples "$MAX_SAMPLES" int
jset "$RUN/env.json" epochs "$EPOCHS" int
jset "$RUN/env.json" seq_length "$SEQLEN" int
jset "$RUN/env.json" regen_max_tokens "$MAX_TOKENS" int
jset "$RUN/env.json" thinking "$(echo "$SAMPLING" | grep -q 'enable_thinking":true' && echo on || echo off)"
jset "$RUN/env.json" loss_fn '{"ce":0.1,"tv":0.9}'
jset "$RUN/env.json" gpus "regen=$REGEN_GPUS serve=$SERVE_GPUS train=$TRAIN_GPUS"

# ---- phase 0: regen ----
if [ ! -f "$RUN/.regen.done" ]; then
  status "phase0: on-policy regen"
  launch_server "$REGEN_GPUS" 4 "$RUN/vllm_regen.log"
  wait_health 900 || die "regen server not healthy"
  t0=$(date +%s)
  "$BASE_PY" "$WT/runner/regen_wt.py" \
    --regen-script "$SPEC/scripts/response_regeneration/script.py" \
    --data-file "$DATA/train.jsonl" \
    --endpoint "http://127.0.0.1:$PORT/v1/chat/completions" \
    --outfile "$REGEN_OUT" \
    --sampling-params "$SAMPLING" \
    --max-tokens "$MAX_TOKENS" --limit "$REGEN_LIMIT" --concurrency 64 --resume \
    > "$RUN/regen.log" 2>&1
  rc=$?
  jset "$RUN/timing.json" regen_s "$(( $(date +%s) - t0 ))" int
  kill_server; wait_gpu_free "$REGEN_GPUS" 180
  [ $rc -eq 0 ] || die "regen failed rc=$rc (see regen.log)"
  jset "$RUN/env.json" regen_rows "$(wc -l < "$REGEN_OUT")" int
  status "phase0: regen done, rows=$(wc -l < "$REGEN_OUT")"
  touch "$RUN/.regen.done"
else status "phase0: regen already done (skip)"; fi

# ---- phase 1: prepare ----
if [ ! -f "$RUN/.prepare.done" ]; then
  status "phase1: prepare_data -> $PREPARED"
  t0=$(date +%s)
  ( cd "$SPEC" && CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 "$BASE_PY" scripts/prepare_data.py \
      --model "$MODEL" --data "$REGEN_OUT" --output "$PREPARED" \
      --max-samples "$MAX_SAMPLES" --seq-length "$SEQLEN" --overwrite ) \
      > "$RUN/prepare.log" 2>&1
  rc=$?
  jset "$RUN/timing.json" prepare_s "$(( $(date +%s) - t0 ))" int
  [ $rc -eq 0 ] || die "prepare failed rc=$rc (see prepare.log)"
  status "phase1: prepare done"
  touch "$RUN/.prepare.done"
else status "phase1: prepare already done (skip)"; fi

# ---- phase 2: training server ----
if ! curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
  status "phase2: launch training server"
  launch_server "$SERVE_GPUS" 2 "$RUN/vllm_train.log"
  wait_health 900 || die "training server not healthy"
else status "phase2: server already healthy"; fi

# ---- phase 3: lanes ----
train_lane() {
  local lane=$1; shift
  local ldir="$RUN/$lane"; mkdir -p "$ldir"
  if [ -f "$ldir/.train.done" ]; then status "$lane: already done (skip)"; return 0; fi
  printf '{"lane":"%s","epochs":%s,"max_samples":%s,"extra":"%s"}\n' \
    "$lane" "$EPOCHS" "$MAX_SAMPLES" "$*" > "$ldir/config.json"
  status "$lane: train start (epochs=$EPOCHS max_samples=$MAX_SAMPLES)"
  "$BASE_PY" "$WT/runner/mem_sampler.py" --gpus "$TRAIN_GPUS" --interval 2 --out "$ldir/mem.jsonl" &
  MEM_PID=$!
  local t0; t0=$(date +%s)
  ( cd "$SPEC" && CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" HF_HUB_OFFLINE=1 \
    "$BASE_PY" -m torch.distributed.run --standalone --nproc_per_node 2 scripts/train.py \
      --verifier-name-or-path "$MODEL" \
      --data-path "$PREPARED" \
      --vllm-endpoint "http://localhost:$PORT/v1" \
      --save-path "$ldir/checkpoints" \
      --draft-vocab-size 32000 \
      --epochs "$EPOCHS" \
      --loss-fn '{"ce": 0.1, "tv": 0.9}' \
      --optimizer muon \
      --total-seq-len "$SEQLEN" \
      --logger tensorboard --log-dir "$ldir/tb" --log-freq 10 \
      --on-missing generate --on-generate delete \
      "$@" ) > "$ldir/train.log" 2>&1
  local rc=$?
  jset "$RUN/timing.json" "${lane}_train_s" "$(( $(date +%s) - t0 ))" int
  kill "$MEM_PID" 2>/dev/null || true; MEM_PID=""
  if [ $rc -eq 0 ]; then touch "$ldir/.train.done"; status "$lane: done rc=0";
  else status "$lane: FAILED rc=$rc (continuing to next lane; see $lane/train.log)"; fi
}

train_lane eagle3 --lr 1e-4
train_lane dflash --lr 3e-4 --speculator-type dflash --block-size 8 \
  --max-anchors 3072 --num-layers 5 --target-layer-ids 2 18 33
train_lane dspark --lr 3e-4 --speculator-type dspark --block-size 8 \
  --max-anchors 3072 --num-layers 5 --target-layer-ids 2 18 33 \
  --markov-rank 256 --markov-head-type vanilla --confidence-head-alpha 1.0

kill_server; wait_gpu_free "$SERVE_GPUS" 120

# ---- phase 4: collect ----
status "phase4: collect metrics"
"$BASE_PY" "$WT/runner/collect.py" --run "$RUN" --lanes eagle3,dflash,dspark \
  > "$RUN/collect.log" 2>&1 || status "collect had errors (see collect.log)"
status "=== calibration COMPLETE — see SUMMARY.md ==="
