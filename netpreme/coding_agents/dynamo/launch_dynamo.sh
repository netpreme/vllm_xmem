#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
#  Launch the Dynamo + vLLM inference stack for Claude Code.
#
#  Architecture (shared KV, 2 workers × TP=2):
#
#    Claude Code
#      │  POST /v1/messages   (native Anthropic API format)
#      │  ANTHROPIC_BASE_URL=http://localhost:8000
#      ▼
#    dynamo.frontend  (:8000)
#      │  --enable-anthropic-api: accepts /v1/messages natively
#      │  --discovery-backend nats: workers registered via NATS
#      │  --router-mode kv: routes each request to the worker
#      │    with the most prefix-cache overlap (uses KV events)
#      │  Internal TCP transport to workers (not HTTP)
#      ▼
#    NATS  (:4222)  — KV event pub/sub backbone
#      ▼
#    dynamo.vllm worker 0  (GPUs 0,1 — TP=2, DYN_SYSTEM_PORT=8081)
#    dynamo.vllm worker 1  (GPUs 2,3 — TP=2, DYN_SYSTEM_PORT=8082)
#      │  --tensor-parallel-size 2: model sharded across 2x H100
#      │  --enable-prefix-caching: per-worker prefix hash caching
#      │  --kv-events-config enable=true: publishes KV block events
#      │    to NATS so the router knows what each worker has cached
#      │  PYTHONHASHSEED=0: deterministic dict ordering for stable
#      │    prefix hashes across restarts
#      ▼
#    vLLM EngineCore (subprocess, owns the GPU memory)
#
#  Shared KV mechanics:
#    Each worker publishes KV block events via NATS.  The router
#    subscribes and maintains a per-worker KV index.  New requests
#    are sent to the worker with the highest prefix-cache overlap,
#    minimising recomputed tokens and lowering TTFT.
#
#  Metrics:
#    http://localhost:8000/metrics  — frontend + router
#    http://localhost:8081/metrics  — worker 0 (vLLM + dynamo)
#    http://localhost:8082/metrics  — worker 1 (vLLM + dynamo)
#
#  Usage:  ./launch_dynamo.sh                         # hybrid CPU offload (default)
#          ./launch_dynamo.sh --no-offload            # all KV stays in HBM
#          ./launch_dynamo.sh --force-offload         # always evict → CPU
#          ./launch_dynamo.sh --mtier                 # hybrid xmem chip offload
#          ./launch_dynamo.sh --mtier --force-offload # always evict → xmem chip
#  Stop:   Ctrl-C
#  Env:    NUM_WORKERS=2, TENSOR_PARALLEL_SIZE=2, CUDA_VISIBLE_DEVICES=0,1,2,3
#          FORCE_OFFLOAD_GPU_BLOCKS=256  — GPU KV block cap for --force-offload
#  Logs:   tail -f /tmp/dynamo_worker_{8081,8082}.log /tmp/dynamo_frontend_8000.log
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="$(command -v python3)"
fi

if [[ -z "$PYTHON_BIN" ]]; then
    echo "ERROR: No Python interpreter found." >&2
    exit 1
fi

# ── Load .env ────────────────────────────────────────────────────
if [[ -f "$ENV_FILE" ]]; then
    while IFS='=' read -r key value; do
        [[ -z "$key" || "$key" == \#* ]] && continue
        [[ -v "$key" ]] || export "$key=$value"
    done < "$ENV_FILE"
else
    echo "WARNING: .env not found, using built-in defaults." >&2
fi

# ── Parse flags ─────────────────────────────────────────────────
USE_MTIER=0
FORCE_OFFLOAD=0
NO_OFFLOAD=0
for arg in "$@"; do
    case "$arg" in
        --mtier)         USE_MTIER=1 ;;
        --force-offload) FORCE_OFFLOAD=1 ;;
        --no-offload)    NO_OFFLOAD=1 ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

MODEL="${MODEL:-Qwen/Qwen2.5-Coder-32B-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${DYNAMO_PORT:-8000}"
TP="${TENSOR_PARALLEL_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-2}"
DTYPE="${DTYPE:-auto}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEM_UTIL="${GPU_MEMORY_UTILIZATION:-0.92}"
NATS_PORT="${NATS_PORT:-4222}"
NATS_SERVER_URL="nats://localhost:${NATS_PORT}"

KV_TIER="cpu"
MTIER_FLAG=""
KV_OFFLOAD_SIZE="72"
KV_OFFLOAD_FLAG="--kv-offloading-size"
if [[ "$USE_MTIER" == "1" ]]; then
    KV_TIER="chip"
    MTIER_FLAG="--kv-offloading-mtier"
    KV_OFFLOAD_SIZE="68"   # chip has 77.3 GB total; 68 -> ~73 GB actual, 4 GB headroom
fi

# --no-offload: disable KV offloading entirely — all KV blocks stay in HBM
if [[ "$NO_OFFLOAD" == "1" ]]; then
    KV_TIER="hbm-only"
    KV_OFFLOAD_FLAG=""
    KV_OFFLOAD_SIZE=""
    MTIER_FLAG=""
fi

# --force-offload: cap GPU KV blocks so the GPU cache fills immediately,
# forcing every block through the evict→CPU/MTier→fetch cycle.
FORCE_OFFLOAD_FLAG=""
FORCE_OFFLOAD_GPU_BLOCKS="${FORCE_OFFLOAD_GPU_BLOCKS:-256}"
if [[ "$FORCE_OFFLOAD" == "1" ]]; then
    FORCE_OFFLOAD_FLAG="--num-gpu-blocks-override $FORCE_OFFLOAD_GPU_BLOCKS"
    KV_TIER="${KV_TIER}+forced-evict(${FORCE_OFFLOAD_GPU_BLOCKS} GPU blocks)"
fi

# ── GPU assignment: split CVD_ALL evenly across NUM_WORKERS ─────
CVD_ALL="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -ra ALL_GPUS <<< "$CVD_ALL"
GPUS_PER_WORKER=$(( ${#ALL_GPUS[@]} / NUM_WORKERS ))

declare -a WORKER_CVD
for (( w=0; w<NUM_WORKERS; w++ )); do
    start=$(( w * GPUS_PER_WORKER ))
    cvd=""
    for (( g=0; g<GPUS_PER_WORKER; g++ )); do
        [[ -n "$cvd" ]] && cvd+=","
        cvd+="${ALL_GPUS[$(( start + g ))]}"
    done
    WORKER_CVD[$w]="$cvd"
done

FRONTEND_LOG="${DYNAMO_FRONTEND_LOG:-/tmp/dynamo_frontend_${PORT}.log}"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Model       : $MODEL"
echo "  Host        : $HOST"
echo "  Port        : $PORT  (/v1/messages + /v1/chat/completions)"
echo "  GPUs        : $CVD_ALL  (${NUM_WORKERS} workers × TP=${TP})"
echo "  Max len     : $MAX_MODEL_LEN"
echo "  KV tier     : $KV_TIER"
echo "  Shared KV   : enabled (NATS KV events → router-mode kv)"
echo "  NATS        : $NATS_SERVER_URL"
echo "═══════════════════════════════════════════════════════"
echo ""

# ── Kill any stale stack already using our port ─────────────────
if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
    echo "WARNING: port $PORT already in use — killing stale dynamo/vllm processes..."
    pkill -9 -f "dynamo.vllm" 2>/dev/null || true
    pkill -9 -f "dynamo.frontend" 2>/dev/null || true
    sleep 3
fi

# ── Cleanup function ─────────────────────────────────────────────
WORKER_PIDS=()
FRONTEND_PID=""
TAIL_PID=""
NATS_PID=""

cleanup() {
    echo ""
    echo "Stopping Dynamo..."
    [[ -n "$TAIL_PID"     ]] && kill "$TAIL_PID"     2>/dev/null || true
    [[ -n "$FRONTEND_PID" ]] && kill "$FRONTEND_PID" 2>/dev/null || true
    for pid in "${WORKER_PIDS[@]+"${WORKER_PIDS[@]}"}"; do
        kill "$pid" 2>/dev/null || true
    done
    [[ -n "$NATS_PID" ]] && kill "$NATS_PID" 2>/dev/null || true
    wait 2>/dev/null || true

    echo "Freeing GPU memory on device(s): $CVD_ALL ..."
    for gpu_id in ${CVD_ALL//,/ }; do
        nvidia-smi --query-compute-apps=pid --format=csv,noheader --id="$gpu_id" 2>/dev/null \
            | xargs -r kill -9 2>/dev/null || true
    done

    if [[ "$USE_MTIER" == "1" ]]; then
        echo "Resetting MTier memory..."
        echo "yes" | mtier_service reset 2>/dev/null || true
    fi

    echo "Done."
    exit 0
}
trap cleanup INT TERM

# ── Step 0: NATS server (KV event pub/sub backbone) ─────────────
echo "[0/4] Starting NATS server on :${NATS_PORT} ..."
if ! command -v nats-server &>/dev/null; then
    echo "ERROR: nats-server not found. Install: https://nats.io/download/" >&2
    exit 1
fi
nats-server -p "$NATS_PORT" > /tmp/nats_${NATS_PORT}.log 2>&1 &
NATS_PID=$!
echo "      PID $NATS_PID  (log: /tmp/nats_${NATS_PORT}.log)"
sleep 1   # brief pause for NATS to bind its port

# ── Step 1: vLLM workers ────────────────────────────────────────
echo "[1/4] Starting ${NUM_WORKERS} vLLM workers (TP=${TP} each) ..."
WORKER_LOG_FILES=()
for (( w=0; w<NUM_WORKERS; w++ )); do
    WPORT=$(( 8081 + w ))
    WCVD="${WORKER_CVD[$w]}"
    WLOG="/tmp/dynamo_worker_${WPORT}.log"
    WSTORE="${DYNAMO_STORE_PATH:-/tmp/dynamo_store_kv_${WPORT}}"
    WORKER_LOG_FILES+=("$WLOG")

    CUDA_VISIBLE_DEVICES="$WCVD" \
    PYTHONHASHSEED=0 \
    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    HF_HUB_OFFLINE=1 \
    NATS_SERVER="$NATS_SERVER_URL" \
    DYN_SYSTEM_PORT="$WPORT" \
    DYN_FILE_KV="$WSTORE" \
        "$PYTHON_BIN" -m dynamo.vllm \
            --model "$MODEL" \
            --served-model-name "$MODEL" \
            --tensor-parallel-size "$TP" \
            --dtype "$DTYPE" \
            --max-model-len "$MAX_MODEL_LEN" \
            --gpu-memory-utilization "$GPU_MEM_UTIL" \
            --enable-prefix-caching \
            ${KV_OFFLOAD_FLAG:+$KV_OFFLOAD_FLAG "$KV_OFFLOAD_SIZE"} \
            ${MTIER_FLAG:+$MTIER_FLAG} \
            ${FORCE_OFFLOAD_FLAG:+$FORCE_OFFLOAD_FLAG} \
            --discovery-backend nats \
            --kv-events-config '{"enable_kv_cache_events": true}' \
            > "$WLOG" 2>&1 &
    WORKER_PIDS+=($!)
    echo "      Worker ${w}: GPUs ${WCVD}, DYN_PORT ${WPORT}, PID ${WORKER_PIDS[-1]}  (log: ${WLOG})"
done

# ── Step 2: Dynamo frontend with Anthropic API ──────────────────
echo "[2/4] Starting Dynamo frontend on :$PORT ..."
NATS_SERVER="$NATS_SERVER_URL" \
"$PYTHON_BIN" -m dynamo.frontend \
    --http-port "$PORT" \
    --http-host "$HOST" \
    --discovery-backend nats \
    --enable-anthropic-api \
    --dyn-chat-processor vllm \
    --router-mode kv \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    > "$FRONTEND_LOG" 2>&1 &
FRONTEND_PID=$!
echo "      PID $FRONTEND_PID  (log: $FRONTEND_LOG)"

# ── Step 3: Wait for health ──────────────────────────────────────
echo ""
echo "[3/4] Waiting for stack to be ready (model load takes ~2-3 min)..."
until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do
    for (( w=0; w<NUM_WORKERS; w++ )); do
        if ! kill -0 "${WORKER_PIDS[$w]}" 2>/dev/null; then
            echo ""
            echo "ERROR: Worker ${w} died. Check: tail ${WORKER_LOG_FILES[$w]}"
            cleanup
        fi
    done
    if ! kill -0 $FRONTEND_PID 2>/dev/null; then
        echo ""
        echo "ERROR: Frontend died. Check: tail $FRONTEND_LOG"
        cleanup
    fi
    printf "."
    sleep 5
done
echo ""
echo "✅ Dynamo + vLLM ready! (${NUM_WORKERS} workers, shared KV via NATS)"
echo ""
echo "  Claude Code:  ANTHROPIC_BASE_URL=http://localhost:$PORT"
echo "  Metrics:      http://localhost:$PORT/metrics"
for (( w=0; w<NUM_WORKERS; w++ )); do
    echo "  Worker ${w}:     http://localhost:$(( 8081 + w ))/metrics  (GPUs ${WORKER_CVD[$w]})"
done
echo "  Ctrl-C to stop."
echo ""

# ── Step 4: Tail logs in foreground ─────────────────────────────
tail -f "${WORKER_LOG_FILES[@]}" "$FRONTEND_LOG" &
TAIL_PID=$!

while true; do
    for (( w=0; w<NUM_WORKERS; w++ )); do
        if ! kill -0 "${WORKER_PIDS[$w]}" 2>/dev/null; then
            echo ""
            echo "Worker ${w} (${WORKER_PIDS[$w]}) exited. Cleaning up..."
            cleanup
        fi
    done
    if ! kill -0 $FRONTEND_PID 2>/dev/null; then
        echo ""
        echo "Frontend process ($FRONTEND_PID) exited. Cleaning up..."
        cleanup
    fi
    sleep 5
done
cleanup
