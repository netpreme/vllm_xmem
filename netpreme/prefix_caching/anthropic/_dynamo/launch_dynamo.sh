#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
#  Launch the Dynamo + vLLM inference stack for Claude Code.
#
#  Architecture (no proxy, no translation layer):
#
#    Claude Code
#      │  POST /v1/messages   (native Anthropic API format)
#      │  ANTHROPIC_BASE_URL=http://localhost:8000
#      ▼
#    dynamo.frontend  (:8000)
#      │  --enable-anthropic-api: accepts /v1/messages natively,
#      │    no LiteLLM or format conversion needed.
#      │  --discovery-backend file: worker is found via a local
#      │    file on disk (no NATS/etcd required for single-node).
#      │  --dyn-chat-processor dynamo (default): the Rust frontend
#      │    applies the chat template and tokenizes requests before
#      │    forwarding token IDs to the worker.
#      │  Internal TCP transport to worker (not HTTP).
#      ▼
#    dynamo.vllm  (GPUs 6+7, internal TCP port assigned by OS)
#      │  --tensor-parallel-size 2: model sharded across 2x H100.
#      │  --enable-prefix-caching: vLLM caches KV blocks by prefix
#      │    hash so repeated prefixes (system prompt, conversation
#      │    history) skip recomputation on subsequent turns.
#      │  --kv-events-config enable=false: disables cross-worker KV
#      │    block event publishing (only relevant for multi-worker
#      │    setups; single worker doesn't need it).
#      │  PYTHONHASHSEED=0: makes Python dict ordering deterministic,
#      │    which helps prefix hash stability across restarts.
#      ▼
#    vLLM EngineCore (subprocess, owns the GPU memory)
#      KV cache: 18562 blocks × 16 tokens = ~297K token capacity
#
#  Metrics:  http://localhost:8000/metrics
#    dynamo_frontend_input_sequence_tokens_sum  — total input tokens
#    dynamo_frontend_cached_tokens_sum          — tokens served from
#      KV cache (sourced from vLLM's num_cached_tokens per request)
#    dynamo_frontend_model_total_kv_blocks      — GPU block count
#
#
#  Usage:  ./launch_dynamo.sh           # CPU tier (pinned RAM)
#          ./launch_dynamo.sh --mtier   # xmem chip tier
#  Stop:   Ctrl-C
#  Logs:   tail -f /tmp/dynamo_worker_<PORT>.log /tmp/dynamo_frontend_<PORT>.log
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
for arg in "$@"; do
    case "$arg" in
        --mtier) USE_MTIER=1 ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

MODEL="${MODEL:-Qwen/Qwen2.5-Coder-32B-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${DYNAMO_PORT:-8000}"
TP="${TENSOR_PARALLEL_SIZE:-1}"
DTYPE="${DTYPE:-auto}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEM_UTIL="${GPU_MEMORY_UTILIZATION:-0.92}"

KV_TIER="cpu"
MTIER_FLAG=""
KV_OFFLOAD_SIZE="72"
if [[ "$USE_MTIER" == "1" ]]; then
    KV_TIER="chip"
    MTIER_FLAG="--kv-offloading-mtier"
    KV_OFFLOAD_SIZE="68"   # chip has 77.3 GB total; 68 -> ~73 GB actual, 4 GB headroom
fi

# Use CUDA_VISIBLE_DEVICES from environment or .env; default to 0 if unset
CVD="${CUDA_VISIBLE_DEVICES:-0}"

# Isolate per-instance paths using the port number so two stacks don't collide
STORE_PATH="${DYNAMO_STORE_PATH:-/tmp/dynamo_store_kv_${PORT}}"
WORKER_LOG="${DYNAMO_WORKER_LOG:-/tmp/dynamo_worker_${PORT}.log}"
FRONTEND_LOG="${DYNAMO_FRONTEND_LOG:-/tmp/dynamo_frontend_${PORT}.log}"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Model   : $MODEL"
echo "  Host    : $HOST"
echo "  Port    : $PORT  (/v1/messages + /v1/chat/completions)"
echo "  GPUs    : $CVD"
echo "  TP      : $TP"
echo "  Max len : $MAX_MODEL_LEN"
echo "  KV tier : $KV_TIER"
echo "  Store   : $STORE_PATH"
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
WORKER_PID=""
FRONTEND_PID=""
TAIL_PID=""

cleanup() {
    echo ""
    echo "Stopping Dynamo..."
    [[ -n "$TAIL_PID" ]] && kill $TAIL_PID 2>/dev/null
    [[ -n "$FRONTEND_PID" ]] && kill $FRONTEND_PID 2>/dev/null
    [[ -n "$WORKER_PID" ]] && kill $WORKER_PID 2>/dev/null
    wait 2>/dev/null
    echo "Done."
    exit 0
}
trap cleanup INT TERM

# ── Step 1: vLLM worker ─────────────────────────────────────────
echo "[1/3] Starting vLLM worker on GPUs $CVD ..."
CUDA_VISIBLE_DEVICES="$CVD" \
PYTHONHASHSEED=0 \
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
HF_HUB_OFFLINE=1 \
DYN_FILE_KV="$STORE_PATH" \
    "$PYTHON_BIN" -m dynamo.vllm \
        --model "$MODEL" \
        --served-model-name "$MODEL" \
        --tensor-parallel-size "$TP" \
        --dtype "$DTYPE" \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --enable-prefix-caching \
        --kv-offloading-size "$KV_OFFLOAD_SIZE" \
        ${MTIER_FLAG:+$MTIER_FLAG} \
        --discovery-backend file \
        --kv-events-config '{"enable_kv_cache_events": false}' \
        > "$WORKER_LOG" 2>&1 &
WORKER_PID=$!
echo "      PID $WORKER_PID  (log: $WORKER_LOG)"

# ── Step 2: Dynamo frontend with Anthropic API ──────────────────
echo "[2/3] Starting Dynamo frontend on :$PORT ..."
CUDA_VISIBLE_DEVICES="$CVD" \
DYN_FILE_KV="$STORE_PATH" \
"$PYTHON_BIN" -m dynamo.frontend \
    --http-port "$PORT" \
    --http-host "$HOST" \
    --discovery-backend file \
    --enable-anthropic-api \
    > "$FRONTEND_LOG" 2>&1 &
FRONTEND_PID=$!
echo "      PID $FRONTEND_PID  (log: $FRONTEND_LOG)"

# ── Step 3: Wait for health ──────────────────────────────────────
echo ""
echo "[3/3] Waiting for stack to be ready (model load takes ~2-3 min)..."
until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do
    if ! kill -0 $WORKER_PID 2>/dev/null; then
        echo ""
        echo "ERROR: Worker died. Check: tail $WORKER_LOG"
        cleanup
    fi
    if ! kill -0 $FRONTEND_PID 2>/dev/null; then
        echo ""
        echo "ERROR: Frontend died. Check: tail $FRONTEND_LOG"
        cleanup
    fi
    printf "."
    sleep 5
done
echo ""
echo "✅ Dynamo + vLLM ready!"
echo ""
echo "  Claude Code:  ANTHROPIC_BASE_URL=http://localhost:$PORT"
echo "  Metrics:      http://localhost:$PORT/metrics"
echo "  PIDs:         worker=$WORKER_PID  frontend=$FRONTEND_PID"
echo "  Ctrl-C to stop."
echo ""

# ── Tail logs in foreground ──────────────────────────────────────
tail -f "$WORKER_LOG" "$FRONTEND_LOG" &
TAIL_PID=$!

while true; do
    if ! kill -0 $WORKER_PID 2>/dev/null; then
        echo ""
        echo "Worker process ($WORKER_PID) exited. Cleaning up..."
        break
    fi
    if ! kill -0 $FRONTEND_PID 2>/dev/null; then
        echo ""
        echo "Frontend process ($FRONTEND_PID) exited. Cleaning up..."
        break
    fi
    sleep 5
done
cleanup
