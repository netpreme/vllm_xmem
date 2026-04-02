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
#  --dyn-chat-processor dynamo (default, Rust tokenizer): works with
#    this vLLM version. Prefix cache hit rate is near 0% because the
#    Rust tokenizer produces non-matching block hashes across turns.
#    NOTE: --dyn-chat-processor vllm (Python tokenizer, would fix
#    cache hits) is broken against the installed vLLM — two API
#    mismatches: eos_token_id read-only property, and
#    InputProcessor.process_inputs() missing supported_tasks arg.
#
#  Usage:  ./launch_dynamo.sh
#  Stop:   Ctrl-C
#  Logs:   tail -f /tmp/dynamo_worker.log /tmp/dynamo_frontend.log
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
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

MODEL="${MODEL:-Qwen/Qwen2.5-Coder-32B-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${DYNAMO_PORT:-8000}"
TP="${TENSOR_PARALLEL_SIZE:-2}"
DTYPE="${DTYPE:-auto}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
GPU_MEM_UTIL="${GPU_MEMORY_UTILIZATION:-0.90}"
CVD="${CUDA_VISIBLE_DEVICES:-6,7}"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Model   : $MODEL"
echo "  Host    : $HOST"
echo "  Port    : $PORT  (/v1/messages + /v1/chat/completions)"
echo "  GPUs    : $CVD"
echo "  TP      : $TP"
echo "  Max len : $MAX_MODEL_LEN"
echo "═══════════════════════════════════════════════════════"
echo ""

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
    "$PYTHON_BIN" -m dynamo.vllm \
        --model "$MODEL" \
        --served-model-name "$MODEL" \
        --tensor-parallel-size "$TP" \
        --dtype "$DTYPE" \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --enable-prefix-caching \
        --discovery-backend file \
        --kv-events-config '{"enable_kv_cache_events": false}' \
        > /tmp/dynamo_worker.log 2>&1 &
WORKER_PID=$!
echo "      PID $WORKER_PID  (log: /tmp/dynamo_worker.log)"

# ── Step 2: Dynamo frontend with Anthropic API ──────────────────
echo "[2/3] Starting Dynamo frontend on :$PORT ..."
"$PYTHON_BIN" -m dynamo.frontend \
    --http-port "$PORT" \
    --http-host "$HOST" \
    --discovery-backend file \
    --enable-anthropic-api \
    > /tmp/dynamo_frontend.log 2>&1 &
FRONTEND_PID=$!
echo "      PID $FRONTEND_PID  (log: /tmp/dynamo_frontend.log)"

# ── Step 3: Wait for health ──────────────────────────────────────
echo ""
echo "[3/3] Waiting for stack to be ready (model load takes ~2-3 min)..."
until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do
    if ! kill -0 $WORKER_PID 2>/dev/null; then
        echo ""
        echo "ERROR: Worker died. Check: tail /tmp/dynamo_worker.log"
        cleanup
    fi
    if ! kill -0 $FRONTEND_PID 2>/dev/null; then
        echo ""
        echo "ERROR: Frontend died. Check: tail /tmp/dynamo_frontend.log"
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
tail -f /tmp/dynamo_worker.log /tmp/dynamo_frontend.log &
TAIL_PID=$!

wait -n $WORKER_PID $FRONTEND_PID 2>/dev/null || true
echo ""
echo "A process exited. Cleaning up..."
cleanup
