#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
#  PHASE 1: Dynamo + vLLM (OpenAI format)
#  Run this on the GPU machine in your Claude Code terminal.
#  Uses your existing .env — no changes needed.
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

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
PORT="${PORT:-8000}"
TP="${TENSOR_PARALLEL_SIZE:-2}"
DTYPE="${DTYPE:-auto}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
GPU_MEM_UTIL="${GPU_MEMORY_UTILIZATION:-0.90}"
CVD="${CUDA_VISIBLE_DEVICES:-6,7}"

# ── Step 0: Install Dynamo if not already installed ──────────────
if ! python3 -c "import dynamo" 2>/dev/null; then
    echo "[0/3] Installing ai-dynamo[vllm]..."
    pip install "ai-dynamo[vllm]" -q
else
    echo "[0/3] dynamo already installed, skipping."
fi

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Model   : $MODEL"
echo "  Port    : $PORT  (OpenAI format)"
echo "  GPUs    : $CVD"
echo "  TP      : $TP"
echo "  Max len : $MAX_MODEL_LEN"
echo "═══════════════════════════════════════════════════════"
echo ""

# ── Cleanup function ─────────────────────────────────────────────
cleanup() {
    echo ""
    echo "Stopping Dynamo..."
    kill $FRONTEND_PID $WORKER_PID 2>/dev/null
    wait $FRONTEND_PID $WORKER_PID 2>/dev/null
    echo "Done."
    exit 0
}
trap cleanup INT TERM

# ── Step 1: vLLM worker on GPUs (starts first, takes longer to load) ──
echo "[1/3] Starting vLLM worker on GPUs $CVD ..."
CUDA_VISIBLE_DEVICES="$CVD" \
PYTHONHASHSEED=0 \
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    python3 -m dynamo.vllm \
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

# ── Step 2: Dynamo frontend (OpenAI-compatible, CPU only) ────────
echo "[2/3] Starting Dynamo frontend on :$PORT ..."
python3 -m dynamo.frontend \
    --http-port "$PORT" \
    --discovery-backend file \
    > /tmp/dynamo_frontend.log 2>&1 &
FRONTEND_PID=$!
echo "      PID $FRONTEND_PID  (log: /tmp/dynamo_frontend.log)"

# ── Step 3: Wait for health ──────────────────────────────────────
echo ""
echo "[3/3] Waiting for stack to be ready (model load takes ~2-3 min)..."
until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do
    # Check if either process died
    if ! kill -0 $WORKER_PID 2>/dev/null; then
        echo ""
        echo "ERROR: Worker died. Check: tail /tmp/dynamo_worker.log"
        kill $FRONTEND_PID 2>/dev/null
        exit 1
    fi
    if ! kill -0 $FRONTEND_PID 2>/dev/null; then
        echo ""
        echo "ERROR: Frontend died. Check: tail /tmp/dynamo_frontend.log"
        kill $WORKER_PID 2>/dev/null
        exit 1
    fi
    printf "."
    sleep 5
done
echo ""
echo "✅ Dynamo + vLLM ready!"
echo ""
echo "PIDs:  frontend=$FRONTEND_PID  worker=$WORKER_PID"
echo "Ctrl-C to stop both."
echo ""

# ── Tail both logs in foreground ─────────────────────────────────
tail -f /tmp/dynamo_frontend.log /tmp/dynamo_worker.log &
TAIL_PID=$!

# Wait for either process to exit
wait -n $FRONTEND_PID $WORKER_PID 2>/dev/null || true
echo ""
echo "A process exited. Cleaning up..."
kill $TAIL_PID 2>/dev/null
cleanup