#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
#  Launch vLLM for Claude Code — no Dynamo required.
#
#  Architecture:
#
#    Claude Code
#      │  POST /v1/messages   (native Anthropic API format)
#      │  ANTHROPIC_BASE_URL=http://localhost:8000
#      ▼
#    vLLM Server (:8000)
#      │  vllm.entrypoints.openai.api_server
#      │  --served-model-name "$MODEL" (open-source model only)
#      ▼
#    vLLM EngineCore (owns GPU memory)
#      KV offloading: OffloadingConnector → CPU DRAM or MTier chip
#
# ───────────────────────────────────────────────────────────────
#  KV Cache Modes
# ───────────────────────────────────────────────────────────────
#
#  (default) --hybrid-cpu
#    GPU HBM prefix cache + CPU DRAM overflow.
#    Hot blocks in GPU; evicted blocks spill to CPU DRAM.
#    Next turn: GPU hit first, CPU fallback for anything evicted.
#    Flags: --enable-prefix-caching + OffloadingConnector (CPU)
#
#  --hbm-only
#    All KV stays in GPU HBM. No offloading.
#    Flags: --enable-prefix-caching, no OffloadingConnector.
#
#  --cpu-only
#    GPU prefix caching disabled. OffloadingConnector stores ALL
#    blocks to CPU DRAM after every request and reloads them on
#    the next turn. GPU is compute-only; CPU is the KV store.
#    Flags: --no-enable-prefix-caching + OffloadingConnector (CPU)
#
#  --mtier-only
#    Same as --cpu-only but offload target is the MTier chip
#    instead of CPU DRAM.
#    Flags: --no-enable-prefix-caching + OffloadingConnector (MTier)
#
#  --hybrid-mtier
#    GPU HBM prefix cache + MTier chip overflow.
#    Hot blocks in GPU; evicted blocks spill to MTier.
#    Flags: --enable-prefix-caching + OffloadingConnector (MTier)
#
# ───────────────────────────────────────────────────────────────
#  Metrics: http://localhost:8000/metrics
#
#    vllm:request_prompt_tokens_{sum,count}       ISL per request
#    vllm:request_generation_tokens_{sum,count}   OSL per request
#    vllm:time_to_first_token_seconds_{sum,count} TTFT histogram
#    vllm:inter_token_latency_seconds_{sum,count} ITL histogram
#    vllm:e2e_request_latency_seconds_{sum,count} E2E histogram
#    vllm:prefix_cache_hits_total                 GPU (HBM) cache hits
#    vllm:prefix_cache_queries_total              GPU cache queries
#    vllm:external_prefix_cache_hits_total        CPU/MTier cache hits
#    vllm:external_prefix_cache_queries_total     CPU/MTier queries
#    vllm:kv_offload_total_bytes{transfer_type="gpu_to_cpu"}  G→C bytes
#    vllm:kv_offload_total_bytes{transfer_type="cpu_to_gpu"}  C→G bytes
#
#  Monitor live:
#    python3 netpreme/coding_agents/monitoring/watch_vllm.py
#
# ───────────────────────────────────────────────────────────────
#  Usage:
#    ./start_vllm_server.sh                          # hybrid-cpu (default)
#    ./start_vllm_server.sh --hbm-only               # GPU HBM only, no offload
#    ./start_vllm_server.sh --cpu-only               # CPU DRAM only (no GPU cache)
#    ./start_vllm_server.sh --mtier-only             # MTier chip only (no GPU cache)
#    ./start_vllm_server.sh --hybrid-cpu             # GPU HBM + CPU DRAM overflow
#    ./start_vllm_server.sh --hybrid-mtier           # GPU HBM + MTier chip overflow
#    ./start_vllm_server.sh --gpu-util 0.75          # override GPU memory utilization
#    ./start_vllm_server.sh --hybrid-cpu --gpu-util 0.60  # combine mode + util
#    ./start_vllm_server.sh --port 8001                   # run on a different port
#    ./start_vllm_server.sh --hybrid-cpu --port 8001      # combine mode + port
#
#  Connect Claude Code (separate terminal after server is ready):
#    ./run_claude_local.sh
#
#  Stop:   Ctrl-C
#  Log:    tail -f /tmp/vllm_server_<PORT>.log
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# .env lives one level up (next to launch_dynamo.sh)
ENV_FILE="${SCRIPT_DIR}/../.env"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="$(command -v python3 2>/dev/null || true)"
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
    echo "WARNING: .env not found at $ENV_FILE, using built-in defaults." >&2
fi

# ── Parse mode ───────────────────────────────────────────────────
MODE="hybrid-cpu"
GPU_UTIL_OVERRIDE=""
PORT_OVERRIDE=""
MAX_NUM_SEQS_OVERRIDE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --hbm-only)     MODE="hbm-only" ;;
        --cpu-only)     MODE="cpu-only" ;;
        --mtier-only)   MODE="mtier-only" ;;
        --hybrid-cpu)   MODE="hybrid-cpu" ;;
        --hybrid-mtier) MODE="hybrid-mtier" ;;
        --gpu-util)
            shift
            GPU_UTIL_OVERRIDE="$1"
            ;;
        --gpu-util=*)
            GPU_UTIL_OVERRIDE="${1#--gpu-util=}"
            ;;
        --port)
            shift
            PORT_OVERRIDE="$1"
            ;;
        --port=*)
            PORT_OVERRIDE="${1#--port=}"
            ;;
        --max-num-seqs)
            shift
            MAX_NUM_SEQS_OVERRIDE="$1"
            ;;
        --max-num-seqs=*)
            MAX_NUM_SEQS_OVERRIDE="${1#--max-num-seqs=}"
            ;;
        *) echo "ERROR: Unknown argument: $1" >&2
           echo "       Valid modes: --hbm-only | --cpu-only | --mtier-only | --hybrid-cpu | --hybrid-mtier" >&2
           echo "       Options:     --gpu-util <0.0-1.0>  --port <number>  --max-num-seqs <N>" >&2
           exit 1 ;;
    esac
    shift
done

# ── Core settings ────────────────────────────────────────────────
MODEL="${MODEL:-qwen/qwen3-coder-30b-a3b-instruct-fp8}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT_OVERRIDE:-${PORT:-8000}}"
TP="${TENSOR_PARALLEL_SIZE:-1}"
DTYPE="${DTYPE:-auto}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
# fp8 KV cache halves per-token KV memory (131 KB vs 262 KB in bf16), allowing
# the startup check to pass with 9.33 GiB GPU KV for up to ~76K tokens.
# At MAX_MODEL_LEN=65536 the check needs 8.0 GiB < 9.33 GiB available.
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
GPU_MEM_UTIL="${GPU_UTIL_OVERRIDE:-${GPU_MEMORY_UTILIZATION:-0.75}}"
MAX_NUM_SEQS="${MAX_NUM_SEQS_OVERRIDE:-${MAX_NUM_SEQS:-256}}"
NUM_GPU_BLOCKS_OVERRIDE="${NUM_GPU_BLOCKS_OVERRIDE:-}"   # empty = let vLLM decide

# Use CUDA_VISIBLE_DEVICES from env or .env; default to 0 if unset
CVD="${CUDA_VISIBLE_DEVICES:-0,1}"

VLLM_LOG="${VLLM_LOG:-/tmp/vllm_server_${PORT}.log}"

# ── KV cache config by mode ──────────────────────────────────────
#
# Byte sizes:
#   72 GiB = 77,309,411,328 bytes  (CPU DRAM)
#   68 GiB = 73,014,444,032 bytes  (MTier bank 0: 70 GB capacity, 2 GiB headroom)
#   NOTE: xmem MTier allocates from bank_id=0 only (cpu_gpu.py:484). The hardware
#   has 4×70 GB banks but only bank 0 is usable until multi-bank support is added.
#
PREFIX_CACHE_FLAG=""
KV_OFFLOAD_FLAG=""
KV_OFFLOAD_SIZE=""
KV_TRANSFER_CONFIG=""
MTIER_FLAG=""
DISABLE_HMA_FLAG=""

case "$MODE" in
    hbm-only)
        # GPU HBM prefix cache only. No OffloadingConnector.
        # HMA left enabled — no connector conflict.
        PREFIX_CACHE_FLAG="--enable-prefix-caching"
        ;;

    cpu-only)
        # GPU prefix caching OFF → OffloadingConnector is the sole KV store.
        # Every request stores all blocks to CPU; next turn reloads from CPU.
        PREFIX_CACHE_FLAG="--no-enable-prefix-caching"
        KV_OFFLOAD_FLAG="--kv-offloading-size"
        KV_OFFLOAD_SIZE="72"
        KV_TRANSFER_CONFIG='{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":77309411328,"store_threshold":0,"eviction_policy":"lru"}}'
        DISABLE_HMA_FLAG="--disable-hybrid-kv-cache-manager"
        ;;

    mtier-only)
        # GPU prefix caching OFF → OffloadingConnector (MTier) is the sole KV store.
        # xmem uses bank_id=0 only (single bank, 70 GB). Max usable = 68 GiB with headroom.
        PREFIX_CACHE_FLAG="--no-enable-prefix-caching"
        KV_OFFLOAD_FLAG="--kv-offloading-size"
        KV_OFFLOAD_SIZE="68"
        MTIER_FLAG="--kv-offloading-mtier"
        KV_TRANSFER_CONFIG='{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":73014444032,"store_threshold":0,"eviction_policy":"lru"}}'
        DISABLE_HMA_FLAG="--disable-hybrid-kv-cache-manager"
        ;;

    hybrid-cpu)
        # GPU HBM prefix cache + CPU DRAM overflow via OffloadingConnector.
        PREFIX_CACHE_FLAG="--enable-prefix-caching"
        KV_OFFLOAD_FLAG="--kv-offloading-size"
        KV_OFFLOAD_SIZE="72"
        KV_TRANSFER_CONFIG='{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":77309411328,"store_threshold":0,"eviction_policy":"lru"}}'
        DISABLE_HMA_FLAG="--disable-hybrid-kv-cache-manager"
        ;;

    hybrid-mtier)
        # GPU HBM prefix cache + MTier chip overflow via OffloadingConnector.
        # xmem uses bank_id=0 only (single bank, 70 GB). Max usable = 68 GiB with headroom.
        PREFIX_CACHE_FLAG="--enable-prefix-caching"
        KV_OFFLOAD_FLAG="--kv-offloading-size"
        KV_OFFLOAD_SIZE="68"
        MTIER_FLAG="--kv-offloading-mtier"
        KV_TRANSFER_CONFIG='{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":73014444032,"store_threshold":0,"eviction_policy":"lru"}}'
        DISABLE_HMA_FLAG="--disable-hybrid-kv-cache-manager"
        ;;
esac

# ── Print config ─────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Model   : $MODEL"
echo "  Host    : $HOST"
echo "  Port    : $PORT  (/v1/messages + /v1/chat/completions)"
echo "  GPUs    : $CVD"
echo "  TP      : $TP"
echo "  Max len : $MAX_MODEL_LEN"
echo "  KV dtype: $KV_CACHE_DTYPE"
echo "  Mode    : $MODE"
echo "  Max seqs: $MAX_NUM_SEQS"
echo "  Sampling: temperature=0, seed=42 (deterministic)"
echo "  Log     : $VLLM_LOG"
echo "═══════════════════════════════════════════════════════"
echo ""

# ── Kill any stale vLLM process on our port ──────────────────────
if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
    echo "WARNING: port $PORT already in use — killing stale vLLM processes..."
    pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
    sleep 3
fi

# ── Cleanup ──────────────────────────────────────────────────────
VLLM_PID=""
TAIL_PID=""

cleanup() {
    echo ""
    echo "Stopping vLLM..."
    [[ -n "$TAIL_PID"  ]] && kill "$TAIL_PID"  2>/dev/null || true
    [[ -n "$VLLM_PID"  ]] && kill "$VLLM_PID"  2>/dev/null || true
    wait 2>/dev/null || true

    echo "Freeing GPU memory on device(s): $CVD ..."
    for gpu_id in ${CVD//,/ }; do
        nvidia-smi --query-compute-apps=pid --format=csv,noheader --id="$gpu_id" 2>/dev/null \
            | xargs -r kill -9 2>/dev/null || true
    done

    if [[ "$MODE" == *mtier* ]]; then
        echo "Resetting MTier memory..."
        echo "yes" | mtier_service reset 2>/dev/null || true
    fi

    echo "Done."
    exit 0
}
trap cleanup INT TERM

# ── Start vLLM server ────────────────────────────────────────────
echo "[1/2] Starting vLLM server on GPUs $CVD ..."
CUDA_VISIBLE_DEVICES="$CVD" \
PYTHONHASHSEED=0 \
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
VLLM_LOG_STATS_INTERVAL=1 \
HF_HUB_OFFLINE=1 \
    "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --served-model-name "$MODEL" \
        --host "$HOST" \
        --port "$PORT" \
        --tensor-parallel-size "$TP" \
        --dtype "$DTYPE" \
        --max-model-len "$MAX_MODEL_LEN" \
        --kv-cache-dtype "$KV_CACHE_DTYPE" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        $PREFIX_CACHE_FLAG \
        ${KV_OFFLOAD_FLAG:+$KV_OFFLOAD_FLAG "$KV_OFFLOAD_SIZE"} \
        ${MTIER_FLAG:+$MTIER_FLAG} \
        ${KV_TRANSFER_CONFIG:+--kv-transfer-config "$KV_TRANSFER_CONFIG"} \
        ${DISABLE_HMA_FLAG:+$DISABLE_HMA_FLAG} \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --max-num-batched-tokens 65536 \
        ${NUM_GPU_BLOCKS_OVERRIDE:+--num-gpu-blocks-override "$NUM_GPU_BLOCKS_OVERRIDE"} \
        --override-generation-config '{"temperature":0,"seed":42}' \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_coder \
        > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
echo "      PID $VLLM_PID  (log: $VLLM_LOG)"

# ── Wait for health ───────────────────────────────────────────────
echo ""
echo "[2/2] Waiting for vLLM to be ready — streaming log:"
echo ""
tail -f "$VLLM_LOG" | grep -v '"GET /metrics HTTP' &
TAIL_PID=$!

until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$TAIL_PID" 2>/dev/null || true
        echo ""
        echo "ERROR: vLLM died. Check: tail $VLLM_LOG"
        cleanup
    fi
    sleep 2
done
kill "$TAIL_PID" 2>/dev/null || true
TAIL_PID=""
echo ""
echo "  vLLM ready!"
echo ""
echo "  Claude Code:  ANTHROPIC_BASE_URL=http://localhost:$PORT"
echo "  Launch:       ./run_claude.sh"
echo "  Metrics:      http://localhost:$PORT/metrics"
echo "  Watch:        python3 netpreme/coding_agents/monitoring/watch_vllm.py"
echo "  PID:          vllm=$VLLM_PID"
echo "  Ctrl-C to stop."
echo ""

# ── Tail log in foreground ────────────────────────────────────────
tail -f "$VLLM_LOG" | grep -v '"GET /metrics HTTP' &
TAIL_PID=$!

while true; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo ""
        echo "vLLM process ($VLLM_PID) exited. Cleaning up..."
        break
    fi
    sleep 5
done
cleanup
