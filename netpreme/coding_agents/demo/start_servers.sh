#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
#  Demo: start hybrid-cpu (GPU 1, port 8001) and hybrid-mtier
#  (GPU 0, port 8002) side-by-side.
#  MTier MUST be on GPU 0 (cuMemCreate topology — GPU 1 causes 3.7× slowdown).
#
#  Passes --num-gpu-blocks-override 1024 so the GPU cache fills
#  quickly (at ~c=2-4 with 30K-token prompts), forcing eviction to
#  CPU DRAM vs MTier chip and making the speedup visible immediately.
#
#  Usage:
#    ./demo_start.sh               # start both servers (foreground)
#    ./demo_start.sh --no-cap      # no GPU block cap (full HBM cache)
#
#  Then in another terminal:
#    python3 demo_compare.py
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
START_SCRIPT="${SCRIPT_DIR}/../start_server.sh"
MONITORING_SCRIPT="${SCRIPT_DIR}/../monitoring/start_monitoring.sh"

# ── Kill any stale vLLM/demo servers ────────────────────────────────
echo "Killing any stale vLLM processes..."
pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
sleep 2

# ── Start monitoring (Prometheus + Grafana) ──────────────────────────
echo "Starting monitoring stack (Prometheus :9090, Grafana :3000)..."
bash "$MONITORING_SCRIPT" &
MONITORING_PID=$!
echo "  PID=$MONITORING_PID  logs: /tmp/prometheus.log  /tmp/grafana.log"
echo ""

# ── Parse flags ──────────────────────────────────────────────────────
GPU_BLOCKS_OVERRIDE="${GPU_BLOCKS_OVERRIDE:-}"
for arg in "$@"; do
    case "$arg" in
        --no-cap) GPU_BLOCKS_OVERRIDE="" ;;
        --cap=*) GPU_BLOCKS_OVERRIDE="${arg#--cap=}" ;;
        --cap) shift; GPU_BLOCKS_OVERRIDE="$1" ;;
        *) echo "Unknown arg: $arg" >&2; exit 1 ;;
    esac
done

EXTRA_FLAG=""
[[ -n "$GPU_BLOCKS_OVERRIDE" ]] && EXTRA_FLAG="--num-gpu-blocks-override $GPU_BLOCKS_OVERRIDE"

echo ""
echo "Starting demo servers..."
if [[ -n "$GPU_BLOCKS_OVERRIDE" ]]; then
    echo "  GPU block cap: $GPU_BLOCKS_OVERRIDE blocks (~$((GPU_BLOCKS_OVERRIDE * 16 / 1024))K token GPU KV budget)"
else
    echo "  GPU block cap: none (full HBM cache)"
fi
echo ""

# ── Start CPU server (GPU 1, port 8001) ──────────────────────────────
# CPU offloading has no GPU topology constraint; use GPU 1 to leave GPU 0 for MTier.
echo "[1/2] hybrid-cpu  → GPU 1, port 8001"
CUDA_VISIBLE_DEVICES=1 PORT=8001 GPU_MEMORY_UTILIZATION=0.75 \
    MAX_MODEL_LEN=262144 \
    NUM_GPU_BLOCKS_OVERRIDE="$GPU_BLOCKS_OVERRIDE" \
    bash "$START_SCRIPT" --hybrid-cpu &
CPU_PID=$!
echo "      PID=$CPU_PID  log: /tmp/vllm_server_8001.log"

# ── Start MTier server (GPU 0, port 8002) ────────────────────────────
# MTier MUST run on GPU 0: cuMemCreate allocates on physical GPU 0.
# Running on GPU 1 routes transfers through NVLink causing 3.7× slowdown.
echo "[2/2] hybrid-mtier → GPU 0, port 8002"
CUDA_VISIBLE_DEVICES=0 PORT=8002 GPU_MEMORY_UTILIZATION=0.75 \
    MAX_MODEL_LEN=262144 \
    NUM_GPU_BLOCKS_OVERRIDE="$GPU_BLOCKS_OVERRIDE" \
    bash "$START_SCRIPT" --hybrid-mtier &
MTIER_PID=$!
echo "      PID=$MTIER_PID  log: /tmp/vllm_server_8002.log"

echo ""
echo "Waiting for both servers to be healthy..."

wait_healthy() {
    local port=$1 label=$2
    local t0; t0=$(date +%s)
    while true; do
        if curl -sf "http://localhost:$port/health" > /dev/null 2>&1; then
            local elapsed=$(( $(date +%s) - t0 ))
            echo "  ✓ $label ready (${elapsed}s)"
            return 0
        fi
        sleep 3
    done
}

wait_healthy 8001 "hybrid-cpu  (port 8001)" &
wait_healthy 8002 "hybrid-mtier(port 8002)" &
wait

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Both servers ready."
echo "  CPU   : http://localhost:8001   GPU 1"
echo "  MTier : http://localhost:8002   GPU 0"
echo "  Metrics: :8001/metrics  :8002/metrics"
echo ""
echo "  Run the demo:  python3 demo_compare.py"
echo "  Grafana:       http://localhost:3000"
echo "  Ctrl-C to stop both servers."
echo "═══════════════════════════════════════════════════════════════"

cleanup() {
    echo ""
    echo "Stopping demo servers and monitoring..."
    kill $CPU_PID  2>/dev/null || true
    kill $MTIER_PID 2>/dev/null || true
    kill $MONITORING_PID 2>/dev/null || true
    pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
    nvidia-smi --query-compute-apps=pid --format=csv,noheader --id=0 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    nvidia-smi --query-compute-apps=pid --format=csv,noheader --id=1 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    echo "yes" | mtier_service reset 2>/dev/null || true
    echo "Done."
}
trap cleanup INT TERM

wait $CPU_PID $MTIER_PID $MONITORING_PID
