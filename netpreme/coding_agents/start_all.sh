#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
#  Start the full monitoring stack (Prometheus + Grafana + KV/GPU exporters)
#  AND a vLLM server in one command.
#
#  Usage (forward any flag start_server.sh accepts):
#     bash start_all.sh --hybrid-mtier              # mtier → :8001, GPU 0
#     bash start_all.sh --hybrid-cpu                # cpu   → :8002, GPU 1
#     bash start_all.sh --mtier-only
#     bash start_all.sh --cpu-only
#     bash start_all.sh --hbm-only
#
#  Optional env overrides (just like start_server.sh):
#     PORT=8003 CUDA_VISIBLE_DEVICES=2 bash start_all.sh --hybrid-mtier
#
#  Ctrl-C cleanly stops vLLM, Prometheus, Grafana, and the exporters.
#
#  Open:
#     Grafana    : http://localhost:3000      (no login)
#     Prometheus : http://localhost:9090
#     Dashboard  : http://localhost:3000/d/mtier-vs-cpu-sweep
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
START_MON="$SCRIPT_DIR/monitoring/start_monitoring.sh"
START_VLLM="$SCRIPT_DIR/start_server.sh"

if [[ ! -x "$START_MON"  ]]; then echo "missing $START_MON";  exit 1; fi
if [[ ! -x "$START_VLLM" ]]; then echo "missing $START_VLLM"; exit 1; fi

# ── 1. Start monitoring in background ────────────────────────────────────────
echo "[start_all] Launching monitoring stack (Prometheus + Grafana + exporters)..."
"$START_MON" > /tmp/start_monitoring.out 2>&1 &
MON_PID=$!
echo "[start_all] monitoring pid=$MON_PID  (log: /tmp/start_monitoring.out)"

# Wait for Prometheus + Grafana to be ready (≤30s)
for i in {1..30}; do
    if curl -fsS http://localhost:9090/-/ready  >/dev/null 2>&1 \
    && curl -fsS http://localhost:3000/api/health >/dev/null 2>&1; then
        echo "[start_all] Prometheus + Grafana ready (${i}s)"
        break
    fi
    if ! kill -0 "$MON_PID" 2>/dev/null; then
        echo "[start_all] ERROR: monitoring died during startup (see /tmp/start_monitoring.out)"
        exit 1
    fi
    sleep 1
done

# ── 2. Cleanup trap — stop everything on Ctrl-C / exit ────────────────────────
cleanup() {
    echo ""
    echo "[start_all] Stopping ..."
    # Stop monitoring (this also kills exporters via its own trap)
    if kill -0 "$MON_PID" 2>/dev/null; then
        kill -INT "$MON_PID" 2>/dev/null || true
        wait "$MON_PID" 2>/dev/null || true
    fi
    # vLLM is in the foreground — it gets its own SIGINT from the shell.
    # Belt-and-braces: in case it forked, also try to kill any remaining vllm.
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    echo "[start_all] Done."
}
trap cleanup INT TERM EXIT

# ── 3. Start vLLM in the foreground (its own trap stops on Ctrl-C) ───────────
echo "[start_all] Launching vLLM ..."
"$START_VLLM" "$@"
