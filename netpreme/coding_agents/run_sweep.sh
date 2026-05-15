#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
#  Sweep both hybrid-mtier and hybrid-cpu back-to-back, same concurrency list,
#  same duration per level. Snapshots land under benchmarks/results_benchmarks/.
#
#  mtier serves on :8001 (GPU 0), cpu serves on :8002 (GPU 1).
#  Start ./monitoring/start_monitoring.sh first if you want the live Grafana
#  dashboard "MTier vs CPU — Sweep (live)" at http://localhost:3000.
#
#  Usage:
#    bash run_sweep.sh --concurrency 1 10 12 14 16 --sustained-mins 30
#    bash run_sweep.sh --concurrency 1 8 16 --sustained-mins 10 --difficulty hard vhard
#
#  All flags are forwarded to bench_concurrent_users.py except --setup, which
#  is forced to "hybrid-mtier hybrid-cpu".
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$SCRIPT_DIR/benchmarks/bench_concurrent_users.py"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"

# Refuse if the user tried to pass --setup themselves
for arg in "$@"; do
    if [[ "$arg" == "--setup" ]]; then
        echo "ERROR: --setup is forced to 'hybrid-mtier hybrid-cpu' by this wrapper." >&2
        echo "       Call bench_concurrent_users.py directly if you need a single setup." >&2
        exit 2
    fi
done

# ── Pre-flight cleanup ──────────────────────────────────────────────────────
# Kill any vLLM already bound to mtier (:8001) or cpu (:8002) ports, and any
# leftover bench/claude/prometheus processes from a prior run. Then reset MTier.
# We disable -e for this block because the checks naturally exit non-zero when
# there's nothing to kill (which is the happy case).
set +e
echo "[run_sweep] Pre-flight cleanup ..."
fuser -k 8001/tcp 8002/tcp 9090/tcp >/dev/null 2>&1
pkill -9 -f "vllm.entrypoints"          >/dev/null 2>&1
pkill -9 -f "bench_concurrent"          >/dev/null 2>&1
pkill -9 -f "claude --model"            >/dev/null 2>&1
pkill -9 -f "prometheus --config.file"  >/dev/null 2>&1
sleep 2

echo "[run_sweep] Resetting MTier memory ..."
echo yes | mtier_service reset >/dev/null 2>&1
set -e

# ── Ensure Grafana is running so the live dashboard is reachable ────────────
# The bench manages Prometheus itself (per-level), so we only start Grafana
# and the supporting exporters here. Grafana queries http://localhost:9090
# which the bench's per-level Prometheus will serve.
GRAFANA_DIR="$HOME/grafana"
PY_VENV="$HOME/vllm_xmem/.venv/bin/python"
[[ -x "$PY_VENV" ]] || PY_VENV="$(command -v python3)"
MONITORING_DIR="$SCRIPT_DIR/monitoring"

if ! curl -fsS http://localhost:3000/api/health >/dev/null 2>&1; then
    if [[ -x "$GRAFANA_DIR/bin/grafana-server" ]]; then
        echo "[run_sweep] Starting Grafana on :3000 ..."
        rm -rf /tmp/grafana_data; mkdir -p /tmp/grafana_data
        GF_PATHS_PROVISIONING="$MONITORING_DIR/grafana_provisioning" \
        GF_SERVER_HTTP_PORT=3000 \
        GF_AUTH_ANONYMOUS_ENABLED=true \
        GF_AUTH_ANONYMOUS_ORG_NAME="Main Org." \
        GF_AUTH_ANONYMOUS_ORG_ROLE=Admin \
        GF_SECURITY_ALLOW_EMBEDDING=true \
            "$GRAFANA_DIR/bin/grafana-server" \
            --homepath="$GRAFANA_DIR" \
            cfg:paths.data=/tmp/grafana_data \
            cfg:paths.logs=/tmp/grafana.log \
            > /tmp/grafana_stdout.log 2>&1 &
        disown
        for i in {1..20}; do
            curl -fsS http://localhost:3000/api/health >/dev/null 2>&1 && break
            sleep 1
        done
    else
        echo "[run_sweep] (Grafana not installed — skip dashboard. Run setup_monitoring.sh once.)"
    fi
fi

# GPU exporter on :9092 (gpu utilization → Prometheus). bench also tries to
# auto-start it but we ensure it's up before grafana queries.
if ! curl -fsS http://localhost:9092/metrics >/dev/null 2>&1; then
    [[ -f "$MONITORING_DIR/gpu_exporter.py" ]] && \
        nohup "$PY_VENV" "$MONITORING_DIR/gpu_exporter.py" --port 9092 \
              > /tmp/gpu_exporter.log 2>&1 & disown
fi

# Print the dashboard URL so the IDE (Cursor/VSCode remote) auto-forwards it.
if curl -fsS http://localhost:3000/api/health >/dev/null 2>&1; then
    echo ""
    echo "  ────────────────────────────────────────────────────────────────"
    echo "   Live dashboard:  http://localhost:3000/d/mtier-vs-cpu-sweep"
    echo "  ────────────────────────────────────────────────────────────────"
    echo ""
fi

exec "$PYTHON_BIN" "$BENCH" --setup hybrid-mtier hybrid-cpu "$@"
