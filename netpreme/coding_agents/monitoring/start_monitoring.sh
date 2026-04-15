#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════
#  Start Prometheus + Grafana for vLLM / XMem monitoring.
#  Prometheus data is wiped on every start (fresh slate).
#  Run setup_monitoring.sh once before first use.
#
#  Usage:
#    ./start_monitoring.sh            # fresh start (default)
#    ./start_monitoring.sh --keep     # keep existing Prometheus data
# ═══════════════════════════════════════════════════════════
set -euo pipefail

MONITORING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${MONITORING_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="$(command -v python3)"
fi
GRAFANA_DIR="$HOME/grafana"
PROM_DATA_DIR="/tmp/prometheus_data"
GRAFANA_DATA_DIR="/tmp/grafana_data"
KEEP_DATA="${1:-}"

# ── 1. Kill any existing instances ──────────────────────────
echo "Stopping any running Prometheus / Grafana / kv_exporter..."
pkill -f "prometheus --config.file" 2>/dev/null || true
# The grafana binary spawns as "grafana server" (space, not hyphen).
# Match on "grafana" broadly to catch manually started instances too.
pkill -f "grafana"                  2>/dev/null || true
pkill -f "kv_exporter.py"          2>/dev/null || true
sleep 2  # wait for processes to die before wiping data dir

# ── 2. Reset Prometheus data (unless --keep) ─────────────────
if [[ "$KEEP_DATA" == "--keep" ]]; then
    echo "Keeping existing Prometheus data at $PROM_DATA_DIR"
else
    echo "Wiping Prometheus data at $PROM_DATA_DIR ..."
    rm -rf "$PROM_DATA_DIR"
fi
mkdir -p "$PROM_DATA_DIR"

# ── 3. Start Prometheus ──────────────────────────────────────
prometheus \
    --config.file="$MONITORING_DIR/prometheus.yml" \
    --storage.tsdb.path="$PROM_DATA_DIR" \
    --storage.tsdb.retention.time=1d \
    > /tmp/prometheus.log 2>&1 &
PROM_PID=$!
echo "Prometheus started (pid $PROM_PID) → http://localhost:9090"
echo "  log: /tmp/prometheus.log"

# ── 4. Start Grafana ─────────────────────────────────────────
if [[ ! -d "$GRAFANA_DIR" ]]; then
    echo "ERROR: Grafana not found at $GRAFANA_DIR"
    echo "       Run setup_monitoring.sh first."
    kill "$PROM_PID" 2>/dev/null || true
    exit 1
fi

# Wipe Grafana session data so provisioning always applies cleanly
rm -rf "$GRAFANA_DATA_DIR"
mkdir -p "$GRAFANA_DATA_DIR"

GF_PATHS_PROVISIONING="$MONITORING_DIR/grafana_provisioning" \
GF_SERVER_HTTP_PORT=3000 \
GF_AUTH_ANONYMOUS_ENABLED=true \
GF_AUTH_ANONYMOUS_ORG_NAME="Main Org." \
GF_AUTH_ANONYMOUS_ORG_ROLE=Admin \
GF_SECURITY_ALLOW_EMBEDDING=true \
    "$GRAFANA_DIR/bin/grafana-server" \
    --homepath="$GRAFANA_DIR" \
    cfg:paths.data="$GRAFANA_DATA_DIR" \
    cfg:paths.logs=/tmp/grafana.log \
    > /tmp/grafana_stdout.log 2>&1 &
GRAFANA_PID=$!
echo "Grafana   started (pid $GRAFANA_PID) → http://localhost:3000"
echo "  log: /tmp/grafana.log"

# ── 5. Start KV exporter ─────────────────────────────────────
"$PYTHON_BIN" "$MONITORING_DIR/kv_exporter.py" \
    --log "/tmp/dynamo_worker_*.log" \
    --port 9091 > /tmp/kv_exporter.log 2>&1 &
EXPORTER_PID=$!
echo "KV exporter started (pid $EXPORTER_PID) → http://localhost:9091"
echo "  log: /tmp/kv_exporter.log"

echo ""
echo "Dashboard auto-loaded: 'vLLM + XMem — Unified'"
echo "  Prometheus:   http://localhost:9090"
echo "  Grafana:      http://localhost:3000  (no login required)"
echo "  KV exporter:  http://localhost:9091/metrics"
echo ""
echo "Press Ctrl+C to stop all."

# ── 6. Stay in foreground — Ctrl+C kills all ─────────────────
cleanup() {
    echo ""
    echo "Stopping Prometheus / Grafana / kv_exporter..."
    kill "$PROM_PID"     2>/dev/null || true
    kill "$GRAFANA_PID"  2>/dev/null || true
    kill "$EXPORTER_PID" 2>/dev/null || true
    wait "$PROM_PID"     2>/dev/null || true
    wait "$GRAFANA_PID"  2>/dev/null || true
    wait "$EXPORTER_PID" 2>/dev/null || true
    echo "Done."
}
trap cleanup INT TERM

wait "$PROM_PID" "$GRAFANA_PID" "$EXPORTER_PID"
