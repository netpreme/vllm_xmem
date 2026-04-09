#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════
#  Start monitoring stack:
#    1. kv_exporter.py     → :9091  (KV transfer metrics)
#    2. prometheus          → :9090  (scrapes :8000 + :9091)
#    3. grafana             → :3000  (dashboard UI)
#
#  Prerequisite: run setup_monitoring.sh once first.
#  Ctrl-C stops all three.
# ═══════════════════════════════════════════════════════════
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
[[ ! -x "$PYTHON_BIN" ]] && PYTHON_BIN="$(command -v python3)"

GRAFANA_DIR="$HOME/grafana"
PROM_CONFIG="$SCRIPT_DIR/prometheus.yml"

cleanup() {
    echo ""
    echo "Stopping monitoring stack..."
    kill "$EXPORTER_PID" "$PROM_PID" "$GRAFANA_PID" 2>/dev/null || true
    wait 2>/dev/null
    echo "Done."
    exit 0
}
trap cleanup INT TERM

# ── 1. KV exporter ──────────────────────────────────────────
echo "[1/3] Starting KV exporter on :9091 ..."
"$PYTHON_BIN" "$SCRIPT_DIR/kv_exporter.py" \
    --log "/tmp/dynamo_worker_*.log" \
    --port 9091 > /tmp/kv_exporter.log 2>&1 &
EXPORTER_PID=$!
echo "      PID $EXPORTER_PID  (log: /tmp/kv_exporter.log)"

# ── 2. Prometheus ────────────────────────────────────────────
echo "[2/3] Starting Prometheus on :9090 ..."
prometheus \
    --config.file="$PROM_CONFIG" \
    --storage.tsdb.path="/tmp/prometheus_data" \
    --web.listen-address=":9090" \
    > /tmp/prometheus.log 2>&1 &
PROM_PID=$!
echo "      PID $PROM_PID  (log: /tmp/prometheus.log)"

# ── 3. Grafana ───────────────────────────────────────────────
if [[ -d "$GRAFANA_DIR" ]]; then
    echo "[3/3] Starting Grafana on :3000 ..."
    GF_PATHS_DATA="/tmp/grafana_data" \
    GF_SERVER_HTTP_PORT=3000 \
    GF_SECURITY_ADMIN_PASSWORD=admin \
    GF_AUTH_ANONYMOUS_ENABLED=true \
    GF_AUTH_ANONYMOUS_ORG_ROLE=Admin \
        "$GRAFANA_DIR/bin/grafana" server \
            --homepath "$GRAFANA_DIR" \
            > /tmp/grafana.log 2>&1 &
    GRAFANA_PID=$!
    echo "      PID $GRAFANA_PID  (log: /tmp/grafana.log)"
else
    echo "[3/3] Grafana not found at $GRAFANA_DIR — skipping."
    echo "      Run setup_monitoring.sh first, or add Prometheus manually as a datasource."
    GRAFANA_PID=""
fi

echo ""
echo "══════════════════════════════════════════════"
echo "  Prometheus:  http://localhost:9090"
echo "  Grafana:     http://localhost:3000  (admin/admin)"
echo "  KV Exporter: http://localhost:9091/metrics"
echo ""
echo "  Import dashboard:"
echo "    Grafana → Dashboards → Import → Upload JSON"
echo "    File: $SCRIPT_DIR/grafana_dashboard.json"
echo "══════════════════════════════════════════════"
echo ""
echo "Ctrl-C to stop everything."

wait
