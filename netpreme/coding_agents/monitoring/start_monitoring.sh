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
# Persistent data + log locations (NOT /tmp — tmp gets cleaned and we lose Prom history).
MON_HOME="${MON_HOME:-$HOME/monitoring_state}"
PROM_DATA_DIR="$MON_HOME/prometheus_data"
GRAFANA_DATA_DIR="$MON_HOME/grafana_data"
MON_LOG_DIR="$MON_HOME/logs"
mkdir -p "$PROM_DATA_DIR" "$GRAFANA_DATA_DIR" "$MON_LOG_DIR"

# Flag parsing (any order):
#   --keep      : do NOT wipe Prom data (default now — used to be opt-in)
#   --wipe      : explicitly wipe Prom data before start
#   --headless  : launch processes and exit; do NOT wait/trap. Each process is
#                 setsid-detached so SIGTERM to this script never propagates.
KEEP_DATA="--keep"
HEADLESS=""
for arg in "$@"; do
    case "$arg" in
        --keep)     KEEP_DATA="--keep" ;;
        --wipe)     KEEP_DATA="--wipe" ;;
        --headless) HEADLESS="1" ;;
        *)          echo "WARN: unknown arg $arg" ;;
    esac
done

# ── 1. Kill any existing instances ──────────────────────────
echo "Stopping any running Prometheus / Grafana / kv_exporter / gpu_exporter..."
pkill -f "prometheus --config.file" 2>/dev/null || true
# The grafana binary spawns as "grafana server" (space, not hyphen).
# Match on "grafana" broadly to catch manually started instances too.
pkill -f "grafana"                  2>/dev/null || true
pkill -f "kv_exporter.py"          2>/dev/null || true
pkill -f "gpu_exporter.py"         2>/dev/null || true
sleep 2  # wait for processes to die before wiping data dir

# ── 2. Reset Prometheus data (only if --wipe) ─────────────────
if [[ "$KEEP_DATA" == "--wipe" ]]; then
    echo "Wiping Prometheus data at $PROM_DATA_DIR ..."
    rm -rf "$PROM_DATA_DIR"
fi
mkdir -p "$PROM_DATA_DIR"

# ── 3. Start Prometheus (setsid-detached so it survives parent TERM) ─
setsid prometheus \
    --config.file="$MONITORING_DIR/prometheus.yml" \
    --storage.tsdb.path="$PROM_DATA_DIR" \
    --storage.tsdb.retention.time=1d \
    --web.enable-admin-api \
    </dev/null > "$MON_LOG_DIR/prometheus.log" 2>&1 &
PROM_PID=$!
disown 2>/dev/null
echo "Prometheus started (pid $PROM_PID) → http://localhost:9090"
echo "  log: $MON_LOG_DIR/prometheus.log"

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

setsid env \
    GF_PATHS_PROVISIONING="$MONITORING_DIR/grafana_provisioning" \
    GF_SERVER_HTTP_PORT=3000 \
    GF_AUTH_ANONYMOUS_ENABLED=true \
    GF_AUTH_ANONYMOUS_ORG_NAME="Main Org." \
    GF_AUTH_ANONYMOUS_ORG_ROLE=Admin \
    GF_SECURITY_ALLOW_EMBEDDING=true \
    "$GRAFANA_DIR/bin/grafana-server" \
    --homepath="$GRAFANA_DIR" \
    cfg:paths.data="$GRAFANA_DATA_DIR" \
    cfg:paths.logs="$MON_LOG_DIR/grafana.log" \
    </dev/null > "$MON_LOG_DIR/grafana_stdout.log" 2>&1 &
GRAFANA_PID=$!
disown 2>/dev/null
echo "Grafana   started (pid $GRAFANA_PID) → http://localhost:3000"
echo "  log: $MON_LOG_DIR/grafana.log"

# ── 5. Start KV exporter ─────────────────────────────────────
setsid "$PYTHON_BIN" "$MONITORING_DIR/kv_exporter.py" \
    --log "/tmp/dynamo_worker_*.log" \
    --port 9091 </dev/null > "$MON_LOG_DIR/kv_exporter.log" 2>&1 &
EXPORTER_PID=$!
disown 2>/dev/null
echo "KV exporter started (pid $EXPORTER_PID) → http://localhost:9091"
echo "  log: $MON_LOG_DIR/kv_exporter.log"

# ── 5b. Start GPU exporter (nvidia-smi → prom_client) ────────
setsid "$PYTHON_BIN" "$MONITORING_DIR/gpu_exporter.py" \
    --port 9092 --interval 1.0 </dev/null > "$MON_LOG_DIR/gpu_exporter.log" 2>&1 &
GPU_EXPORTER_PID=$!
disown 2>/dev/null
echo "GPU exporter started (pid $GPU_EXPORTER_PID) → http://localhost:9092"
echo "  log: $MON_LOG_DIR/gpu_exporter.log"

echo ""
echo "Dashboard auto-loaded: 'vLLM + XMem — Unified'"
echo "  Prometheus:   http://localhost:9090"
echo "  Grafana:      http://localhost:3000  (no login required)"
echo "  KV exporter:  http://localhost:9091/metrics"
echo ""
if [[ -n "$HEADLESS" ]]; then
    echo "Headless mode — services launched detached; this script exits now."
    exit 0
fi

echo "Press Ctrl+C to stop this watcher (services keep running — they are setsid-detached)."

# ── 6. Optional foreground wait — Ctrl+C only exits this script,
#       it no longer kills the children (they are in different sessions).
trap "echo; echo 'Watcher exiting; services remain running.'; exit 0" INT TERM
wait "$PROM_PID" "$GRAFANA_PID" "$EXPORTER_PID" "$GPU_EXPORTER_PID" 2>/dev/null || true
