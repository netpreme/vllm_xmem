#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════
#  One-time monitoring setup: Prometheus + Grafana
#  Run once, then use start_monitoring.sh for daily use.
# ═══════════════════════════════════════════════════════════
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 1. Install Prometheus ────────────────────────────────────
echo "Installing Prometheus..."
sudo apt-get install -y prometheus

# Stop the default prometheus service (we'll run ours manually)
sudo systemctl stop prometheus 2>/dev/null || true
sudo systemctl disable prometheus 2>/dev/null || true

# ── 2. Download Grafana standalone binary ────────────────────
GRAFANA_DIR="$HOME/grafana"
if [[ ! -d "$GRAFANA_DIR" ]]; then
    echo "Downloading Grafana standalone binary..."
    GRAFANA_VER="11.4.0"
    GRAFANA_TAR="grafana-${GRAFANA_VER}.linux-amd64.tar.gz"
    wget -q "https://dl.grafana.com/oss/release/${GRAFANA_TAR}" -O "/tmp/${GRAFANA_TAR}"
    mkdir -p "$GRAFANA_DIR"
    tar -xzf "/tmp/${GRAFANA_TAR}" --strip-components=1 -C "$GRAFANA_DIR"
    rm "/tmp/${GRAFANA_TAR}"
    echo "Grafana extracted to $GRAFANA_DIR"
else
    echo "Grafana already at $GRAFANA_DIR, skipping download."
fi

echo ""
echo "Setup complete. Run ./start_monitoring.sh to start everything."
