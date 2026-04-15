#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════
#  One-time setup: install Prometheus + Grafana.
#  Run once, then use start_monitoring.sh for daily use.
# ═══════════════════════════════════════════════════════════
set -euo pipefail

# ── 1. Install Prometheus ────────────────────────────────────
echo "Installing Prometheus..."
sudo apt-get install -y prometheus

# Stop the default service — we run ours manually with our own config
sudo systemctl stop prometheus 2>/dev/null || true
sudo systemctl disable prometheus 2>/dev/null || true

# ── 2. Install prometheus_client Python package ──────────────
echo "Installing prometheus_client Python package..."
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
uv pip install prometheus_client --quiet --python "${REPO_ROOT}/.venv"

# ── 3. Download Grafana standalone binary ────────────────────
GRAFANA_DIR="$HOME/grafana"
if [[ ! -d "$GRAFANA_DIR" ]]; then
    echo "Downloading Grafana..."
    GRAFANA_VER="11.4.0"
    GRAFANA_TAR="grafana-${GRAFANA_VER}.linux-amd64.tar.gz" # ok with nvidia
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
