#!/usr/bin/env bash
# One-time install: Prometheus, Grafana, GPU exporter deps, and the
# pre-commit hooks expected by AGENTS.md.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

echo "[1/4] Installing Prometheus + psmisc (for fuser) ..."
sudo apt-get install -y prometheus psmisc
sudo systemctl stop prometheus    2>/dev/null || true
sudo systemctl disable prometheus 2>/dev/null || true

echo "[2/4] Installing Python deps in the project venv ..."
uv pip install --quiet --python "${REPO_ROOT}/.venv" \
    prometheus_client requests aiohttp datasets pandas matplotlib

echo "[3/4] Downloading Grafana standalone binary ..."
GRAFANA_DIR="$HOME/grafana"
if [[ ! -d "$GRAFANA_DIR" ]]; then
    GRAFANA_VER="11.4.0"
    GRAFANA_TAR="grafana-${GRAFANA_VER}.linux-amd64.tar.gz"
    wget -q "https://dl.grafana.com/oss/release/${GRAFANA_TAR}" -O "/tmp/${GRAFANA_TAR}"
    mkdir -p "$GRAFANA_DIR"
    tar -xzf "/tmp/${GRAFANA_TAR}" --strip-components=1 -C "$GRAFANA_DIR"
    rm "/tmp/${GRAFANA_TAR}"
else
    echo "  Grafana already installed at $GRAFANA_DIR"
fi

echo "[4/4] Done."
echo ""
echo "Start the monitoring stack with:"
echo "  bash ${REPO_ROOT}/netpreme/coding_agents/monitoring/start_monitoring.sh"
echo "Then run a benchmark:"
echo "  ./bench.sh --concurrency 16 --sustained-mins 20"
