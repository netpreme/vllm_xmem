#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════
#  One-time install for the coding-agent stack.
#  Assumes vLLM is already installed via `uv pip install -e .` —
#  its requirements/common.txt already provides numpy, requests,
#  aiohttp, and prometheus_client, so we only install what's
#  missing across monitoring, analysis, and the benchmark harness.
#
#  Installs:
#    * prometheus        (apt) — TSDB scraped by Grafana + bench snapshots
#    * psmisc            (apt) — provides `fuser`, used for port cleanup
#    * grafana           (standalone tarball into ~/grafana)
#    * matplotlib pandas (uv pip) — analysis figures
#    * datasets          (uv pip) — HuggingFace SWE-bench loader
# ═══════════════════════════════════════════════════════════
set -euo pipefail

echo "[1/3] Installing apt packages (prometheus, psmisc) ..."
sudo apt-get install -y prometheus psmisc
sudo systemctl stop prometheus    2>/dev/null || true
sudo systemctl disable prometheus 2>/dev/null || true

echo "[2/3] Installing Python deps (matplotlib, pandas, datasets) ..."
uv pip install --quiet matplotlib pandas datasets

echo "[3/3] Downloading Grafana standalone binary ..."
GRAFANA_DIR="$HOME/grafana"
if [[ ! -d "$GRAFANA_DIR" ]]; then
    GRAFANA_VER="11.4.0"
    GRAFANA_TAR="grafana-${GRAFANA_VER}.linux-amd64.tar.gz"
    wget -q "https://dl.grafana.com/oss/release/${GRAFANA_TAR}" -O "/tmp/${GRAFANA_TAR}"
    mkdir -p "$GRAFANA_DIR"
    tar -xzf "/tmp/${GRAFANA_TAR}" --strip-components=1 -C "$GRAFANA_DIR"
    rm "/tmp/${GRAFANA_TAR}"
    echo "  Grafana extracted to $GRAFANA_DIR"
else
    echo "  Grafana already installed at $GRAFANA_DIR"
fi

echo ""
echo "Done. Run a benchmark (monitoring auto-starts on first invocation):"
echo "  ./benchmarks/benchmark.sh --concurrency 16 --sustained-mins 20"
