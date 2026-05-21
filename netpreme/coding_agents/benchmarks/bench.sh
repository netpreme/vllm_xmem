#!/usr/bin/env bash
# Single entrypoint. Always runs hybrid-mtier (GPU0:8001) + hybrid-cpu (GPU1:8002)
# in parallel. Monitoring stack (Prometheus + Grafana + exporters) is auto-started
# if not already running. Analysis figures + per-turn ISL/OSL/uncached/timings
# CSV are emitted automatically.
#
# Usage:
#   ./bench.sh --concurrency 16 --sustained-mins 20
#   ./bench.sh --concurrency 16 --sustained-mins 20 --save-trace results_benchmarks/record_c016/
#   ./bench.sh --from-trace results_benchmarks/record_c016/
#   ./bench.sh --from-trace results_benchmarks/record_c016/ --osl 1
#
# View live: http://localhost:3000 (Grafana, no login)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PY="${PY:-${REPO_ROOT}/.venv/bin/python}"
[[ -x "$PY" ]] || PY="$(command -v python3)"

# Determinism is controlled by bench.py's --deterministic flag; we don't
# default VLLM_BATCH_INVARIANT here. bench.py sets the env explicitly.

# ── Auto-start the monitoring stack if Grafana is not already up ──────────
if ! curl -fsS http://localhost:3000/api/health > /dev/null 2>&1; then
    echo "[bench] Starting monitoring stack (Prometheus + Grafana + exporters)..."
    MON_SCRIPT="${SCRIPT_DIR}/../monitoring/start_monitoring.sh"
    nohup bash "$MON_SCRIPT" > /tmp/monitoring.log 2>&1 &
    for i in {1..30}; do
        sleep 1
        if curl -fsS http://localhost:3000/api/health > /dev/null 2>&1; then
            echo "[bench] Monitoring ready — http://localhost:3000  (Grafana)"
            break
        fi
        if [[ $i -eq 30 ]]; then
            echo "[bench] WARN: monitoring did not become ready in 30 s (see /tmp/monitoring.log)" >&2
        fi
    done
else
    echo "[bench] Monitoring already running — http://localhost:3000"
fi

exec "$PY" "${SCRIPT_DIR}/utils/bench.py" "$@"
