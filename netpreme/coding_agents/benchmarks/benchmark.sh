#!/usr/bin/env bash
# Single entrypoint. Replays run hybrid-mtier (GPU0:8001) + hybrid-cpu (GPU1:8002)
# in parallel. Agent capture mode runs mtier only.
# Monitoring stack (Prometheus + Grafana + exporters) is auto-started if not
# already running. Analysis figures + per-turn ISL/OSL/uncached/timings CSV
# are emitted automatically.
#
# Determinism (orthogonal to mode — append --deterministic to any command):
#   --deterministic  Pins VLLM_BATCH_INVARIANT=1 + temp=0 + seed=42 on the vLLM
#                    server. In replay mode, also pins OSL exactly via
#                    min_tokens + ignore_eos. Default OFF.
#
# Usage modes:
#   # Agent capture (SWE-bench tasks via Claude Code, mtier only):
#   ./benchmark.sh --concurrency 16 --sustained-mins 20 --save-trace
#   ./benchmark.sh --concurrency 16 --sustained-mins 20 --save-trace --deterministic
#
#   # Synthetic capture (controlled ISL/ISL_new/OSL, no GPU, no agents):
#   ./benchmark.sh --save-trace --isl 27000 --osl 110 --isl-new 500 \
#              --n-turns 50 --n-sessions 30
#
#   # Replay any capture (mtier + cpu side-by-side):
#   ./benchmark.sh --from-trace results_benchmarks/bench_sweep_xxx/c016/ --concurrency 16
#   ./benchmark.sh --from-trace <dir> --concurrency 16 --deterministic
#   ./benchmark.sh --from-trace <dir> --concurrency 16 --osl 1   # override OSL to 1
#
#   # Plain dual-backend benchmark (no trace work):
#   ./benchmark.sh --concurrency 16 --sustained-mins 20
#   ./benchmark.sh --concurrency 16 --sustained-mins 20 --deterministic
#
# View live: http://localhost:3000 (Grafana, no login)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PY="${PY:-${REPO_ROOT}/.venv/bin/python}"
[[ -x "$PY" ]] || PY="$(command -v python3)"

# Determinism is controlled by cli.py's --deterministic flag; we don't
# default VLLM_BATCH_INVARIANT here. cli.py sets the env explicitly.

# ── Auto-start the monitoring stack if Grafana is not already up ──────────
if ! curl -fsS http://localhost:3000/api/health > /dev/null 2>&1; then
    echo "[bench] Starting monitoring stack (Prometheus + Grafana + exporters)..."
    MON_SCRIPT="${SCRIPT_DIR}/../monitoring/run.sh"
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

exec "$PY" "${SCRIPT_DIR}/cli.py" "$@"
