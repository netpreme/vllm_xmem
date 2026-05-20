#!/usr/bin/env bash
# Cold-restart vLLM so each problem starts with an empty prefix cache.
# Kills the server + its forked EngineCore worker (which pkill -f "vllm serve"
# alone misses), waits for the GPU to release, reboots via server.sh, and
# blocks until /v1/models responds.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PORT=8000
READY_TIMEOUT=600
GPU_RELEASE_TIMEOUT=60

echo "[reset_vllm] killing vllm + workers"
for pat in "vllm serve" "VLLM::EngineCore" "vllm.v1.engine" \
           "multiprocessing.resource_tracker"; do
    pkill -9 -f "$pat" 2>/dev/null || true
done

# Wait for the listening port AND the GPU to actually release. Without the
# GPU check, the next vllm process can OOM during CUDA-context init and die.
start=$(date +%s)
while ss -tln 2>/dev/null | grep -q ":${PORT}\b"; do sleep 1; done
echo "[reset_vllm] waiting for GPU memory"
while :; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    (( used < 1000 )) && { echo "[reset_vllm] GPU free after $(($(date +%s) - start))s"; break; }
    (( $(date +%s) - start > GPU_RELEASE_TIMEOUT )) && { echo "[reset_vllm] timeout — proceeding ($used MiB still used)"; break; }
    sleep 2
done

nohup bash "$ROOT/server.sh" >/tmp/vllm_server.log 2>&1 &
PID=$!
echo "[reset_vllm] started pid $PID, waiting for /v1/models"
start=$(date +%s)
while :; do
    curl -fsS --max-time 5 "http://localhost:${PORT}/v1/models" >/dev/null 2>&1 \
        && { echo "[reset_vllm] ready after $(($(date +%s) - start))s"; exit 0; }
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "[reset_vllm] vllm died — tail of /tmp/vllm_server.log:" >&2
        tail -n 40 /tmp/vllm_server.log >&2
        exit 1
    fi
    (( $(date +%s) - start > READY_TIMEOUT )) && { tail -n 40 /tmp/vllm_server.log >&2; exit 1; }
    sleep 2
done
