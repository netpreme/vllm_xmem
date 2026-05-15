#!/usr/bin/env bash
# Kills any running vllm (server + EngineCore worker), waits for GPU memory
# to release, starts a fresh server via server.sh, and waits until
# /v1/models responds. Used to give each problem a cold prefix cache.
#
# Usage: bash reset_vllm.sh
# Env:   PORT (default 8000)
#        VLLM_READY_TIMEOUT_S      (default 600)
#        VLLM_GPU_RELEASE_TIMEOUT_S (default 60)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PORT="${PORT:-8000}"
READY_TIMEOUT="${VLLM_READY_TIMEOUT_S:-600}"
GPU_RELEASE_TIMEOUT="${VLLM_GPU_RELEASE_TIMEOUT_S:-60}"

echo "[reset_vllm] killing existing vllm + EngineCore workers"
# Kill the API server, the EngineCore worker procs, and the shared-mem
# resource tracker. The EngineCore is a *forked* multiprocessing child of
# `vllm serve` — pkill -f "vllm serve" alone misses it.
for pat in "vllm serve" "VLLM::EngineCore" "vllm.v1.engine" "multiprocessing.resource_tracker"; do
    pkill -9 -f "$pat" 2>/dev/null || true
done

# Wait for the port to be released.
for _ in $(seq 1 60); do
    ss -tln 2>/dev/null | grep -q ":${PORT}\b" || break
    sleep 1
done

# Wait for GPU memory to actually free — the killed processes can take a few
# seconds to release their CUDA context. Without this the next vllm sees
# OOM and silently dies.
echo "[reset_vllm] waiting for GPU memory to release (timeout ${GPU_RELEASE_TIMEOUT}s)"
start=$(date +%s)
while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    if (( used < 1000 )); then
        echo "[reset_vllm] GPU free after $(($(date +%s) - start))s (${used} MiB used)"
        break
    fi
    if (( $(date +%s) - start > GPU_RELEASE_TIMEOUT )); then
        echo "[reset_vllm] GPU memory still ${used} MiB after ${GPU_RELEASE_TIMEOUT}s — proceeding anyway"
        break
    fi
    sleep 2
done

nohup bash "$ROOT/server.sh" >/tmp/vllm_server.log 2>&1 &
VLLM_PID=$!
echo "[reset_vllm] started pid $VLLM_PID, waiting for /v1/models (timeout ${READY_TIMEOUT}s)"
start=$(date +%s)
while true; do
    if curl -fsS --max-time 5 "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
        echo "[reset_vllm] ready after $(($(date +%s) - start))s"
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[reset_vllm] vllm process died; tail of /tmp/vllm_server.log:"
        tail -n 40 /tmp/vllm_server.log >&2
        exit 1
    fi
    if (( $(date +%s) - start > READY_TIMEOUT )); then
        echo "[reset_vllm] timeout waiting for /v1/models"
        tail -n 40 /tmp/vllm_server.log >&2
        exit 1
    fi
    sleep 2
done
