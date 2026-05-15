#!/usr/bin/env bash
# Boots the vLLM server with the Anthropic /v1/messages endpoint enabled
# (provided by vllm_xmem). All knobs come from the sibling .env file.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a
# shellcheck disable=SC1091
source "$HERE/.env"
set +a

: "${MODEL_NAME:?MODEL_NAME missing in .env}"
: "${SERVED_MODEL_NAME:?SERVED_MODEL_NAME missing in .env}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8000}"
: "${TENSOR_PARALLEL_SIZE:=1}"
: "${MAX_MODEL_LEN:=131072}"
: "${GPU_MEMORY_UTILIZATION:=0.92}"
: "${TOOL_CALL_PARSER:=qwen3_coder}"

echo "[server] model=$MODEL_NAME served-as=$SERVED_MODEL_NAME tp=$TENSOR_PARALLEL_SIZE port=$PORT"

# Resolve vllm binary: prefer the project venv, fall back to PATH.
if [[ -x /root/vllm_xmem/.venv/bin/vllm ]]; then
    VLLM_BIN=/root/vllm_xmem/.venv/bin/vllm
else
    VLLM_BIN=vllm
fi

exec "$VLLM_BIN" serve "$MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --enable-auto-tool-choice \
    --tool-call-parser "$TOOL_CALL_PARSER" \
    --enable-prompt-tokens-details \
    "$@"
