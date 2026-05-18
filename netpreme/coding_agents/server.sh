#!/usr/bin/env bash
# Boots the vLLM server with the Anthropic /v1/messages endpoint enabled
# (provided by vllm_xmem). All knobs come from the sibling .env file.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a
# shellcheck disable=SC1091
source "$HERE/.env"
set +a

# All knobs must come from .env so there's a single source of truth.
: "${MODEL_NAME:?MODEL_NAME missing in .env}"
: "${SERVED_MODEL_NAME:?SERVED_MODEL_NAME missing in .env}"
: "${HOST:?HOST missing in .env}"
: "${PORT:?PORT missing in .env}"
: "${TENSOR_PARALLEL_SIZE:?TENSOR_PARALLEL_SIZE missing in .env}"
: "${MAX_MODEL_LEN:?MAX_MODEL_LEN missing in .env}"
: "${GPU_MEMORY_UTILIZATION:?GPU_MEMORY_UTILIZATION missing in .env}"
: "${TOOL_CALL_PARSER:?TOOL_CALL_PARSER missing in .env}"

echo "[server] model=$MODEL_NAME served-as=$SERVED_MODEL_NAME tp=$TENSOR_PARALLEL_SIZE port=$PORT max_model_len=$MAX_MODEL_LEN"

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
