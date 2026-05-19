#!/usr/bin/env bash
# Boots the vLLM server with the Anthropic /v1/messages endpoint enabled
# (provided by vllm_xmem). Knobs sourced from .env if present, else defaulted.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$HERE/.env" ]] && { set -a; source "$HERE/.env"; set +a; }

: "${MODEL_NAME:=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8}"
: "${TENSOR_PARALLEL_SIZE:=1}"
: "${MAX_MODEL_LEN:=262144}"
: "${GPU_MEMORY_UTILIZATION:=0.92}"
: "${TOOL_CALL_PARSER:=qwen3_coder}"

echo "[server] model=$MODEL_NAME tp=$TENSOR_PARALLEL_SIZE max_model_len=$MAX_MODEL_LEN"

VLLM_BIN=/root/vllm_xmem/.venv/bin/vllm
[[ -x "$VLLM_BIN" ]] || VLLM_BIN=vllm

exec "$VLLM_BIN" serve "$MODEL_NAME" \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --enable-auto-tool-choice \
    --tool-call-parser "$TOOL_CALL_PARSER" \
    --enable-prompt-tokens-details \
    "$@"
