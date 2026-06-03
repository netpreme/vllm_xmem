#!/usr/bin/env bash
# Boots the vLLM server with the Anthropic /v1/messages endpoint enabled
# (provided by vllm_xmem). Knobs sourced from .env if present, else defaulted.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Variable precedence is: caller env > .env file > hardcoded defaults.
# `run.sh` exports overridden values before invoking reset_vllm.sh → us,
# so caller-set vars must win over what's in .env.
if [[ -f "$HERE/.env" ]]; then
    while IFS='=' read -r key val; do
        [[ -z "$key" || "$key" =~ ^[[:space:]]*# ]] && continue
        val="${val%%[[:space:]]#*}"             # strip inline comment
        val="${val%"${val##*[![:space:]]}"}"    # rtrim trailing whitespace
        [[ -z "${!key+x}" ]] && export "$key=$val"
    done < "$HERE/.env"
fi

: "${MODEL_NAME:=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8}"
: "${TENSOR_PARALLEL_SIZE:=1}"
: "${MAX_MODEL_LEN:=131072}"
: "${GPU_MEMORY_UTILIZATION:=0.9}"
: "${TOOL_CALL_PARSER:=qwen3_coder}"
: "${PORT:=8000}"

# Model-specific args that have no sensible shared default. Only set what the
# model genuinely requires; never override knobs the user can pick.
REASONING_ARGS=()
case "$MODEL_NAME" in
    openai/gpt-oss-*)
        REASONING_ARGS=( --reasoning-parser openai_gptoss )
        ;;
esac

echo "[server] model=$MODEL_NAME tp=$TENSOR_PARALLEL_SIZE max_model_len=$MAX_MODEL_LEN port=$PORT"

# Use the `vllm` console script from the SAME env that launched the run: the
# Server exports VLLM_PYTHON=sys.executable, and the vllm entrypoint sits next
# to that interpreter (e.g. <env>/bin/vllm). This never hardcodes a venv path
# and always matches the env the user is actually in — which carries our
# editable vLLM patches (e.g. return_token_ids). Fall back to PATH if absent.
VLLM_BIN=""
[[ -n "${VLLM_PYTHON:-}" ]] && VLLM_BIN="$(dirname "$VLLM_PYTHON")/vllm"
[[ -x "$VLLM_BIN" ]] || VLLM_BIN=vllm

exec "$VLLM_BIN" serve "$MODEL_NAME" \
    --host 0.0.0.0 --port "$PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --enable-auto-tool-choice \
    --tool-call-parser "$TOOL_CALL_PARSER" \
    "${REASONING_ARGS[@]}" \
    --enable-prompt-tokens-details \
    "$@"
