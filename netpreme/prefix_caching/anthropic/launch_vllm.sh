#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./launch_vllm.sh
#   MODEL="Qwen/Qwen2.5-Coder-7B-Instruct" PORT=9000 ./launch_vllm.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"
VLLM_ROOT="${VLLM_ROOT:-}"

# Load .env (skip blank lines and comments); env vars already set take precedence
if [[ -f "$ENV_FILE" ]]; then
    while IFS='=' read -r key value; do
        [[ -z "$key" || "$key" == \#* ]] && continue
        [[ -v "$key" ]] || export "$key=$value"
    done < "$ENV_FILE"
else
    echo "WARNING: .env not found at $ENV_FILE, using built-in defaults." >&2
fi

MODEL="${MODEL:-Qwen/Qwen2.5-Coder-32B-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DTYPE="${DTYPE:-auto}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
KV_OFFLOAD_MTIER="${KV_OFFLOAD_MTIER:-0}"

echo "Starting vLLM server (Anthropic-compatible)"
echo "  model                  : $MODEL"
echo "  host:port              : $HOST:$PORT"
echo "  tensor-parallel-size   : $TENSOR_PARALLEL_SIZE"
echo "  dtype                  : $DTYPE"
echo "  max-model-len          : $MAX_MODEL_LEN"
echo "  gpu-memory-utilization : $GPU_MEMORY_UTILIZATION"
echo "  kv-offload-mtier       : $KV_OFFLOAD_MTIER"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "  CUDA_VISIBLE_DEVICES   : $CUDA_VISIBLE_DEVICES"
fi

# Optional: run from a local vLLM checkout
if [[ -n "$VLLM_ROOT" ]]; then
    if [[ ! -d "$VLLM_ROOT" ]]; then
        echo "Error: VLLM_ROOT directory not found: $VLLM_ROOT" >&2
        exit 1
    fi
    cd "$VLLM_ROOT"
fi

# Build the env prefix — only set CUDA_VISIBLE_DEVICES if the caller provided it
ENV_PREFIX="VLLM_ALLOW_LONG_MAX_MODEL_LEN=1"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    ENV_PREFIX="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES $ENV_PREFIX"
fi

# Build optional flags
EXTRA_FLAGS=""
if [[ "$KV_OFFLOAD_MTIER" == "1" || "$KV_OFFLOAD_MTIER" == "true" ]]; then
    EXTRA_FLAGS="--kv-offloading-mtier"
fi

exec env $ENV_PREFIX \
    python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name "$MODEL" \
        "claude-opus-4-6" \
        "claude-sonnet-4-6" \
        "claude-haiku-4-5-20251001" \
        "claude-3-5-sonnet-20241022" \
        "claude-3-5-haiku-20241022" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --dtype "$DTYPE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    $EXTRA_FLAGS
