#!/usr/bin/env bash
set -euo pipefail

# Start a vLLM OpenAI-compatible server from this local checkout.
# Override with env vars if needed:
#   MODEL="Qwen/Qwen3.5-0.8B" HOST="0.0.0.0" PORT="8000" ./api.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT="${VLLM_ROOT:-$SCRIPT_DIR/vllm_xmem}"

if [[ ! -d "$VLLM_ROOT" ]]; then
  echo "Error: vllm_xmem directory not found at: $VLLM_ROOT"
  echo "Set VLLM_ROOT to your vllm_xmem path and retry."
  exit 1
fi

MODEL="${MODEL:-Qwen/Qwen3.5-0.8B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
DTYPE="${DTYPE:-auto}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
# Qwen3 reasoning is on by default, but keep this explicit.
DEFAULT_CHAT_TEMPLATE_KWARGS="${DEFAULT_CHAT_TEMPLATE_KWARGS:-{\"enable_thinking\": true}}"

cd "$VLLM_ROOT"

exec env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --reasoning-parser "$REASONING_PARSER" \
  --default-chat-template-kwargs "$DEFAULT_CHAT_TEMPLATE_KWARGS" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype "$DTYPE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
