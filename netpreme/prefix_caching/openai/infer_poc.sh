#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./infer_poc.sh
#   PROMPT="Write a haiku about Linux." ./infer_poc.sh

API_BASE="${API_BASE:-http://127.0.0.1:8000}"
MODEL="${MODEL:-Qwen/Qwen3.5-0.8B}"
PROMPT="${PROMPT:-Give me 3 concise bullet points on why quantization helps LLM inference.}"

curl -s "$API_BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":512,\"chat_template_kwargs\":{\"enable_thinking\":false}}"
