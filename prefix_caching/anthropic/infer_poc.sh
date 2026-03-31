#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./infer_poc.sh
#   PROMPT="Write a haiku about Linux." ./infer_poc.sh

API_BASE="${API_BASE:-http://127.0.0.1:8000}"
MODEL="${MODEL:-Qwen/Qwen2.5-Coder-32B-Instruct}"
PROMPT="${PROMPT:-Give me 3 concise bullet points on why quantization helps LLM inference.}"

curl -s "$API_BASE/v1/messages" \
  -H "Content-Type: application/json" \
  -H "x-api-key: dummy" \
  -H "anthropic-version: 2023-06-01" \
  -d "{
    \"model\": \"$MODEL\",
    \"max_tokens\": 512,
    \"messages\": [
      {\"role\": \"user\", \"content\": \"$PROMPT\"}
    ]
  }"
