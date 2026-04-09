#!/usr/bin/env bash
set -euo pipefail

# Launch Claude Code pointed at the local vLLM server.
# The server must already be running via ./launch_vllm.sh, or pass --start to
# launch it in the background first.
#
# Usage:
#   ./run_local.sh              # server already running
#   ./run_local.sh --start      # start server in background, then open claude

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

# Load .env
if [[ -f "$ENV_FILE" ]]; then
    while IFS='=' read -r key value; do
        [[ -z "$key" || "$key" == \#* ]] && continue
        [[ -v "$key" ]] || export "$key=$value"
    done < "$ENV_FILE"
else
    echo "WARNING: .env not found, using built-in defaults." >&2
fi

MODEL="${MODEL:-Qwen/Qwen2.5-Coder-32B-Instruct}"
PORT="${PORT:-8000}"
BASE_URL="http://localhost:${PORT}"

# Optionally start the server in the background
if [[ "${1:-}" == "--start" ]]; then
    echo "Starting vLLM server in background..."
    bash "${SCRIPT_DIR}/launch_vllm.sh" &
    SERVER_PID=$!
    echo "  server pid: $SERVER_PID"
fi

# Wait for the server to be ready
echo "Waiting for server at ${BASE_URL} ..."
python3 "${SCRIPT_DIR}/healthcheck.py" --base-url "$BASE_URL"

echo ""
echo "Launching Claude Code"
echo "  ANTHROPIC_BASE_URL = $BASE_URL"
echo "  ANTHROPIC_MODEL    = $MODEL"
echo ""

exec env \
    ANTHROPIC_BASE_URL="$BASE_URL" \
    ANTHROPIC_API_KEY="dummy" \
    ANTHROPIC_MODEL="$MODEL" \
    claude "$@"
