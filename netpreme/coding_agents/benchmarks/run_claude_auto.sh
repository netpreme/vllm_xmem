#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
#  Run Claude Code autonomously against the local vLLM server.
#
#  Requires the server to already be running:
#    ./start_vllm_server.sh
#
#  Usage:
#    ./run_claude_auto.sh "please generate me transformers code"
#
#  Claude Code runs non-interactively (-p) with all tool-call
#  permission prompts suppressed (--dangerously-skip-permissions),
#  driving itself turn-by-turn until the task is complete.
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/../.env"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="$(command -v python3 2>/dev/null || true)"
fi

if [[ -z "$PYTHON_BIN" ]]; then
    echo "ERROR: No Python interpreter found." >&2
    exit 1
fi

# ── Load .env ────────────────────────────────────────────────────
if [[ -f "$ENV_FILE" ]]; then
    while IFS='=' read -r key value; do
        [[ -z "$key" || "$key" == \#* ]] && continue
        [[ -v "$key" ]] || export "$key=$value"
    done < "$ENV_FILE"
else
    echo "WARNING: .env not found at $ENV_FILE, using built-in defaults." >&2
fi

MODEL="${MODEL:-qwen/qwen3-coder-30b-a3b-instruct-fp8}"
PORT="${PORT:-8000}"
BASE_URL="http://localhost:${PORT}"

# ── Require a task prompt ────────────────────────────────────────
if [[ $# -eq 0 ]]; then
    echo "ERROR: No task provided." >&2
    echo "Usage: $0 [--start] \"your task here\"" >&2
    exit 1
fi

TASK="$1"

# ── Wait for health ───────────────────────────────────────────────
echo "Waiting for vLLM at ${BASE_URL}/health ..."
until curl -sf "$BASE_URL/health" > /dev/null 2>&1; do
    printf "."
    sleep 2
done
echo " ready!"
echo ""

# ── Model name validation ────────────────────────────────────────
available=$(curl -s "$BASE_URL/v1/models" | "$PYTHON_BIN" -c "
import json, sys
try:
    data = json.load(sys.stdin)
    models = [m['id'] for m in data.get('data', [])]
    print('\n'.join(models))
except Exception:
    pass
" 2>/dev/null || true)

if [[ -z "$available" ]]; then
    echo "WARNING: Could not fetch model list from $BASE_URL/v1/models"
elif ! echo "$available" | grep -qxF "$MODEL"; then
    echo "ERROR: Model \"$MODEL\" not found in vLLM."
    echo "Available models:"
    echo "$available" | sed 's/^/  /'
    echo "Update MODEL= in .env to match one of the above."
    exit 1
fi

echo "Running autonomous Claude Code"
echo "  ANTHROPIC_BASE_URL = $BASE_URL"
echo "  model              = $MODEL"
echo "  task               = $TASK"
echo ""

exec env \
    ANTHROPIC_BASE_URL="$BASE_URL" \
    ANTHROPIC_API_KEY="dummy" \
    ANTHROPIC_AUTH_TOKEN="dummy" \
    ANTHROPIC_DEFAULT_OPUS_MODEL="$MODEL" \
    ANTHROPIC_DEFAULT_SONNET_MODEL="$MODEL" \
    ANTHROPIC_DEFAULT_HAIKU_MODEL="$MODEL" \
    claude --model "$MODEL" \
           --dangerously-skip-permissions \
           --verbose \
           -p "$TASK"
