#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
#  Launch Claude Code pointed at the local Dynamo + vLLM stack.
#
#  Requires the stack to already be running:  ./launch_dynamo.sh
#
#  What this script does:
#    1. Loads MODEL and DYNAMO_PORT from .env (same values used
#       by launch_dynamo.sh, so they stay in sync).
#    2. Waits for Dynamo's /health endpoint to be ready.
#    3. Calls GET /v1/models and verifies that $MODEL is actually
#       registered — Dynamo silently 404s if the model name sent
#       by Claude Code doesn't exactly match what the vLLM worker
#       registered on startup (e.g. case or path differences).
#    4. Launches Claude Code with:
#         ANTHROPIC_BASE_URL  — points Claude at local Dynamo
#                               instead of api.anthropic.com
#         ANTHROPIC_API_KEY   — Dynamo doesn't validate the key,
#                               "dummy" is intentional
#         ANTHROPIC_AUTH_TOKEN — Claude Code expects it in normal
#                               mode; "dummy" keeps auth local
#         ANTHROPIC_DEFAULT_*  — route Claude Code's tiered model
#                               lookups to the same local model
#         --model $MODEL      — must be a CLI flag, not an env var;
#                               ANTHROPIC_MODEL is silently ignored
#                               by Claude Code
#
#  Usage:  ./run_claude_local.sh [claude flags...]
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="$(command -v python3)"
fi

if [[ -z "$PYTHON_BIN" ]]; then
    echo "ERROR: No Python interpreter found." >&2
    exit 1
fi

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
PORT="${DYNAMO_PORT:-8000}"
BASE_URL="http://localhost:${PORT}"

# Wait for Dynamo to be ready
echo "Waiting for Dynamo at ${BASE_URL} ..."
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
    echo "ERROR: Model \"$MODEL\" not found in Dynamo."
    echo "Available models:"
    echo "$available" | sed 's/^/  /'
    echo "Update MODEL= in .env to match one of the above."
    exit 1
fi

echo "Launching Claude Code"
echo "  ANTHROPIC_BASE_URL = $BASE_URL"
echo "  model flag         = $MODEL"
echo ""

exec env \
    ANTHROPIC_BASE_URL="$BASE_URL" \
    ANTHROPIC_API_KEY="dummy" \
    ANTHROPIC_AUTH_TOKEN="dummy" \
    ANTHROPIC_DEFAULT_OPUS_MODEL="$MODEL" \
    ANTHROPIC_DEFAULT_SONNET_MODEL="$MODEL" \
    ANTHROPIC_DEFAULT_HAIKU_MODEL="$MODEL" \
    claude --model "$MODEL" "$@"
