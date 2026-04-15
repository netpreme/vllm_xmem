#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
#  Launch Claude Code pointed at the local vLLM server.
#
#  Requires the server to already be running:
#    ./start_vllm_server.sh
#
#  What this script does:
#    1. Loads MODEL and PORT from .env (same values used by
#       start_vllm_server.sh, so they stay in sync).
#    2. Waits for vLLM /health to be ready.
#    3. Calls GET /v1/models and verifies $MODEL is registered.
#       Claude Code 404s silently if the model name doesn't exactly
#       match what vLLM registered on startup.
#    4. Launches Claude Code with:
#         ANTHROPIC_BASE_URL      — points at the local vLLM server
#         ANTHROPIC_API_KEY       — vLLM doesn't validate; "dummy" is fine
#         ANTHROPIC_AUTH_TOKEN    — Claude Code's normal-mode auth token;
#                                   "dummy" keeps auth local
#         ANTHROPIC_DEFAULT_*     — map all tiered model lookups
#                                   (opus/sonnet/haiku) to $MODEL so
#                                   Claude Code's internal routing resolves
#         --model $MODEL          — must be a CLI flag; ANTHROPIC_MODEL is
#                                   silently ignored by Claude Code
#
#  Note on CLAUDE_CODE_ATTRIBUTION_HEADER:
#    When using plain vLLM (not Dynamo), Claude Code's per-request billing
#    header is automatically stripped by vLLM's Anthropic endpoint before
#    tokenization, so prefix caching is not disrupted. No need to set
#    CLAUDE_CODE_ATTRIBUTION_HEADER=0 here (unlike the Dynamo setup).
#
#  Usage:
#    CLAUDE_CODE_ATTRIBUTION_HEADER=0 ./run_claude_local.sh              # server already running
#    CLAUDE_CODE_ATTRIBUTION_HEADER=0 ./run_claude_local.sh --start      # start server in background first
#    CLAUDE_CODE_ATTRIBUTION_HEADER=0 ./run_claude_local.sh -- --resume  # pass flags through to claude
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

MODEL="${MODEL:-Qwen/Qwen2.5-Coder-32B-Instruct}"
PORT="${PORT:-8000}"
BASE_URL="http://localhost:${PORT}"

# ── Optionally start the server in the background ───────────────
if [[ "${1:-}" == "--start" ]]; then
    shift
    echo "Starting vLLM server in background..."
    bash "${SCRIPT_DIR}/start_vllm_server.sh" &
    SERVER_PID=$!
    echo "  server pid: $SERVER_PID"
fi

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

echo "Launching Claude Code"
echo "  ANTHROPIC_BASE_URL = $BASE_URL"
echo "  model flag         = $MODEL"
echo ""

# Pass any remaining args (after optional --start) through to claude.
# Separate with -- to allow: ./run_claude_local.sh -- --resume
CLAUDE_ARGS=()
for arg in "$@"; do
    [[ "$arg" == "--" ]] && continue
    CLAUDE_ARGS+=("$arg")
done

exec env \
    ANTHROPIC_BASE_URL="$BASE_URL" \
    ANTHROPIC_API_KEY="dummy" \
    ANTHROPIC_AUTH_TOKEN="dummy" \
    ANTHROPIC_DEFAULT_OPUS_MODEL="$MODEL" \
    ANTHROPIC_DEFAULT_SONNET_MODEL="$MODEL" \
    ANTHROPIC_DEFAULT_HAIKU_MODEL="$MODEL" \
    claude --model "$MODEL" "${CLAUDE_ARGS[@]}"
