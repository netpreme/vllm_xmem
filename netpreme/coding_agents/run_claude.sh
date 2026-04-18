#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
#  Launch Claude Code pointed at the local vLLM / Dynamo server.
#
#  Requires the server to already be running (or use --start):
#    ./launch_dynamo.sh
#
#  Usage:
#    ./run_claude.sh                          # interactive session
#    ./run_claude.sh --start                  # start server first, then interactive
#    ./run_claude.sh -- --resume              # pass flags through to claude
#    ./run_claude.sh --auto "fix the bug"     # autonomous: run task, pipe to parse_stream.py
#    ./run_claude.sh --start --auto "task"    # start server + autonomous
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

# ── Parse flags ──────────────────────────────────────────────────
START_SERVER=0
AUTO_TASK=""
CLAUDE_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --start)
            START_SERVER=1
            shift
            ;;
        --auto)
            if [[ $# -lt 2 || -z "$2" ]]; then
                echo "ERROR: --auto requires a task argument." >&2
                exit 1
            fi
            AUTO_TASK="$2"
            shift 2
            ;;
        --)
            shift
            CLAUDE_ARGS+=("$@")
            break
            ;;
        *)
            CLAUDE_ARGS+=("$1")
            shift
            ;;
    esac
done

# ── Optionally start the server ──────────────────────────────────
if [[ "$START_SERVER" == "1" ]]; then
    echo "Starting vLLM server in background..."
    bash "${SCRIPT_DIR}/start_vllm_server.sh" &
    echo "  server pid: $!"
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

# ── Launch ───────────────────────────────────────────────────────
COMMON_ENV=(
    ANTHROPIC_BASE_URL="$BASE_URL"
    ANTHROPIC_API_KEY="dummy"
    ANTHROPIC_AUTH_TOKEN="dummy"
    ANTHROPIC_DEFAULT_OPUS_MODEL="$MODEL"
    ANTHROPIC_DEFAULT_SONNET_MODEL="$MODEL"
    ANTHROPIC_DEFAULT_HAIKU_MODEL="$MODEL"
)

if [[ -n "$AUTO_TASK" ]]; then
    echo "Running autonomous Claude Code"
    echo "  ANTHROPIC_BASE_URL = $BASE_URL"
    echo "  model              = $MODEL"
    echo "  task               = $AUTO_TASK"
    echo ""
    env "${COMMON_ENV[@]}" \
        claude --model "$MODEL" \
               --dangerously-skip-permissions \
               --verbose \
               --output-format stream-json \
               -p "$AUTO_TASK" \
      | python3 "${SCRIPT_DIR}/parse_stream.py"
else
    echo "Launching Claude Code"
    echo "  ANTHROPIC_BASE_URL = $BASE_URL"
    echo "  model              = $MODEL"
    echo ""
    exec env "${COMMON_ENV[@]}" \
        claude --model "$MODEL" "${CLAUDE_ARGS[@]}"
fi
