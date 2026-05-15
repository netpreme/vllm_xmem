#!/usr/bin/env bash
# Boots vLLM, runs claude -p over SWE-bench Verified problems, and reports
# ISL/OSL distributions split by assistant-output shape (text/tool/mixed/empty).
#
# Usage:  bash run.sh                     # uses .env defaults
#         SWE_LIMIT=20 bash run.sh        # override per-invocation
#
# Layout (under runs/<timestamp>/):
#   server.log        - vllm stdout/stderr
#   problems.jsonl    - SWE-bench verified rows
#   usage.jsonl       - one row per assistant turn (ISL/OSL/category)
#   solve_<id>.json   - per-problem run summary
#   summary.json      - aggregate report (output of summarize.py)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

# Python interpreter: use the vllm venv if it exists, else system python3.
# The proxy needs httpx/fastapi/uvicorn which only live in the venv.
if [[ -x /root/vllm_xmem/.venv/bin/python3 ]]; then
    VENV_PY=/root/vllm_xmem/.venv/bin/python3
else
    VENV_PY=python3
fi

# Preserve command-line env overrides for any var also set in .env.
__OV_KEYS=( SWE_DATASET SWE_SPLIT SWE_LIMIT CLAUDE_MAX_TURNS CLAUDE_TIMEOUT_SECS MAX_TOKENS_CAP SERVED_MODEL_NAME MODEL_NAME PORT PROXY_PORT HOST )
declare -A __OV
for __k in "${__OV_KEYS[@]}"; do
    if [[ -n "${!__k+set}" ]]; then __OV[$__k]="${!__k}"; fi
done
set -a
# shellcheck disable=SC1091
source "$ROOT/.env"
set +a
for __k in "${!__OV[@]}"; do export "$__k=${__OV[$__k]}"; done

: "${SERVED_MODEL_NAME:?}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8000}"
: "${PROXY_PORT:=9001}"
: "${SWE_DATASET:=princeton-nlp/SWE-bench_Verified}"
: "${SWE_SPLIT:=test}"
: "${SWE_LIMIT:=500}"
: "${CLAUDE_MAX_TURNS:=30}"
: "${CLAUDE_TIMEOUT_SECS:=600}"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$HERE/runs/$STAMP"
mkdir -p "$RUN_DIR"
echo "[run] writing to $RUN_DIR"

PROBLEMS="$RUN_DIR/problems.jsonl"
SOLVED="$RUN_DIR/solved.txt"
PER_PROBLEM_CSV_DIR="$RUN_DIR/per_problem"
# Workdirs MUST be outside RUN_DIR — claude walks up parent dirs from cwd and
# would otherwise read the 8 MB problems.jsonl as part of its context.
WORKDIRS="/tmp/swe_workdirs/$STAMP"
mkdir -p "$WORKDIRS" "$PER_PROBLEM_CSV_DIR"
: >"$SOLVED"

# --- 1. start vLLM (skip if already up on this port) -------------------------
VLLM_URL="http://localhost:${PORT}"
PROXY_URL="http://localhost:${PROXY_PORT}"
VLLM_PID=""
PROXY_PID=""
cleanup() {
    if [[ -n "${PROXY_PID:-}" ]] && kill -0 "$PROXY_PID" 2>/dev/null; then
        echo "[run] stopping proxy (pid $PROXY_PID)"
        kill "$PROXY_PID" 2>/dev/null || true
        wait "$PROXY_PID" 2>/dev/null || true
    fi
    if [[ -n "${VLLM_PID:-}" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[run] stopping vllm (pid $VLLM_PID)"
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if curl -fsS "$VLLM_URL/v1/models" >/dev/null 2>&1; then
    echo "[run] vllm already serving on $VLLM_URL — reusing"
else
    echo "[run] starting vllm (logs: $RUN_DIR/server.log)"
    "$ROOT/server.sh" >"$RUN_DIR/server.log" 2>&1 &
    VLLM_PID=$!

    echo -n "[run] waiting for vllm /v1/models"
    for _ in $(seq 1 240); do
        if curl -fsS "$VLLM_URL/v1/models" >/dev/null 2>&1; then
            echo " — ready"
            break
        fi
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo
            echo "[run] vllm exited; tail of server.log:"
            tail -n 50 "$RUN_DIR/server.log" >&2
            exit 1
        fi
        echo -n "."
        sleep 5
    done
    if ! curl -fsS "$VLLM_URL/v1/models" >/dev/null 2>&1; then
        echo " — timed out"
        exit 1
    fi
fi

# Verify the Anthropic Messages route exists on this build.
if ! curl -fsS -o /dev/null -X POST "$VLLM_URL/v1/messages" \
        -H 'content-type: application/json' \
        -d "{\"model\":\"$SERVED_MODEL_NAME\",\"max_tokens\":1,\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}]}"; then
    echo "[run] WARNING: $VLLM_URL/v1/messages probe failed — check server.log" >&2
fi

# --- 1b. start logging proxy ------------------------------------------------
echo "[run] starting proxy on $PROXY_URL"
"$VENV_PY" "$HERE/proxy.py" \
    --upstream "$VLLM_URL" \
    --port "$PROXY_PORT" \
    --per-problem-csv-dir "$PER_PROBLEM_CSV_DIR" \
    --max-tokens-cap "${MAX_TOKENS_CAP:-4096}" \
    >/dev/null 2>&1 &
PROXY_PID=$!

echo -n "[run] waiting for proxy /health"
for _ in $(seq 1 30); do
    if curl -fsS "$PROXY_URL/health" >/dev/null 2>&1; then
        echo " — ready"
        break
    fi
    if ! kill -0 "$PROXY_PID" 2>/dev/null; then
        echo
        echo "[run] proxy exited; tail of proxy.log:"
        tail -n 30 "$RUN_DIR/proxy.log" >&2
        exit 1
    fi
    echo -n "."
    sleep 1
done
if ! curl -fsS "$PROXY_URL/health" >/dev/null 2>&1; then
    echo " — timed out"
    exit 1
fi

# --- 2. fetch dataset --------------------------------------------------------
if [[ ! -s "$PROBLEMS" ]]; then
    "$VENV_PY" "$HERE/fetch_dataset.py" \
        --dataset "$SWE_DATASET" \
        --split "$SWE_SPLIT" \
        --limit "$SWE_LIMIT" \
        --out "$PROBLEMS"
fi
TOTAL=$(wc -l < "$PROBLEMS")
echo "[run] $TOTAL problems queued"

# --- 3. solve each problem ---------------------------------------------------
i=0
while IFS= read -r line; do
    i=$((i + 1))
    instance_id=$(printf '%s' "$line" | python3 -c 'import sys,json; print(json.loads(sys.stdin.read())["instance_id"])')
    repo=$(printf '%s' "$line" | python3 -c 'import sys,json; print(json.loads(sys.stdin.read())["repo"])')
    base_commit=$(printf '%s' "$line" | python3 -c 'import sys,json; print(json.loads(sys.stdin.read())["base_commit"])')
    problem=$(printf '%s' "$line" | python3 -c 'import sys,json; print(json.loads(sys.stdin.read())["problem_statement"])')

    echo "[run] [$i/$TOTAL] $instance_id ($repo @ ${base_commit:0:8})"

    # Cold-start vLLM so each problem sees an empty prefix cache.
    bash "$HERE/reset_vllm.sh"
    # Restart the proxy too — it kept the upstream connection pool that just
    # went stale when we killed vllm.
    if [[ -n "${PROXY_PID:-}" ]] && kill -0 "$PROXY_PID" 2>/dev/null; then
        kill "$PROXY_PID" 2>/dev/null || true
        wait "$PROXY_PID" 2>/dev/null || true
    fi
    "$VENV_PY" "$HERE/proxy.py" \
        --upstream "$VLLM_URL" \
        --port "$PROXY_PORT" \
        --per-problem-csv-dir "$PER_PROBLEM_CSV_DIR" \
        --max-tokens-cap "${MAX_TOKENS_CAP:-4096}" \
        >/dev/null 2>&1 &
    PROXY_PID=$!
    until curl -fsS "$PROXY_URL/health" >/dev/null 2>&1; do sleep 1; done

    if "$VENV_PY" "$HERE/run_one.py" \
        --instance-id "$instance_id" \
        --repo "$repo" \
        --base-commit "$base_commit" \
        --problem-statement "$problem" \
        --model "$SERVED_MODEL_NAME" \
        --base-url "$PROXY_URL" \
        --workdir-root "$WORKDIRS" \
        --max-turns "$CLAUDE_MAX_TURNS" \
        --timeout-secs "$CLAUDE_TIMEOUT_SECS" \
        >/dev/null 2>/dev/null
    then
        echo "$instance_id" >> "$SOLVED"
    else
        echo "[run]   ! failed (exit $?), continuing"
    fi
done < "$PROBLEMS"

echo
echo "[run] done. results at $RUN_DIR"
echo "[run]   solved.txt: $(wc -l <"$SOLVED") of $TOTAL"
echo "[run]   per-problem CSVs: $(ls "$PER_PROBLEM_CSV_DIR" | wc -l)"
