#!/usr/bin/env bash
# Codex variant of run.sh — drives `codex exec` over a SWE-bench-style dataset
# against a local vLLM server. ISL/OSL/cache stats are parsed from
# `codex exec --json` events (turn.completed.usage), so no logging proxy is
# required for codex (vLLM /v1/responses + per-turn usage works directly).
#
# Usage:    bash run_codex.sh
#           SWE_LIMIT=20 SWE_DATASET=ScaleAI/SWE-bench_Pro bash run_codex.sh
#
# Layout (under runs_codex/<timestamp>/):
#   server.log        - vllm server log (only used if we boot it here)
#   problems.jsonl    - dataset rows
#   usage.jsonl       - one row per codex turn (ISL/OSL/category/cache)
#   per_problem/*.json - per-problem run summary
#   summary.json      - aggregate report
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

if [[ -x /root/vllm_xmem/.venv/bin/python3 ]]; then
    VENV_PY=/root/vllm_xmem/.venv/bin/python3
else
    VENV_PY=python3
fi

set -a
# shellcheck disable=SC1091
source "$ROOT/.env"
set +a

: "${SERVED_MODEL_NAME:?}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8000}"
: "${SWE_DATASET:=princeton-nlp/SWE-bench_Verified}"
: "${SWE_SPLIT:=test}"
: "${SWE_LIMIT:=500}"
: "${CLAUDE_MAX_TURNS:=15}"          # we reuse this turn cap for codex
: "${CLAUDE_TIMEOUT_SECS:=600}"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$HERE/runs_codex/$STAMP"
# Workdirs MUST be outside RUN_DIR (parent-dir walking by codex / sandbox).
WORKDIRS="/tmp/swe_workdirs_codex/$STAMP"
mkdir -p "$RUN_DIR" "$RUN_DIR/per_problem" "$WORKDIRS"
PROBLEMS="$RUN_DIR/problems.jsonl"
USAGE="$RUN_DIR/usage.jsonl"
SUMMARY="$RUN_DIR/summary.json"
echo "[run-codex] writing to $RUN_DIR"

# --- 1. confirm vLLM is up ---------------------------------------------------
VLLM_URL="http://localhost:${PORT}"
if ! curl -fsS "$VLLM_URL/v1/models" >/dev/null 2>&1; then
    echo "[run-codex] vllm is not up at $VLLM_URL — start it via server.sh"
    exit 1
fi
echo "[run-codex] vllm reachable at $VLLM_URL"

# --- 2. fetch dataset --------------------------------------------------------
if [[ ! -s "$PROBLEMS" ]]; then
    "$VENV_PY" "$HERE/fetch_dataset.py" \
        --dataset "$SWE_DATASET" --split "$SWE_SPLIT" \
        --limit "$SWE_LIMIT" --out "$PROBLEMS"
fi
TOTAL=$(wc -l < "$PROBLEMS")
echo "[run-codex] $TOTAL problems queued"

# --- 3. solve each problem with codex ----------------------------------------
i=0
while IFS= read -r line; do
    i=$((i + 1))
    instance_id=$(printf '%s' "$line" | "$VENV_PY" -c 'import sys,json; print(json.loads(sys.stdin.read())["instance_id"])')
    repo=$(printf '%s' "$line" | "$VENV_PY" -c 'import sys,json; print(json.loads(sys.stdin.read())["repo"])')
    base_commit=$(printf '%s' "$line" | "$VENV_PY" -c 'import sys,json; print(json.loads(sys.stdin.read())["base_commit"])')
    problem=$(printf '%s' "$line" | "$VENV_PY" -c 'import sys,json; print(json.loads(sys.stdin.read())["problem_statement"])')
    echo "[run-codex] [$i/$TOTAL] $instance_id ($repo @ ${base_commit:0:8})"

    # Cold-start vLLM so each problem sees an empty prefix cache.
    bash "$HERE/reset_vllm.sh"

    if "$VENV_PY" "$HERE/run_one_codex.py" \
        --instance-id "$instance_id" \
        --repo "$repo" \
        --base-commit "$base_commit" \
        --problem-statement "$problem" \
        --model "$SERVED_MODEL_NAME" \
        --base-url "$VLLM_URL/v1" \
        --workdir-root "$WORKDIRS" \
        --usage-out "$USAGE" \
        --max-turns "$CLAUDE_MAX_TURNS" \
        --timeout-secs "$CLAUDE_TIMEOUT_SECS" \
        > "$RUN_DIR/per_problem/${instance_id}.json" 2>>"$RUN_DIR/run_one_codex.err"
    then :
    else echo "[run-codex]   ! failed (exit $?), continuing"
    fi
done < "$PROBLEMS"

# --- 4. summarize ------------------------------------------------------------
"$VENV_PY" "$HERE/summarize.py" --usage "$USAGE" --out "$SUMMARY"
echo
echo "[run-codex] done. results at $RUN_DIR"
