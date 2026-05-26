#!/usr/bin/env bash
# End-to-end orchestrator for one ISL/OSL benchmarking run.
#
# Flow per problem:
#   1. cold-restart vLLM so each problem starts with an empty prefix cache
#   2. write the current instance_id to the watcher's control file
#   3. invoke claude-cli pointed straight at vLLM
#   4. claude makes N internal /v1/messages calls; the background metrics
#      watcher detects N completions in vLLM's Prometheus counters and
#      writes one CSV row per turn into runs/<stamp>/per_problem/<id>.csv
#   5. clear the control file when claude exits
#
# When all problems are done analyze.sh consolidates the CSVs into
# data.npz and renders the figures.
#
# Usage:
#   ./run.sh                              # vllm × Verified × 500 (defaults)
#   ./run.sh --dataset pro
#   ./run.sh --limit 50
#   ./run.sh --random 100 --seed 0
#   ./run.sh --no-analysis
set -euo pipefail


# ---------------------------------------------------------------------------
# Resolve paths, source the environment, apply defaults.
# ---------------------------------------------------------------------------

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

if [[ -f "$ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$ROOT/.env"
    set +a
fi

: "${MODEL_NAME:=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8}"
: "${CLAUDE_MAX_TURNS:=999}"
: "${CLAUDE_TIMEOUT_SECS:=86400}"

VLLM_URL="http://localhost:8000"
LABELER_URL="http://127.0.0.1:8001"   # agent_labeler proxy, sits in front of vLLM


# ---------------------------------------------------------------------------
# CLI flag parsing.
# ---------------------------------------------------------------------------

SWE_DATASET="princeton-nlp/SWE-bench_Verified"
SWE_LIMIT=500           # take first N rows of the dataset
SWE_RANDOM=0            # >0 = uniformly random sample, overrides SWE_LIMIT
SWE_SEED=0
RUN_ANALYSIS=1
MODEL_FLAG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            case "$2" in
                verified) SWE_DATASET="princeton-nlp/SWE-bench_Verified" ;;
                pro)      SWE_DATASET="ScaleAI/SWE-bench_Pro" ;;
                *) echo "unknown --dataset: $2" >&2; exit 2 ;;
            esac
            shift 2 ;;
        --limit)       SWE_LIMIT="$2";  shift 2 ;;
        --random)      SWE_RANDOM="$2"; shift 2 ;;
        --seed)        SWE_SEED="$2";   shift 2 ;;
        --model)       MODEL_FLAG="$2"; shift 2 ;;
        --no-analysis) RUN_ANALYSIS=0;  shift   ;;
        -h|--help)     sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

[[ "$SWE_RANDOM" -gt 0 ]] && SWE_LIMIT="$SWE_RANDOM"
[[ -n "$MODEL_FLAG"   ]] && MODEL_NAME="$MODEL_FLAG"

if ! curl -fsS "$VLLM_URL/v1/models" >/dev/null 2>&1; then
    echo "[run] vllm is not reachable at $VLLM_URL — start it first:" >&2
    echo "       bash $ROOT/server.sh" >&2
    exit 1
fi

PY=/root/vllm_xmem/.venv/bin/python3
[[ -x "$PY" ]] || PY=python3


# ---------------------------------------------------------------------------
# Create the run directory and write config.json.
# ---------------------------------------------------------------------------

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$HERE/runs/$STAMP"
CSV_DIR="$RUN_DIR/per_problem"
WORKDIRS="/tmp/swe_workdirs/$STAMP"
PROBLEMS="$RUN_DIR/problems.jsonl"
SOLVED="$RUN_DIR/solved.txt"
CONTROL_FILE="$RUN_DIR/.active_instance"
LABELS_FILE="$RUN_DIR/.agent_labels"

mkdir -p "$CSV_DIR" "$WORKDIRS"
: >"$SOLVED"
: >"$CONTROL_FILE"

echo "[run] writing to $RUN_DIR"

STAMP="$STAMP" MODEL_NAME="$MODEL_NAME" \
SWE_DATASET="$SWE_DATASET" SWE_LIMIT="$SWE_LIMIT" \
CLAUDE_MAX_TURNS="$CLAUDE_MAX_TURNS" CLAUDE_TIMEOUT_SECS="$CLAUDE_TIMEOUT_SECS" \
VLLM_URL="$VLLM_URL" \
    "$PY" - "$RUN_DIR/config.json" <<'PY'
import json, os, platform, socket, subprocess, sys
def gpu_info():
    try:
        lines = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip().splitlines()
        if not lines:
            return None
        name, mem = (x.strip() for x in lines[0].split(",", 1))
        return {"name": name, "count": len(lines), "memory_per_gpu": mem}
    except Exception:
        return None

cfg = {
    "run_id":  os.environ["STAMP"],
    "agent":   "claude",
    "backend": "vllm",
    "machine": {
        "hostname": socket.gethostname(),
        "platform": f"{platform.system()} {platform.release()}",
        "gpu":      gpu_info(),
    },
    "model":   os.environ["MODEL_NAME"],
    "dataset": {
        "name":  os.environ["SWE_DATASET"],
        "limit": int(os.environ["SWE_LIMIT"]),
    },
    "agent_settings": {
        "claude_max_turns":    int(os.environ["CLAUDE_MAX_TURNS"]),
        "claude_timeout_secs": int(os.environ["CLAUDE_TIMEOUT_SECS"]),
    },
}

with open(sys.argv[1], "w") as fh:
    json.dump(cfg, fh, indent=2)

print("[run] resolved config:")
for k in ("run_id", "agent", "backend", "model"):
    print(f"  {k:<14} {cfg[k]}")
g = cfg["machine"]["gpu"]
print(f"  gpu            "
      f"{g['name']} × {g['count']} ({g['memory_per_gpu']})" if g else "  gpu            n/a")
print(f"  dataset        {cfg['dataset']['name']} (limit={cfg['dataset']['limit']})")
print(f"  upstream       {os.environ['VLLM_URL']}")
PY


# ---------------------------------------------------------------------------
# Background processes.
#
# Two long-lived sidecars run for the whole benchmark:
#   * agent_labeler  — pass-through proxy in front of vLLM. Classifies each
#                      /v1/messages call (main vs sub agent) and appends a
#                      label record to .agent_labels.
#   * metrics_watcher — polls vLLM /metrics every 100 ms. On each detected
#                      completion it pops the matching label record and
#                      writes one CSV row to per_problem/<iid>.csv.
# Both are killed in the cleanup trap below.
# ---------------------------------------------------------------------------

LABELER_PID=""
WATCHER_PID=""
cleanup() {
    for proc in "watcher:$WATCHER_PID" "labeler:$LABELER_PID"; do
        name="${proc%%:*}"
        pid="${proc#*:}"
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            echo "[run] stopping $name (pid $pid)"
            kill "$pid"            2>/dev/null || true
            wait "$pid"            2>/dev/null || true
        fi
    done
}
trap cleanup EXIT

start_labeler() {
    "$PY" "$HERE/pipeline/agent_labeler.py" \
        --upstream "$VLLM_URL" \
        --labels-file "$LABELS_FILE" \
        --listen-port 8001 \
        >"$RUN_DIR/.labeler.log" 2>&1 &
    LABELER_PID=$!
    # Wait for the labeler to start listening before we let claude hit it.
    for _ in $(seq 1 50); do
        curl -fsS "$LABELER_URL/v1/models" >/dev/null 2>&1 && return 0
        sleep 0.1
    done
    echo "[run] labeler failed to start; see $RUN_DIR/.labeler.log" >&2
    exit 1
}

start_watcher() {
    "$PY" "$HERE/pipeline/metrics_watcher.py" \
        --vllm-url "$VLLM_URL" \
        --control-file "$CONTROL_FILE" \
        --per-problem-csv-dir "$CSV_DIR" \
        --labels-file "$LABELS_FILE" \
        >"$RUN_DIR/.watcher.log" 2>&1 &
    WATCHER_PID=$!
}

echo "[run] starting agent labeler on $LABELER_URL → $VLLM_URL"
start_labeler
echo "[run] starting metrics watcher on $VLLM_URL/metrics"
start_watcher


# ---------------------------------------------------------------------------
# Fetch problems.
# ---------------------------------------------------------------------------

FETCH_ARGS=( --dataset "$SWE_DATASET" --out "$PROBLEMS" )
if [[ "$SWE_RANDOM" -gt 0 ]]; then
    FETCH_ARGS+=( --random "$SWE_RANDOM" --seed "$SWE_SEED" )
else
    FETCH_ARGS+=( --limit "$SWE_LIMIT" )
fi
"$PY" "$HERE/pipeline/fetch_dataset.py" "${FETCH_ARGS[@]}"

TOTAL=$(wc -l < "$PROBLEMS")
echo "[run] $TOTAL problems queued"


# ---------------------------------------------------------------------------
# Solve loop.
#
# For each problem we:
#   1. cold-restart vLLM (each problem sees an empty prefix cache);
#   2. truncate .agent_labels so labels don't carry across the restart
#      (the watcher's LabelReader notices the shrink and rewinds);
#   3. update the watcher's control file so subsequent completions get
#      attributed to this instance_id;
#   4. sleep briefly so the watcher has at least one post-reset scrape;
#   5. invoke claude-cli pointed at the agent_labeler proxy (which
#      forwards to vLLM and writes labels);
#   6. sleep briefly so the watcher catches the final completion;
#   7. clear the control file.
# ---------------------------------------------------------------------------

extract() {
    # Pull a single JSON field out of a one-line problem record.
    printf '%s' "$1" | "$PY" -c "import sys, json; print(json.loads(sys.stdin.read())[\"$2\"])"
}

i=0
while IFS= read -r line; do
    i=$((i + 1))
    iid=$(extract         "$line" "instance_id")
    repo=$(extract        "$line" "repo")
    base_commit=$(extract "$line" "base_commit")
    problem=$(extract     "$line" "problem_statement")

    echo "[run] [$i/$TOTAL] $iid ($repo @ ${base_commit:0:8})"

    # 1. Cold prefix cache. reset_vllm.sh kills vllm and waits for it to
    #    come back, which (a) clears the kv cache and (b) zeroes the
    #    Prometheus counters. The watcher's baseline is rediscovered on
    #    its next scrape (it tolerates an HTTPError mid-restart).
    bash "$HERE/pipeline/reset_vllm.sh"

    # 2. Reset the label queue. The LabelReader rewinds automatically when
    #    the file shrinks, so any leftover lines from the previous problem
    #    are dropped.
    : >"$LABELS_FILE"

    # 3-4. Update the control file and give the watcher a tick to read
    #      it before claude's first request lands.
    printf '%s' "$iid" >"$CONTROL_FILE"
    sleep 0.3

    # 5. Run the agent against the labeler (which forwards to vLLM).
    if "$PY" "$HERE/pipeline/solve_problem.py" \
            --instance-id        "$iid" \
            --repo               "$repo" \
            --base-commit        "$base_commit" \
            --problem-statement  "$problem" \
            --model              "$MODEL_NAME" \
            --workdir-root       "$WORKDIRS" \
            --max-turns          "$CLAUDE_MAX_TURNS" \
            --timeout-secs       "$CLAUDE_TIMEOUT_SECS" \
            --vllm-url           "$LABELER_URL" \
            --per-problem-dir    "$CSV_DIR" >/dev/null 2>/dev/null
    then
        echo "$iid" >>"$SOLVED"
    else
        echo "[run]   ! failed (exit $?), continuing"
    fi

    # 6-7. Let the watcher catch the last completion before the
    #      attribution changes.
    sleep 0.3
    : >"$CONTROL_FILE"
done < "$PROBLEMS"


# ---------------------------------------------------------------------------
# Wrap up.
# ---------------------------------------------------------------------------

echo
echo "[run] done. results at $RUN_DIR"
echo "[run]   solved.txt:       $(wc -l <"$SOLVED") of $TOTAL"
echo "[run]   per-problem CSVs: $(ls "$CSV_DIR" | wc -l)"

if [[ "$RUN_ANALYSIS" -eq 1 ]]; then
    echo
    bash "$HERE/analyze.sh" "$RUN_DIR"
fi
