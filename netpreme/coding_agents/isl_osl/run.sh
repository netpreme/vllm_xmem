#!/usr/bin/env bash
# Single entrypoint: starts the logging proxy, fetches a SWE-bench dataset,
# runs `claude -p` once per problem through the proxy, then runs analyze.sh.
#
# Examples:
#   ./run.sh                                   # vllm × Verified (.env defaults)
#   ./run.sh --dataset pro
#   ./run.sh --limit 50
#   ./run.sh --backend anthropic               # claude OAuth → api.anthropic.com
#   ./run.sh --backend anthropic --model claude-opus-4-7
#   ./run.sh --no-analysis
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
[[ -f "$ROOT/.env" ]] && { set -a; source "$ROOT/.env"; set +a; }

# Defaults (also applied if .env is absent).
: "${MODEL_NAME:=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8}"
: "${CLAUDE_MAX_TURNS:=999}"
: "${CLAUDE_TIMEOUT_SECS:=86400}"
: "${MAX_TOKENS_CAP:=4096}"

# --- flags ------------------------------------------------------------------
BACKEND="vllm"
SWE_DATASET="princeton-nlp/SWE-bench_Verified"
SWE_LIMIT=500
RUN_ANALYSIS=1
MODEL_FLAG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)     BACKEND="$2"; shift 2 ;;
        --dataset)
            case "$2" in
                verified) SWE_DATASET="princeton-nlp/SWE-bench_Verified" ;;
                pro)      SWE_DATASET="ScaleAI/SWE-bench_Pro" ;;
                *) echo "unknown --dataset: $2" >&2; exit 2 ;;
            esac; shift 2 ;;
        --limit)       SWE_LIMIT="$2"; shift 2 ;;
        --model)       MODEL_FLAG="$2"; shift 2 ;;
        --no-analysis) RUN_ANALYSIS=0; shift ;;
        -h|--help)     sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done
case "$BACKEND" in vllm|anthropic) ;; *) echo "bad --backend: $BACKEND" >&2; exit 2 ;; esac
# Anthropic backend ignores the .env MODEL_NAME (Qwen HF id) — default to
# claude-opus-4-7 unless --model was explicitly passed.
if [[ -n "$MODEL_FLAG" ]]; then
    MODEL_NAME="$MODEL_FLAG"
elif [[ "$BACKEND" == "anthropic" ]]; then
    MODEL_NAME="claude-opus-4-7"
fi

# --- upstream ---------------------------------------------------------------
PROXY_URL="http://localhost:9001"
if [[ "$BACKEND" == "anthropic" ]]; then
    UPSTREAM_URL="https://api.anthropic.com"
    BEARER=$(/root/vllm_xmem/.venv/bin/python3 -c "import json; \
print(json.load(open('/root/.claude/.credentials.json'))['claudeAiOauth']['accessToken'])")
    export ANTHROPIC_BASE_URL="$PROXY_URL" ANTHROPIC_AUTH_TOKEN="$BEARER"
else
    UPSTREAM_URL="http://localhost:8000"
    if ! curl -fsS "$UPSTREAM_URL/v1/models" >/dev/null 2>&1; then
        echo "[run] vllm unreachable at $UPSTREAM_URL — start it: bash $ROOT/server.sh" >&2
        exit 1
    fi
fi

VENV_PY=/root/vllm_xmem/.venv/bin/python3
[[ -x "$VENV_PY" ]] || VENV_PY=python3

# --- run dir + config snapshot ----------------------------------------------
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$HERE/runs/$STAMP"
PER_PROBLEM_CSV_DIR="$RUN_DIR/per_problem"
BODIES_DIR="$RUN_DIR/bodies"
WORKDIRS="/tmp/swe_workdirs/$STAMP"
mkdir -p "$WORKDIRS" "$PER_PROBLEM_CSV_DIR" "$BODIES_DIR"
PROBLEMS="$RUN_DIR/problems.jsonl"
SOLVED="$RUN_DIR/solved.txt"; : >"$SOLVED"
echo "[run] writing to $RUN_DIR"

BACKEND="$BACKEND" STAMP="$STAMP" MODEL_NAME="$MODEL_NAME" \
SWE_DATASET="$SWE_DATASET" SWE_LIMIT="$SWE_LIMIT" \
CLAUDE_MAX_TURNS="$CLAUDE_MAX_TURNS" CLAUDE_TIMEOUT_SECS="$CLAUDE_TIMEOUT_SECS" \
MAX_TOKENS_CAP="$MAX_TOKENS_CAP" UPSTREAM_URL="$UPSTREAM_URL" \
    "$VENV_PY" - "$RUN_DIR/config.json" <<'PY'
import json, os, platform, socket, subprocess, sys
def gpu():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
        if not out: return None
        name, mem = (x.strip() for x in out[0].split(",", 1))
        return {"name": name, "count": len(out), "memory_per_gpu": mem}
    except Exception:
        return None
cfg = {
    "run_id":   os.environ["STAMP"],
    "agent":    "claude",
    "backend":  os.environ["BACKEND"],
    "machine":  {"hostname": socket.gethostname(),
                 "platform": f"{platform.system()} {platform.release()}",
                 "gpu": gpu() if os.environ["BACKEND"] == "vllm" else None},
    "model":    os.environ["MODEL_NAME"],
    "dataset":  {"name": os.environ["SWE_DATASET"],
                 "limit": int(os.environ["SWE_LIMIT"])},
    "agent_settings": {
        "claude_max_turns":   int(os.environ["CLAUDE_MAX_TURNS"]),
        "claude_timeout_secs": int(os.environ["CLAUDE_TIMEOUT_SECS"]),
        "max_tokens_cap":     int(os.environ["MAX_TOKENS_CAP"]),
    },
}
json.dump(cfg, open(sys.argv[1], "w"), indent=2)
print("[run] resolved config:")
for k in ("run_id","agent","backend","model"): print(f"  {k:<14} {cfg[k]}")
g = cfg["machine"]["gpu"]
print(f"  gpu            {g['name']} × {g['count']} ({g['memory_per_gpu']})" if g else "  gpu            n/a")
print(f"  dataset        {cfg['dataset']['name']} (limit={cfg['dataset']['limit']})")
print(f"  upstream       {os.environ['UPSTREAM_URL']}")
PY

# --- proxy ------------------------------------------------------------------
PROXY_PID=""
cleanup() {
    [[ -n "${PROXY_PID:-}" ]] && kill -0 "$PROXY_PID" 2>/dev/null \
        && { kill "$PROXY_PID" 2>/dev/null; wait "$PROXY_PID" 2>/dev/null || true; }
}
trap cleanup EXIT

start_proxy() {
    PROXY_AUTH=()
    [[ "$BACKEND" == "anthropic" ]] && PROXY_AUTH=( --passthrough-auth )
    "$VENV_PY" "$HERE/pipeline/proxy.py" \
        --upstream "$UPSTREAM_URL" --port 9001 \
        --per-problem-csv-dir "$PER_PROBLEM_CSV_DIR" \
        --dump-bodies-dir "$BODIES_DIR" \
        --max-tokens-cap "$MAX_TOKENS_CAP" \
        "${PROXY_AUTH[@]}" >/dev/null 2>&1 &
    PROXY_PID=$!
    for _ in $(seq 1 30); do
        curl -fsS "$PROXY_URL/health" >/dev/null 2>&1 && return 0
        kill -0 "$PROXY_PID" 2>/dev/null || { echo "[run] proxy exited" >&2; exit 1; }
        sleep 1
    done
    echo "[run] proxy /health timed out" >&2; exit 1
}
echo "[run] starting proxy on $PROXY_URL  →  $UPSTREAM_URL"
start_proxy

# --- dataset ----------------------------------------------------------------
"$VENV_PY" "$HERE/pipeline/fetch_dataset.py" \
    --dataset "$SWE_DATASET" --limit "$SWE_LIMIT" --out "$PROBLEMS"
TOTAL=$(wc -l < "$PROBLEMS")
echo "[run] $TOTAL problems queued"

# --- solve loop -------------------------------------------------------------
i=0
while IFS= read -r line; do
    i=$((i + 1))
    instance_id=$(printf '%s' "$line" | "$VENV_PY" -c 'import sys,json; print(json.loads(sys.stdin.read())["instance_id"])')
    repo=$(printf '%s' "$line" | "$VENV_PY" -c 'import sys,json; print(json.loads(sys.stdin.read())["repo"])')
    base_commit=$(printf '%s' "$line" | "$VENV_PY" -c 'import sys,json; print(json.loads(sys.stdin.read())["base_commit"])')
    problem=$(printf '%s' "$line" | "$VENV_PY" -c 'import sys,json; print(json.loads(sys.stdin.read())["problem_statement"])')
    echo "[run] [$i/$TOTAL] $instance_id ($repo @ ${base_commit:0:8})"

    if [[ "$BACKEND" == "vllm" ]]; then
        # Cold-start vLLM so each problem sees an empty prefix cache, then
        # restart the proxy (whose connection pool went stale with vllm).
        bash "$HERE/pipeline/reset_vllm.sh"
        kill "$PROXY_PID" 2>/dev/null; wait "$PROXY_PID" 2>/dev/null || true
        start_proxy
    fi

    if "$VENV_PY" "$HERE/pipeline/run_one.py" \
            --instance-id "$instance_id" --repo "$repo" \
            --base-commit "$base_commit" --problem-statement "$problem" \
            --model "$MODEL_NAME" --workdir-root "$WORKDIRS" \
            --max-turns "$CLAUDE_MAX_TURNS" --timeout-secs "$CLAUDE_TIMEOUT_SECS" \
            --base-url "$PROXY_URL" >/dev/null 2>/dev/null
    then echo "$instance_id" >> "$SOLVED"
    else echo "[run]   ! failed (exit $?), continuing"
    fi
done < "$PROBLEMS"

echo
echo "[run] done. results at $RUN_DIR"
echo "[run]   solved.txt: $(wc -l <"$SOLVED") of $TOTAL"
echo "[run]   per-problem CSVs: $(ls "$PER_PROBLEM_CSV_DIR" | wc -l)"

[[ "$RUN_ANALYSIS" -eq 1 ]] && { echo; bash "$HERE/analyze.sh" "$RUN_DIR"; }
