#!/usr/bin/env bash
# Single entrypoint: boots vLLM, runs claude -p over a SWE-bench dataset,
# writes per-turn telemetry, then runs analysis to produce figures.
#
# Examples:
#   ./run.sh                                   # claude × Verified (defaults from .env)
#   ./run.sh --dataset pro
#   ./run.sh --limit 50
#   ./run.sh --model my-served-name
#   ./run.sh --no-analysis                     # skip analyze.sh at end
#
# Layout (under runs/<timestamp>/):
#   config.json       - resolved run config (model, dataset, machine, etc.)
#   server.log        - vllm stdout/stderr
#   problems.jsonl    - SWE-bench rows
#   usage.jsonl       - one row per assistant turn (ISL/OSL/category/...)
#   per_problem/*.csv - per-problem turn-by-turn CSV (consumed by analyze.sh)
#   solved.txt        - list of solved instance_ids
#   summary.json      - aggregate stats (output of summarize.py)
#   analysis/*.png    - figures from analyze.sh (unless --no-analysis)
set -euo pipefail

# --- flag parsing ------------------------------------------------------------
BACKEND="vllm"
DATASET_FLAG=""
LIMIT_FLAG=""
MODEL_FLAG=""
RUN_ANALYSIS=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)  BACKEND="$2"; shift 2 ;;
        --dataset)  DATASET_FLAG="$2"; shift 2 ;;
        --limit)    LIMIT_FLAG="$2"; shift 2 ;;
        --model)    MODEL_FLAG="$2"; shift 2 ;;
        --no-analysis) RUN_ANALYSIS=0; shift ;;
        -h|--help)  sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done
case "$BACKEND" in
    vllm|anthropic) ;;
    *) echo "unknown --backend (use vllm|anthropic): $BACKEND" >&2; exit 2 ;;
esac
case "${DATASET_FLAG:-}" in
    "")        ;;
    verified)  export SWE_DATASET="princeton-nlp/SWE-bench_Verified" ;;
    pro)       export SWE_DATASET="ScaleAI/SWE-bench_Pro" ;;
    *) echo "unknown --dataset (use verified|pro): $DATASET_FLAG" >&2; exit 2 ;;
esac
[[ -n "$LIMIT_FLAG" ]] && export SWE_LIMIT="$LIMIT_FLAG"
[[ -n "$MODEL_FLAG" ]] && export SERVED_MODEL_NAME="$MODEL_FLAG"

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
# Export so the Python heredoc + run_one.py subprocess inherit them.
export SWE_DATASET SWE_SPLIT SWE_LIMIT CLAUDE_MAX_TURNS CLAUDE_TIMEOUT_SECS \
       MAX_TOKENS_CAP SERVED_MODEL_NAME MODEL_NAME PORT PROXY_PORT HOST \
       MAX_MODEL_LEN TENSOR_PARALLEL_SIZE GPU_MEMORY_UTILIZATION \
       TOOL_CALL_PARSER ANTHROPIC_MODEL

# --- resolve upstream (local vLLM vs Anthropic via claude OAuth) ------------
if [[ "$BACKEND" == "anthropic" ]]; then
    # Route claude → local proxy → api.anthropic.com. The proxy reuses the
    # same SSE-parsing/timing path as the vLLM branch, so per-turn ttft_ms /
    # decode_ms / itl_ms get captured. Authentication is whatever claude
    # already uses (OAuth bearer from `claude login`), passed through.
    : "${ANTHROPIC_MODEL:=claude-opus-4-7}"
    SERVED_MODEL_NAME="$ANTHROPIC_MODEL"
    UPSTREAM_URL="https://api.anthropic.com"
else
    UPSTREAM_URL="http://localhost:${PORT}"
    # vLLM must already be running. Start it via the sibling server script:
    #   bash $ROOT/server.sh > /tmp/vllm.log 2>&1 &
    if ! curl -fsS "$UPSTREAM_URL/v1/models" >/dev/null 2>&1; then
        echo "[run] vllm is not reachable at $UPSTREAM_URL" >&2
        echo "       start it first:  bash $ROOT/server.sh" >&2
        exit 1
    fi
fi
export UPSTREAM_URL

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

# Snapshot resolved config into the run dir so later analysis can recover
# exactly which model / dataset / agent caps produced these CSVs.
AGENT_NAME=claude RUN_ID="$STAMP" BACKEND="$BACKEND" \
    "$VENV_PY" - "$RUN_DIR/config.json" <<'PY'
import json, os, platform, socket, subprocess, sys
def g(k, cast=str, default=None):
    v = os.environ.get(k)
    if v is None or v == "":
        return default
    try: return cast(v)
    except Exception: return v
def gpu_info():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        if not lines: return None
        name, mem = (x.strip() for x in lines[0].split(",", 1))
        return {"name": name, "count": len(lines), "memory_per_gpu": mem}
    except Exception:
        return None
backend = os.environ.get("BACKEND", "vllm")
if backend == "anthropic":
    model_block = {
        "served_name": g("ANTHROPIC_MODEL") or g("SERVED_MODEL_NAME"),
        "anthropic_version": g("ANTHROPIC_VERSION", default="2023-06-01"),
    }
else:
    model_block = {
        "hf_id": g("MODEL_NAME"),
        "served_name": g("SERVED_MODEL_NAME"),
        "tool_call_parser": g("TOOL_CALL_PARSER"),
        "max_model_len": g("MAX_MODEL_LEN", int),
        "tensor_parallel_size": g("TENSOR_PARALLEL_SIZE", int),
        "gpu_memory_utilization": g("GPU_MEMORY_UTILIZATION", float),
    }
cfg = {
    "run_id": os.environ["RUN_ID"],
    "agent": os.environ["AGENT_NAME"],
    "backend": backend,
    "machine": {
        "hostname": socket.gethostname(),
        "platform": f"{platform.system()} {platform.release()}",
        "gpu": gpu_info() if backend == "vllm" else None,
    },
    "model": model_block,
    "dataset": {
        "name": g("SWE_DATASET"),
        "split": g("SWE_SPLIT"),
        "limit": g("SWE_LIMIT", int),
    },
    "agent_settings": {
        "claude_max_turns": g("CLAUDE_MAX_TURNS", int),
        "claude_timeout_secs": g("CLAUDE_TIMEOUT_SECS", int),
        "max_tokens_cap": g("MAX_TOKENS_CAP", int),
    },
    "server": {
        "host": g("HOST"),
        "port": g("PORT", int),
        "proxy_port": g("PROXY_PORT", int),
    },
}
with open(sys.argv[1], "w") as fh:
    json.dump(cfg, fh, indent=2); fh.write("\n")

# Console summary so the operator sees the resolved config at run start.
def line(label, value):
    print(f"  {label:<22} {value}")
gpu = cfg["machine"]["gpu"]
gpu_str = (f'{gpu["name"]} × {gpu["count"]} ({gpu["memory_per_gpu"]})'
           if gpu else "n/a")
print("[run] resolved config:")
line("run_id",         cfg["run_id"])
line("agent",          cfg["agent"])
line("backend",        cfg["backend"])
line("hostname",       cfg["machine"]["hostname"])
line("gpu",            gpu_str)
if cfg["backend"] == "anthropic":
    line("model", cfg["model"]["served_name"])
    line("api_version", cfg["model"]["anthropic_version"])
else:
    line("model",
         f'{cfg["model"]["served_name"]}  ({cfg["model"]["hf_id"]})')
    line("max_model_len",  cfg["model"]["max_model_len"])
    line("tensor_parallel", cfg["model"]["tensor_parallel_size"])
    line("gpu_mem_util",   cfg["model"]["gpu_memory_utilization"])
    line("tool_parser",    cfg["model"]["tool_call_parser"])
line("dataset",        f'{cfg["dataset"]["name"]}  '
                       f'(split={cfg["dataset"]["split"]}, '
                       f'limit={cfg["dataset"]["limit"]})')
line("max_turns",      cfg["agent_settings"]["claude_max_turns"])
line("timeout_secs",   cfg["agent_settings"]["claude_timeout_secs"])
line("max_tokens_cap", cfg["agent_settings"]["max_tokens_cap"])
line("upstream",       os.environ.get("UPSTREAM_URL", "?"))
line("proxy_port",     cfg["server"]["proxy_port"])
PY
echo "[run] wrote $RUN_DIR/config.json"

# --- 1. resolve upstream (vLLM-local or Anthropic remote) -------------------
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

# --- 1b. start logging proxy -------------------------------------------------
PROXY_AUTH_ARGS=()
if [[ "$BACKEND" == "anthropic" ]]; then
    # Forward claude's outbound Authorization header (OAuth bearer) verbatim
    # so api.anthropic.com sees the same auth it normally would.
    PROXY_AUTH_ARGS=( --passthrough-auth )
    # Tell claude to talk to the proxy. ANTHROPIC_AUTH_TOKEN keeps the OAuth
    # flow active (vs ANTHROPIC_API_KEY which switches to x-api-key auth).
    BEARER=$("$VENV_PY" -c "
import json
print((json.load(open('/root/.claude/.credentials.json')).get('claudeAiOauth') or json.load(open('/root/.claude/.credentials.json'))).get('accessToken',''))
")
    [[ -z "$BEARER" ]] && { echo "[run] could not read OAuth bearer from ~/.claude/.credentials.json" >&2; exit 1; }
    export ANTHROPIC_BASE_URL="$PROXY_URL"
    export ANTHROPIC_AUTH_TOKEN="$BEARER"
fi

echo "[run] starting proxy on $PROXY_URL  →  $UPSTREAM_URL"
"$VENV_PY" "$HERE/pipeline/proxy.py" \
    --upstream "$UPSTREAM_URL" \
    --port "$PROXY_PORT" \
    --per-problem-csv-dir "$PER_PROBLEM_CSV_DIR" \
    --max-tokens-cap "${MAX_TOKENS_CAP:-4096}" \
    "${PROXY_AUTH_ARGS[@]}" \
    >/dev/null 2>&1 &
PROXY_PID=$!
echo -n "[run] waiting for proxy /health"
for _ in $(seq 1 30); do
    if curl -fsS "$PROXY_URL/health" >/dev/null 2>&1; then
        echo " — ready"; break
    fi
    if ! kill -0 "$PROXY_PID" 2>/dev/null; then
        echo
        echo "[run] proxy exited unexpectedly"; exit 1
    fi
    echo -n "."; sleep 1
done
if ! curl -fsS "$PROXY_URL/health" >/dev/null 2>&1; then
    echo " — timed out"; exit 1
fi

# --- 2. fetch dataset --------------------------------------------------------
if [[ ! -s "$PROBLEMS" ]]; then
    "$VENV_PY" "$HERE/pipeline/fetch_dataset.py" \
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

    if [[ "$BACKEND" == "vllm" ]]; then
        # Cold-start vLLM so each problem sees an empty prefix cache, then
        # restart the proxy (whose connection pool went stale with vllm).
        bash "$HERE/pipeline/reset_vllm.sh"
        if [[ -n "${PROXY_PID:-}" ]] && kill -0 "$PROXY_PID" 2>/dev/null; then
            kill "$PROXY_PID" 2>/dev/null || true
            wait "$PROXY_PID" 2>/dev/null || true
        fi
        "$VENV_PY" "$HERE/pipeline/proxy.py" \
            --upstream "$UPSTREAM_URL" \
            --port "$PROXY_PORT" \
            --per-problem-csv-dir "$PER_PROBLEM_CSV_DIR" \
            --max-tokens-cap "${MAX_TOKENS_CAP:-4096}" \
            >/dev/null 2>&1 &
        PROXY_PID=$!
        until curl -fsS "$PROXY_URL/health" >/dev/null 2>&1; do sleep 1; done
    fi

    # Both backends now go through the proxy, which writes the per-problem
    # CSVs directly with full SSE-derived timing.
    run_one_args=( --base-url "$PROXY_URL" )
    if "$VENV_PY" "$HERE/pipeline/run_one.py" \
        --instance-id "$instance_id" \
        --repo "$repo" \
        --base-commit "$base_commit" \
        --problem-statement "$problem" \
        --model "$SERVED_MODEL_NAME" \
        --workdir-root "$WORKDIRS" \
        --max-turns "$CLAUDE_MAX_TURNS" \
        --timeout-secs "$CLAUDE_TIMEOUT_SECS" \
        "${run_one_args[@]}" \
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

if [[ "$RUN_ANALYSIS" -eq 1 ]]; then
    echo
    bash "$HERE/analyze.sh" "$RUN_DIR"
fi
