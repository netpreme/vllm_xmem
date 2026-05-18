#!/usr/bin/env bash
# Generate all analysis figures for a given run directory.
#
# Usage:
#   ./analyze.sh <run_dir>
#
# Reads <run_dir>/config.json to pick title suffix and bucketing (Verified
# uses `difficulty`; Pro uses `repo_language` and skips difficulty-only plots).
# Writes PNGs to <run_dir>/analysis/.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_DIR="${1:-}"
if [[ -z "$RUN_DIR" || ! -d "$RUN_DIR" ]]; then
    echo "Usage: $(basename "$0") <run_dir>" >&2
    exit 2
fi
RUN_DIR="$(cd "$RUN_DIR" && pwd)"
OUT="$RUN_DIR/analysis"
mkdir -p "$OUT"

if [[ -x /root/vllm_xmem/.venv/bin/python3 ]]; then
    PY=/root/vllm_xmem/.venv/bin/python3
else
    PY=python3
fi

CFG="$RUN_DIR/config.json"
DATASET_NAME=""
AGENT="claude"
BACKEND=""
if [[ -f "$CFG" ]]; then
    DATASET_NAME=$("$PY" -c "import json; print(json.load(open('$CFG'))['dataset']['name'])" 2>/dev/null || echo "")
    AGENT=$("$PY" -c "import json; print(json.load(open('$CFG'))['agent'])" 2>/dev/null || echo "claude")
    BACKEND=$("$PY" -c "import json; print(json.load(open('$CFG'))['backend'])" 2>/dev/null || echo "")
fi

if [[ "$DATASET_NAME" == *"Verified"* ]]; then
    DATASET_LABEL="Verified"; IS_VERIFIED=1
elif [[ "$DATASET_NAME" == *"Pro"* ]]; then
    DATASET_LABEL="Pro"; IS_VERIFIED=0
else
    DATASET_LABEL="(unknown)"; IS_VERIFIED=0
fi

PROBLEMS_COUNT=$(ls "$RUN_DIR/per_problem" 2>/dev/null | wc -l)
SUFFIX="$AGENT × $DATASET_LABEL ($PROBLEMS_COUNT problems)"

echo "[analyze] run_dir=$RUN_DIR"
echo "[analyze] dataset=$DATASET_LABEL  agent=$AGENT  problems=$PROBLEMS_COUNT"
echo "[analyze] writing figures to $OUT"

run_plot() {
    local script="$1"; shift
    local out_name="$1"; shift
    echo "  -> $script"
    "$PY" "$HERE/analysis/$script" \
        --run-dir "$RUN_DIR" \
        --out    "$OUT/$out_name" \
        --title-suffix "$SUFFIX" \
        "$@"
}

# Timing plots need per-turn TTFT/ITL/decode_ms, which only the vLLM proxy can
# capture. Anthropic backend reads token counts only, so skip them.
if [[ "$BACKEND" == "anthropic" ]]; then
    echo "  (skipping timing plots — anthropic backend has no per-turn TTFT/ITL/decode)"
else
    run_plot plot_itl_vs_isl.py           analysis_itl_vs_isl.png
    run_plot plot_ttft_prefill.py         analysis_ttft_prefill.png
    run_plot plot_prefill_decode_ratio.py analysis_prefill_decode_ratio.png
fi

# Distribution and cache plots only make sense on Verified (difficulty bucketing).
if [[ "$IS_VERIFIED" -eq 1 ]]; then
    run_plot plot_dist_grid.py        analysis_dist_grid.png
    run_plot plot_cache_hit.py        analysis_cache.png
else
    echo "  (skipping dist_grid / cache_hit — non-Verified dataset)"
fi

echo "[analyze] done. figures at $OUT"
