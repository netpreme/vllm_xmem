#!/usr/bin/env bash
# Regenerate all figures for an existing run directory.
#   ./analyze.sh runs/<stamp>
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${1:-}"
[[ -n "$RUN_DIR" && -d "$RUN_DIR" ]] || { echo "Usage: $(basename "$0") <run_dir>" >&2; exit 2; }
RUN_DIR="$(cd "$RUN_DIR" && pwd)"
OUT="$RUN_DIR/analysis"; mkdir -p "$OUT"

PY=/root/vllm_xmem/.venv/bin/python3
[[ -x "$PY" ]] || PY=python3

CFG="$RUN_DIR/config.json"
DATASET_NAME=$("$PY" -c "import json; print(json.load(open('$CFG'))['dataset']['name'])")
AGENT=$("$PY"        -c "import json; print(json.load(open('$CFG'))['agent'])")
IS_VERIFIED=0
[[ "$DATASET_NAME" == *"Verified"* ]] && { DATASET_LABEL="Verified"; IS_VERIFIED=1; } \
                                      || DATASET_LABEL="Pro"
PROBLEMS_COUNT=$(ls "$RUN_DIR/per_problem" 2>/dev/null | wc -l)
SUFFIX="$AGENT × $DATASET_LABEL ($PROBLEMS_COUNT problems)"
echo "[analyze] $DATASET_LABEL × $AGENT × $PROBLEMS_COUNT → $OUT"

run_plot() {
    echo "  -> $1"
    "$PY" "$HERE/analysis/$1" --run-dir "$RUN_DIR" --out "$OUT/$2" \
        --title-suffix "$SUFFIX"
}

run_plot plot_itl_vs_isl.py           analysis_itl_vs_isl.png
run_plot plot_ttft_prefill.py         analysis_ttft_prefill.png
run_plot plot_prefill_decode_ratio.py analysis_prefill_decode_ratio.png
run_plot plot_dist_agg.py             analysis_dist_agg.png

if [[ "$IS_VERIFIED" -eq 1 ]]; then
    run_plot plot_dist_grid.py        analysis_dist_grid.png
    run_plot plot_cache_hit.py        analysis_cache.png
fi
