#!/usr/bin/env bash
# Regenerate <run_dir>/data.npz and all figures from <run_dir>/per_problem.
#   ./analyze.sh runs/<stamp>
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${1:-}"
[[ -n "$RUN_DIR" && -d "$RUN_DIR" ]] \
    || { echo "Usage: $(basename "$0") <run_dir>" >&2; exit 2; }
RUN_DIR="$(cd "$RUN_DIR" && pwd)"
OUT="$RUN_DIR/analysis"; mkdir -p "$OUT"

PY=/root/vllm_xmem/.venv/bin/python3
[[ -x "$PY" ]] || PY=python3

CFG="$RUN_DIR/config.json"
DATASET=$("$PY" -c "import json; print(json.load(open('$CFG'))['dataset']['name'])")
AGENT=$("$PY"   -c "import json; print(json.load(open('$CFG'))['agent'])")
case "$DATASET" in
    *Verified*) DATASET_LABEL="Verified"; IS_VERIFIED=1 ;;
    *)          DATASET_LABEL="Pro";      IS_VERIFIED=0 ;;
esac
COUNT=$(ls "$RUN_DIR/per_problem"/*.csv 2>/dev/null | wc -l)
SUFFIX="$AGENT × $DATASET_LABEL ($COUNT problems)"
echo "[analyze] $DATASET_LABEL × $AGENT × $COUNT → $OUT"

# 1. Build the canonical per-turn dataset that every plot reads from.
echo "  -> build_data.py"
"$PY" "$HERE/analysis/build_data.py" --run-dir "$RUN_DIR"

# 2. Generate figures.
plot() {
    echo "  -> $1"
    "$PY" "$HERE/analysis/$1" \
        --run-dir "$RUN_DIR" --out "$OUT/$2" --title-suffix "$SUFFIX"
}
plot plot_itl_vs_isl.py           analysis_itl_vs_isl.png
plot plot_ttft_prefill.py         analysis_ttft_prefill.png
plot plot_dist_agg.py             analysis_dist_agg.png
plot plot_turns.py                analysis_turns.png
# plot_kv_cache.py produces the representative figure AND, with
# --samples-dir, 10 diverse-problem sample figures into analysis/samples/.
echo "  -> plot_kv_cache.py (+ samples)"
"$PY" "$HERE/analysis/plot_kv_cache.py" \
    --run-dir "$RUN_DIR" --out "$OUT/analysis_kv_cache.png" \
    --samples-dir "$OUT/samples" --title-suffix "$SUFFIX"
if [[ "$IS_VERIFIED" -eq 1 ]]; then
    plot plot_dist_grid.py        analysis_dist_grid.png
    plot plot_cache_hit.py        analysis_cache.png
fi
