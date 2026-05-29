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

PY=$(command -v python3)

COUNT=$(ls "$RUN_DIR/per_problem"/*.csv 2>/dev/null | wc -l)
SUFFIX="$COUNT problems"
echo "[analyze] $COUNT problems → $OUT"

# 1. Build the canonical per-turn dataset that every plot reads from.
echo "  -> build_dataset.py"
"$PY" "$HERE/analysis/build_dataset.py" --run-dir "$RUN_DIR"

# 2. Generate figures.
plot() {
    echo "  -> $1"
    "$PY" "$HERE/analysis/$1" \
        --run-dir "$RUN_DIR" --out "$OUT/$2" --title-suffix "$SUFFIX"
}
plot plot_itl_vs_isl.py    analysis_itl_vs_isl.png
plot plot_ttft_prefill.py  analysis_ttft_prefill.png
plot plot_dist_agg.py      analysis_dist_agg.png
plot plot_dist_grid.py     analysis_dist_grid.png
plot plot_turns.py         analysis_turns.png
plot plot_cache_hit.py     analysis_cache.png

# plot_kv_cache.py produces the representative figure AND, with
# --samples-dir, 10 diverse-problem sample figures into analysis/samples/.
echo "  -> plot_kv_cache.py (+ samples)"
"$PY" "$HERE/analysis/plot_kv_cache.py" \
    --run-dir "$RUN_DIR" --out "$OUT/analysis_kv_cache.png" \
    --samples-dir "$OUT/samples" --title-suffix "$SUFFIX"
