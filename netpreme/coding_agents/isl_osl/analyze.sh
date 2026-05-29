#!/usr/bin/env bash
# Regenerate <save_dir>/data.npz and all figures from <save_dir>/per_problem.
#   ./analyze.sh save/<stamp>
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAVE_DIR="${1:-}"
[[ -n "$SAVE_DIR" && -d "$SAVE_DIR" ]] \
    || { echo "Usage: $(basename "$0") <save_dir>" >&2; exit 2; }
SAVE_DIR="$(cd "$SAVE_DIR" && pwd)"
OUT="$SAVE_DIR/analysis"; mkdir -p "$OUT"

PY=$(command -v python3)

COUNT=$(ls "$SAVE_DIR/per_problem"/*.csv 2>/dev/null | wc -l)
SUFFIX="$COUNT problems"
echo "[analyze] $COUNT problems → $OUT"

# 1. Build the canonical per-turn dataset that every plot reads from.
echo "  -> build_dataset.py"
"$PY" "$HERE/analysis/metrics.py" --save-dir "$SAVE_DIR"

# 2. Generate figures.
plot() {
    echo "  -> $1"
    "$PY" "$HERE/analysis/$1" \
        --save-dir "$SAVE_DIR" --out "$OUT/$2" --title-suffix "$SUFFIX"
}
plot plot_itl_vs_isl.py    analysis_itl_vs_isl.png
plot plot_ttft_prefill.py  analysis_ttft_prefill.png
plot plot_dist_agg.py      analysis_dist_agg.png
plot plot_dist_grid.py     analysis_dist_grid.png
plot plot_turns.py         analysis_turns.png
plot plot_cache_hit.py     analysis_cache.png
plot plot_cache_tiers.py   analysis_cache_tiers.png

# plot_kv_cache.py produces the representative figure AND, with
# --samples-dir, 10 diverse-problem sample figures into analysis/samples/.
echo "  -> plot_kv_cache.py (+ samples)"
"$PY" "$HERE/analysis/plot_kv_cache.py" \
    --save-dir "$SAVE_DIR" --out "$OUT/analysis_kv_cache.png" \
    --samples-dir "$OUT/samples" --title-suffix "$SUFFIX"
