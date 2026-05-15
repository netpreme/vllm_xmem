#!/usr/bin/env bash
# Runs Verified, then Pro, sequentially with whatever .env says.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

echo "[wrapper] $(date -u +%FT%TZ) starting Verified"
SWE_DATASET=princeton-nlp/SWE-bench_Verified bash run.sh
echo "[wrapper] $(date -u +%FT%TZ) Verified finished, starting Pro"
SWE_DATASET=ScaleAI/SWE-bench_Pro bash run.sh
echo "[wrapper] $(date -u +%FT%TZ) all done"
