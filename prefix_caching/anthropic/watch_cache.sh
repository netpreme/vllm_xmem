#!/usr/bin/env bash
set -euo pipefail

# Watch vLLM prefix cache metrics in real time.
# Run this in a separate terminal alongside run_claude_local.sh.
#
# Usage:
#   ./watch_cache.sh              # default: poll every 2s
#   ./watch_cache.sh 5            # poll every 5s

API_BASE="${API_BASE:-http://127.0.0.1:8000}"
INTERVAL="${1:-2}"

prev_queries=0
prev_hits=0

echo "Watching vLLM prefix cache metrics (every ${INTERVAL}s)"
echo "  endpoint: $API_BASE/metrics"
echo "──────────────────────────────────────────────────────────────"

while true; do
    metrics=$(curl -s "$API_BASE/metrics" 2>/dev/null || true)
    if [[ -z "$metrics" ]]; then
        echo "  (waiting for server...)"
        sleep "$INTERVAL"
        continue
    fi

    queries=$(echo "$metrics" | grep '^vllm:prefix_cache_queries_total' | awk '{print $2}')
    hits=$(echo "$metrics" | grep '^vllm:prefix_cache_hits_total' | awk '{print $2}')
    num_blocks=$(echo "$metrics" | grep -o 'num_gpu_blocks="[^"]*"' | head -1 | cut -d'"' -f2)

    # Skip if no change
    if [[ "$queries" == "$prev_queries" ]]; then
        sleep "$INTERVAL"
        continue
    fi

    # Compute deltas and totals
    python3 -c "
queries = ${queries:-0}
hits = ${hits:-0}
prev_q = ${prev_queries:-0}
prev_h = ${prev_hits:-0}
num_blocks = ${num_blocks:-0}

dq = queries - prev_q
dh = hits - prev_h
dw = dq - dh
hit_rate = 100 * hits / queries if queries else 0
turn_hit_rate = 100 * dh / dq if dq else 0

print()
print(f'  ┌─ This Turn ───────────────────────────────────────────────')
print(f'  │ Input tokens queried:          {dq:.0f}')
print(f'  │ KV cache reads  (from cache):  {dh:.0f} tokens')
print(f'  │ KV cache writes (to cache):    {dw:.0f} tokens')
print(f'  │ Turn hit rate:                 {turn_hit_rate:.1f}%')
print(f'  ├─ Cumulative (all requests) ───────────────────────────────')
print(f'  │ Total queried tokens:          {queries:.0f}')
print(f'  │ Total cache hit tokens:        {hits:.0f}')
print(f'  │ Total cache write tokens:      {queries - hits:.0f}')
print(f'  │ Overall hit rate:              {hit_rate:.1f}%')
print(f'  ├─ KV Cache Pool ───────────────────────────────────────────')
print(f'  │ Total GPU blocks:              {num_blocks} (block_size=16 tokens)')
print(f'  │ Max token capacity:            {num_blocks * 16}')
print(f'  └──────────────────────────────────────────────────────────')
"

    prev_queries="$queries"
    prev_hits="$hits"

    sleep "$INTERVAL"
done
