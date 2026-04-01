#!/usr/bin/env bash
set -euo pipefail

# Watch prefix cache metrics in real time (supports both vLLM and Dynamo).
# Run this in a separate terminal alongside the server.
#
# Usage:
#   ./watch_cache.sh              # default: poll every 2s
#   ./watch_cache.sh 5            # poll every 5s

API_BASE="${API_BASE:-http://127.0.0.1:8000}"
INTERVAL="${1:-2}"

prev_queries=0
prev_hits=0
prev_cached_count=0

echo "Watching prefix cache metrics (every ${INTERVAL}s)"
echo "  endpoint: $API_BASE/metrics"
echo "──────────────────────────────────────────────────────────────"

while true; do
    metrics=$(curl -s "$API_BASE/metrics" 2>/dev/null || true)
    if [[ -z "$metrics" ]]; then
        echo "  (waiting for server...)"
        sleep "$INTERVAL"
        continue
    fi

    queries=""
    hits=""
    num_blocks=""
    cached_count="0"

    # Try vLLM format first
    queries=$(echo "$metrics" | grep '^vllm:prefix_cache_queries_total' | awk '{print $2}' || true)
    hits=$(echo "$metrics" | grep '^vllm:prefix_cache_hits_total' | awk '{print $2}' || true)
    num_blocks=$(echo "$metrics" | grep -o 'num_gpu_blocks="[^"]*"' | head -1 | cut -d'"' -f2 2>/dev/null || true)

    if [[ -z "$queries" ]]; then
        # Dynamo format — use cached_count to detect completed requests
        cached_count=$(echo "$metrics" | grep '^dynamo_frontend_cached_tokens_count' | awk '{print $2}' || true)

        # Only read when a new request has fully completed
        if [[ "${cached_count:-0}" == "$prev_cached_count" ]]; then
            sleep "$INTERVAL"
            continue
        fi

        queries=$(echo "$metrics" | grep '^dynamo_frontend_input_sequence_tokens_sum' | awk '{print $2}' || true)
        hits=$(echo "$metrics" | grep '^dynamo_frontend_cached_tokens_sum' | awk '{print $2}' || true)
        num_blocks=$(echo "$metrics" | grep '^dynamo_frontend_model_total_kv_blocks' | awk '{print $2}' || true)
    fi

    if [[ -z "$queries" ]]; then
        echo "  (no cache metrics found yet...)"
        sleep "$INTERVAL"
        continue
    fi

    # For vLLM: skip if no change
    if [[ -z "$cached_count" && "$queries" == "$prev_queries" ]]; then
        sleep "$INTERVAL"
        continue
    fi

    python3 -c "
queries = ${queries:-0}
hits = ${hits:-0}
prev_q = ${prev_queries:-0}
prev_h = ${prev_hits:-0}
num_blocks = ${num_blocks:-0}
req_count = ${cached_count:-0}

dq = queries - prev_q
dh = hits - prev_h
dw = dq - dh
hit_rate = 100 * hits / queries if queries else 0
turn_hit_rate = 100 * dh / dq if dq else 0

print()
print(f'  ┌─ Request Tokens ─────────────────────────────────────────────')
print(f'  │ Input tokens (this request):   {dq:.0f}')
print(f'  │ Output tokens (this request):  (see conversation)')
print(f'  ├─ Prefix Cache (cumulative, all {req_count:.0f} requests) ────────────────')
print(f'  │ Queried tokens:               {queries:.0f}')
print(f'  │ Cache hit tokens:             {hits:.0f}')
print(f'  │ Cache hit rate:               {hit_rate:.1f}%')
print(f'  │ KV cache reads  (from cache): {hits:.0f} tokens')
print(f'  │ KV cache writes (to cache):   {queries - hits:.0f} tokens')
if num_blocks:
    print(f'  ├─ KV Cache Pool ──────────────────────────────────────────────')
    print(f'  │ Total GPU blocks:             {int(num_blocks)} (block_size=16 tokens)')
    print(f'  │ Max token capacity:           {int(num_blocks) * 16}')
print(f'  └──────────────────────────────────────────────────────────────')
"

    prev_queries="$queries"
    prev_hits="$hits"
    prev_cached_count="${cached_count:-0}"

    sleep "$INTERVAL"
done
