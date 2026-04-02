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

    queries=""
    hits=""
    kv_usage=""

    # Try vLLM format first
    # vllm:prefix_cache_queries_total = total input tokens queried against prefix cache
    # vllm:prefix_cache_hits_total    = tokens served from cache (= num_cached_tokens)
    # vllm:kv_cache_usage_perc        = fraction of KV cache in use (0.0-1.0)
    queries=$(echo "$metrics" | grep '^vllm:prefix_cache_queries_total' | awk '{print $2}' || true)
    hits=$(echo "$metrics" | grep '^vllm:prefix_cache_hits_total' | awk '{print $2}' || true)
    kv_usage=$(echo "$metrics" | grep '^vllm:kv_cache_usage_perc' | awk '{print $2}' || true)

    # Fall back to Dynamo format
    if [[ -z "$queries" ]]; then
        queries=$(echo "$metrics" | grep '^dynamo_frontend_input_sequence_tokens_sum' | awk '{print $2}' || true)
        hits=$(echo "$metrics" | grep '^dynamo_frontend_cached_tokens_sum' | awk '{print $2}' || true)
        kv_usage=$(echo "$metrics" | grep '^dynamo_frontend_model_total_kv_blocks' | awk '{print $2}' || true)
    fi

    if [[ -z "$queries" ]]; then
        echo "  (no cache metrics found yet...)"
        sleep "$INTERVAL"
        continue
    fi

    # Skip if no change
    if [[ "$queries" == "$prev_queries" ]]; then
        sleep "$INTERVAL"
        continue
    fi

    python3 -c "
queries = ${queries:-0}
hits = ${hits:-0}
prev_q = ${prev_queries:-0}
prev_h = ${prev_hits:-0}
kv_usage = ${kv_usage:-0}

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
if kv_usage:
    print(f'  ├─ KV Cache Pool ───────────────────────────────────────────')
    if kv_usage <= 1.0:
        # vLLM: kv_cache_usage_perc is a 0.0-1.0 fraction
        print(f'  │ KV cache fill:                 {kv_usage*100:.1f}%')
    else:
        # Dynamo: dynamo_frontend_model_total_kv_blocks is a raw block count
        print(f'  │ Total GPU blocks:              {int(kv_usage)} (block_size=16 tokens)')
        print(f'  │ Max token capacity:            {int(kv_usage) * 16}')
print(f'  └──────────────────────────────────────────────────────────')
"

    prev_queries="$queries"
    prev_hits="$hits"

    sleep "$INTERVAL"
done
