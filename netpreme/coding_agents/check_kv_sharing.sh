#!/usr/bin/env bash
# Check that KV cache sharing works for identical requests.
# Sends the same prompt twice and compares:
#   - TTFT (time to first token) — should be lower on the second request
#   - cached_tokens metric       — should increase by the full prompt length
#
# Usage:  ./check_kv_sharing.sh
#         API_BASE=http://localhost:8000 PROMPT="hello" ./check_kv_sharing.sh

set -euo pipefail

API_BASE="${API_BASE:-http://127.0.0.1:8000}"
MODEL="${MODEL:-Qwen/Qwen2.5-Coder-32B-Instruct}"
PROMPT="${PROMPT:-Explain in detail how the transformer attention mechanism works.}"

# ── helpers ────────────────────────────────────────────────────────────────

get_cached_tokens() {
    curl -sf "$API_BASE/metrics" \
        | grep '^dynamo_frontend_cached_tokens_sum' \
        | awk '{print $2}' \
        | head -1
}

# Send one request with temperature=0; print TTFT in ms to stdout.
# Uses curl's time_starttransfer with streaming so we get true TTFT.
send_request() {
    local label="$1"
    echo "  Sending: $label ..." >&2
    local t
    t=$(curl -sf "$API_BASE/v1/messages" \
        -H "Content-Type: application/json" \
        -H "x-api-key: dummy" \
        -H "anthropic-version: 2023-06-01" \
        -o /dev/null \
        -w "%{time_starttransfer}" \
        -d "{
            \"model\": \"$MODEL\",
            \"max_tokens\": 64,
            \"temperature\": 0,
            \"messages\": [{\"role\": \"user\", \"content\": \"$PROMPT\"}]
        }")
    # convert seconds → ms (awk handles the float)
    awk "BEGIN {printf \"%d\", $t * 1000}"
}

# ── main ───────────────────────────────────────────────────────────────────

echo ""
echo "KV cache sharing check"
echo "  API:    $API_BASE"
echo "  Model:  $MODEL"
echo "  Prompt: ${PROMPT:0:60}..."
echo ""

# Check server is up
if ! curl -sf "$API_BASE/health" > /dev/null; then
    echo "ERROR: server not reachable at $API_BASE" >&2
    exit 1
fi

cached_before=$(get_cached_tokens)
echo "cached_tokens before:  ${cached_before:-0}"
echo ""

ttft1=$(send_request "request 1 (cold)")
cached_after1=$(get_cached_tokens)
echo "  TTFT:          ${ttft1} ms"
echo "  cached_tokens: ${cached_after1:-0}  (delta: $(( ${cached_after1:-0} - ${cached_before:-0} )) tokens)"
echo ""

ttft2=$(send_request "request 2 (should hit cache)")
cached_after2=$(get_cached_tokens)
echo "  TTFT:          ${ttft2} ms"
echo "  cached_tokens: ${cached_after2:-0}  (delta: $(( ${cached_after2:-0} - ${cached_after1:-0} )) tokens)"
echo ""

# ── summary ────────────────────────────────────────────────────────────────

echo "────────────────────────────────"
echo "  Request 1 TTFT : ${ttft1} ms  (cold prefill)"
echo "  Request 2 TTFT : ${ttft2} ms  (cache hit)"

if (( ttft2 < ttft1 )); then
    speedup=$(awk "BEGIN {printf \"%.1f\", $ttft1 / $ttft2}")
    echo "  Speedup        : ${speedup}x  ✓ KV cache sharing is working"
else
    echo "  WARNING: request 2 was not faster — cache may not have been hit"
    echo "  Check: cached_tokens delta on request 2 should equal prompt token count"
fi
echo "────────────────────────────────"
echo ""
