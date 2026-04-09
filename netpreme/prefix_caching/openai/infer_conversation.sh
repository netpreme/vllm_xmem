#!/usr/bin/env bash
set -euo pipefail

# Interactive multi-turn conversation script.
# Each turn sends the FULL message history so vLLM's prefix caching
# can reuse the KV cache for the shared prefix.
#
# Usage:
#   ./infer_conversation.sh                       # interactive mode
#   PROMPT="one-shot question" ./infer_conversation.sh  # single-shot

API_BASE="${API_BASE:-http://127.0.0.1:8000}"
MODEL="${MODEL:-Qwen/Qwen2.5-Coder-32B-Instruct}"
MAX_TOKENS="${MAX_TOKENS:-512}"

# ── helpers ──────────────────────────────────────────────────────────
json_escape() {
    python3 -c "import json,sys; print(json.dumps(sys.stdin.read().rstrip()))"
}

build_messages_json() {
    # Build a JSON array from the parallel arrays
    python3 -c "
import json, sys
roles  = sys.argv[1].split('|')
bodies = sys.argv[2].split('|')
msgs = [{'role': r, 'content': b} for r, b in zip(roles, bodies)]
print(json.dumps(msgs))
" "$1" "$2"
}

# ── state ────────────────────────────────────────────────────────────
ROLES=""      # pipe-delimited roles
CONTENTS=""   # pipe-delimited contents

add_message() {
    local role="$1" content="$2"
    if [[ -z "$ROLES" ]]; then
        ROLES="$role"
        CONTENTS="$content"
    else
        ROLES="${ROLES}|${role}"
        CONTENTS="${CONTENTS}|${content}"
    fi
}

send_request() {
    local messages_json
    messages_json=$(build_messages_json "$ROLES" "$CONTENTS")

    local payload
    payload=$(python3 -c "
import json, sys
print(json.dumps({
    'model': sys.argv[1],
    'max_tokens': int(sys.argv[2]),
    'messages': json.loads(sys.argv[3])
}))
" "$MODEL" "$MAX_TOKENS" "$messages_json")

    local response
    response=$(curl -s "$API_BASE/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer dummy" \
        -d "$payload")

    # Extract the text from the response
    local text
    text=$(echo "$response" | python3 -c "
import json, sys
resp = json.load(sys.stdin)
if 'choices' in resp:
    print(resp['choices'][0]['message']['content'])
elif 'error' in resp:
    print('ERROR:', resp['error'].get('message', resp['error']), file=sys.stderr)
    sys.exit(1)
else:
    print(json.dumps(resp, indent=2))
")

    echo "$text"

    # Extract token usage from response (OpenAI format)
    local input_tokens output_tokens
    input_tokens=$(echo "$response" | python3 -c "
import json, sys
resp = json.load(sys.stdin)
print(resp.get('usage', {}).get('prompt_tokens', '?'))
" 2>/dev/null || echo "?")
    output_tokens=$(echo "$response" | python3 -c "
import json, sys
resp = json.load(sys.stdin)
print(resp.get('usage', {}).get('completion_tokens', '?'))
" 2>/dev/null || echo "?")

    # Append assistant reply to history
    add_message "assistant" "$text"

    # Show prefix cache metrics (supports both vLLM and Dynamo endpoints)
    local metrics
    metrics=$(curl -s "$API_BASE/metrics" 2>/dev/null || true)
    if [[ -n "$metrics" ]]; then
        local queries="" hits="" num_blocks=""

        # Try vLLM format first
        queries=$(echo "$metrics" | grep '^vllm:prefix_cache_queries_total' | awk '{print $2}' || true)
        hits=$(echo "$metrics" | grep '^vllm:prefix_cache_hits_total' | awk '{print $2}' || true)
        num_blocks=$(echo "$metrics" | grep -o 'num_gpu_blocks="[^"]*"' | head -1 | cut -d'"' -f2 2>/dev/null || true)

        # Fall back to Dynamo format if vLLM metrics not found
        if [[ -z "$queries" ]]; then
            queries=$(echo "$metrics" | grep '^dynamo_frontend_input_sequence_tokens_sum' | awk '{print $2}' || true)
            hits=$(echo "$metrics" | grep '^dynamo_frontend_cached_tokens_sum' | awk '{print $2}' || true)
            num_blocks=$(echo "$metrics" | grep '^dynamo_frontend_model_total_kv_blocks' | awk '{print $2}' || true)
        fi

        # Only show metrics if we got data
        if [[ -n "$queries" || -n "$hits" ]]; then
            local hit_rate="N/A"
            if [[ -n "$queries" && -n "$hits" && "$queries" != "0" && "$queries" != "0.0" ]]; then
                hit_rate=$(python3 -c "print(f'{100*${hits}/${queries}:.1f}%')" 2>/dev/null || echo "?")
            fi
            local write_tokens
            write_tokens=$(python3 -c "print(f'{${queries:-0} - ${hits:-0}:.0f}')" 2>/dev/null || echo "?")

            echo ""
            echo "  ┌─ Request Tokens ─────────────────────────────────────────────"
            echo "  │ Input tokens (this request):  $input_tokens"
            echo "  │ Output tokens (this request): $output_tokens"
            echo "  ├─ Prefix Cache (cumulative, all requests) ────────────────────"
            echo "  │ Queried tokens:               ${queries:-?}"
            echo "  │ Cache hit tokens:             ${hits:-?}"
            echo "  │ Cache hit rate:               $hit_rate"
            echo "  │ KV cache reads  (from cache): ${hits:-0} tokens"
            echo "  │ KV cache writes (to cache):   ${write_tokens} tokens"
            if [[ -n "$num_blocks" ]]; then
            echo "  ├─ KV Cache Pool ──────────────────────────────────────────────"
            echo "  │ Total GPU blocks:             ${num_blocks} (block_size=16 tokens)"
            echo "  │ Max token capacity:           $(( num_blocks * 16 ))"
            fi
            echo "  └──────────────────────────────────────────────────────────────"
        else
            echo ""
            echo "  [cache metrics not available from $API_BASE/metrics]"
        fi
    fi
}

# ── single-shot mode ────────────────────────────────────────────────
if [[ -n "${PROMPT:-}" ]]; then
    add_message "user" "$PROMPT"
    send_request
    exit 0
fi

# ── interactive loop ─────────────────────────────────────────────────
echo "Multi-turn conversation (prefix caching enabled on server)."
echo "Type your message and press Enter. Ctrl-D or 'exit' to quit."
echo "---"

while true; do
    printf "\n[you]: "
    read -r user_input || break          # Ctrl-D exits
    [[ "$user_input" == "exit" ]] && break
    [[ -z "$user_input" ]] && continue

    add_message "user" "$user_input"

    printf "\n[assistant]: "
    send_request
done

echo -e "\nGoodbye."
