vllm server + client

`launch_vllm.sh`
`infer_poc.sh`

vllm + client + dynamo
`launch_dynamo.sh`
`infer_conversation.sh`
`watch_cache.sh`

  1. launch_dynamo.sh — Starts NVIDIA Dynamo frontend + vLLM
  worker on GPUs 6,7 with TP=2. Runs in foreground, Ctrl-C kills
  both. Dynamo sits in front of vLLM as an OpenAI-compatible API
  gateway.
  2. infer_conversation.sh (openai version) — Multi-turn
  interactive chat using OpenAI /v1/chat/completions format.
  Accumulates full message history each turn so prefix caching
  works. Shows per-request token usage and cumulative cache
  metrics after each response.
  3. watch_cache.sh — Polls Prometheus metrics in a separate
  terminal. Supports both vLLM and Dynamo metric formats. For
  Dynamo, triggers on cached_tokens_count to avoid stale reads.

  Architecture:

  User ─→ infer_conversation.sh ─→ Dynamo frontend (:8000) ─→ vLLM
   worker (GPUs 6,7)
                                          │
                                /metrics (Prometheus)
                                          │
                                watch_cache.sh (polls)

  Key difference from the Anthropic setup: Uses OpenAI API format
  instead of /v1/messages. Dynamo adds an orchestration layer that
   will matter when scaling to multi-node — for now on 2xH100 it's
   a thin pass-through that validates the integration works.

  What's confirmed working:
  - Prefix caching with Dynamo (83%+ hit rate in multi-turn)
  - KV cache pool: 17,597 blocks × 16 tokens = 281K token capacity
   on 2xH100
  - Metrics from both vLLM engine logs and Dynamo's Prometheus
  endpoint

  ---
  