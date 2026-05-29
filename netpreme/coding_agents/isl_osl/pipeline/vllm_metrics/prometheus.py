"""vLLM Prometheus `/metrics`: metric names + text parsing (pure, no I/O).

Per-request histogram _sum counters (seconds) tick atomically at request
completion, in lockstep with our completion trigger. We intentionally skip
`vllm:time_to_first_token_seconds_sum`: vLLM observes TTFT at first-token-time
(mid-request), so its _sum delta against our completion-triggered scrapes is
0. TTFT is recomputed at analysis time as `queue_ms + prefill_ms`.
"""

from __future__ import annotations

import re

PREFILL_SUM = "vllm:request_prefill_time_seconds_sum"
DECODE_SUM = "vllm:request_decode_time_seconds_sum"
QUEUE_SUM = "vllm:request_queue_time_seconds_sum"
TPOT_SUM = "vllm:request_time_per_output_token_seconds_sum"
E2E_SUM = "vllm:e2e_request_latency_seconds_sum"

# Per-request token histograms — atomic at completion.
PROMPT_TOKENS_SUM = "vllm:request_prompt_tokens_sum"
GEN_TOKENS_SUM = "vllm:request_generation_tokens_sum"
# Uncached input tokens that actually went through prefill compute.
PREFILL_KV_COMPUTED_SUM = "vllm:request_prefill_kv_computed_tokens_sum"

# Labelled by finished_reason — the label whose counter incremented by 1
# between scrapes is the stop_reason for this turn.
FINISH_REASON = "vllm:request_success_total"

# Instantaneous gauge: HBM (GPU) block-pool occupancy, 0..1.
KV_USAGE_PCT = "vllm:kv_cache_usage_perc"

# Prefix-cache hit counters (cumulative tokens). `_total` suffix is required
# so we match the counter value and not its `_created` line. The denominator
# is `isl` (request_prompt_tokens), so we don't store the redundant
# vllm:prefix_cache_queries counter.
#   hits           — tokens served from the local HBM/GPU prefix cache
#   external hits  — tokens served from the offload tier (KV connector);
#                    absent (→ 0) unless an offloading connector is on.
PREFIX_CACHE_HITS = "vllm:prefix_cache_hits_total"
EXTERNAL_PREFIX_CACHE_HITS = "vllm:external_prefix_cache_hits_total"

# Completion trigger.
REQUEST_COUNT = "vllm:request_prompt_tokens_count"

FINISHED_REASONS = ("stop", "length", "abort", "error", "repetition")

METRIC_LINE = re.compile(r"^(.+?)\s+([0-9eE.+\-]+|NaN|\+Inf|\-Inf)$")


def parse_raw_response(body: str) -> dict[str, float]:
    """Parse `/metrics` text into `{metric_line: value}` where `metric_line`
    is everything up to the trailing whitespace+value — labels included."""
    parsed: dict[str, float] = {}
    for line in body.splitlines():
        if not line or line.startswith("#"):
            continue
        match = METRIC_LINE.match(line)
        if not match:
            continue
        try:
            parsed[match.group(1)] = float(match.group(2))
        except ValueError:
            # raise ValueError(f"Could not find body: {body}")
            pass
    return parsed


def extract_metric(
    metrics: dict[str, float], name_prefix: str, label_substring: str = ""
) -> float:
    """First metric line that starts with `name_prefix` and (optionally)
    contains `label_substring`. Returns 0.0 if absent."""
    for name, value in metrics.items():
        if not name.startswith(name_prefix):
            continue
        if label_substring and label_substring not in name:
            continue
        return value

    return 0.0
