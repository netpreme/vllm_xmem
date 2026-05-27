# ISL/OSL benchmark results

Per-turn input/output sequence-length distributions for a real coding-agent
workload — Claude Code driving Qwen3-Coder-30B-A3B-Instruct-FP8 via vLLM
across **500 SWE-bench Verified problems** (21,143 assistant turns).

## Files

- `data.npz` — canonical per-turn dataset (one structured row per turn). Every figure is derived from it. Load with `numpy.load(...)["turns"]`.
- `analysis_dist_grid.png` — OSL / ISL / ISL_uncached histograms split by SWE-bench difficulty.
- `analysis_dist_agg.png` — Same histograms aggregated across all problems.
- `analysis_turns.png` — distribution of turns-per-problem (aggregate + per difficulty).
- `analysis_cache.png` — vLLM prefix-cache hit rate per turn, by difficulty.
- `analysis_ttft_prefill.png` — TTFT vs prefill workload with the `β₀ + γ·isl_cached + α·isl_new + δ·isl_new·isl` regression fit.
- `analysis_itl_vs_isl.png` — ITL and per-turn decode wall time vs context size.
- `analysis_kv_cache.png` — Per-turn KV cache + wall-time decomposition for one representative problem.
- `samples/kv_<instance_id>.png` — same decomposition for 10 diverse problems spanning the turn-count range.

## Using `data.npz`

```python
import numpy as np
t = np.load("results/data.npz")["turns"]
t.dtype.names
# ('instance_id', 'difficulty', 'agent', 'num_tool_defs', 'num_messages',
#  'system_prompt_chars', 'turn', 'isl', 'osl', 'isl_new', 'isl_cached',
#  'cache_hit_rate', 'prefix_kv_tokens', 'usable_prefix_kv_tokens',
#  'kv_cache_used_bytes', 'kv_cache_usage_pct', 'stop_reason',
#  'ttft_ms', 'prefill_ms', 'decode_ms', 'itl_ms', 'queue_ms', 'elapsed_ms')

# `agent` is "main" for claude's outer loop, "sub" for Task-tool helpers.
# Older runs (before the agent_labeler was introduced) have agent="main" for
# every row.

t[t["difficulty"] == "1+ hours"]                # only hard problems
t[(t["osl"] > 1) & (t["decode_ms"] > 0)]        # generative turns only
t[t["instance_id"] == "astropy__astropy-12907"] # one problem's full trace
```

## Reproduction

```bash
cd /root/vllm_xmem/netpreme/coding_agents/isl_osl
bash ../server.sh > /tmp/vllm.log 2>&1 &            # vLLM only
./run.py                                            # results in ./runs/<stamp>/
./analyze.sh runs/<stamp>                           # builds data.npz + figures
```
