# ISL/OSL benchmark results

Per-turn input/output sequence-length distributions for a real coding-agent
workload — Claude Code driving Qwen3-Coder-30B-A3B-Instruct-FP8 via vLLM
across **500 SWE-bench Verified problems** (21,143 assistant turns).

## Files

- `data.npz` — canonical per-turn dataset (one structured row per turn). Every figure is derived from it. Load with `numpy.load(...)["turns"]`.
- `analysis_dist_agg.png` — OSL / ISL / ISL_uncached histograms aggregated across all problems.
- `analysis_cache.png` — vLLM prefix-cache hit rate per turn, by difficulty.
- `analysis_turns.png` — distribution of turns-per-problem (aggregate + per difficulty).
- `analysis_ttft_prefill.png` — TTFT / prefill latency vs ISL workload, with regression fit.
- `analysis_itl_vs_isl.png` — ITL and per-turn decode wall time vs context size.
- `samples/kv_<instance_id>.png` — per-turn KV cache + wall-time decomposition for one representative problem (matplotlib-23412, 142 turns).

## Using `data.npz`

`data.npz` stores RAW MEASUREMENTS only. Derivations (`isl_cached`,
`cache_hit_rate`, `ttft_ms`, `agent` main/sub) are computed on demand by
helpers in `analysis/dataset.py`.

```python
import numpy as np
from analysis.dataset import cache_hit_rate, isl_cached, ttft_ms, agent

t = np.load("results/data.npz")["turns"]
t.dtype.names
# ('instance_id', 'difficulty', 'turn', 'ts',
#  # vLLM-side raw
#  'isl', 'osl', 'isl_new', 'prefill_ms', 'decode_ms', 'queue_ms',
#  'e2e_ms', 'itl_ms', 'kv_cache_usage_pct', 'stop_reason',
#  # proxy-side raw (zero/empty if --capture wasn't passed)
#  'system_prompt_chars', 'num_tool_defs', 'num_messages',
#  'num_tool_calls', 'claude_stop_reason', 'tool_names',
#  'has_thinking', 'response_text_chars')

t[t["difficulty"] == "1+ hours"]                # only hard problems
t[(t["osl"] > 1) & (t["decode_ms"] > 0)]        # generative turns only
t[t["instance_id"] == "astropy__astropy-12907"] # one problem's full trace

cache_hit_rate(t)            # isl_cached / isl
agent(t) == "sub"            # boolean mask of Task-tool sub-agent turns
ttft_ms(t)                   # queue_ms + prefill_ms
```

## Reproduction

```bash
cd /root/vllm_xmem/netpreme/coding_agents/isl_osl
bash ../server.sh > /tmp/vllm.log 2>&1 &            # vLLM only
./coding_agent.py                                   # results in ./runs/<stamp>/
./analyze.sh runs/<stamp>                           # builds data.npz + figures
```
