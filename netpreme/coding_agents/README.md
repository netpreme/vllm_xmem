# Coding Agents — Local Dynamo + vLLM Stack

Run Claude Code against a local **Qwen2.5-Coder-32B** model served by Dynamo + vLLM with KV cache offloading to CPU or XMem chip memory.

---

## Setup

```bash
uv pip install "ai-dynamo[vllm]"
uv pip install -e .
uv pip install xmem-mtier   # installs as the 'xmem' package from /usr/local/lib/python3.10
```

Download the model:
```bash
huggingface-cli login
hf download Qwen/Qwen2.5-Coder-32B-Instruct
```

### Required patch

In your environment's dynamo, ex. `.venv/lib/python3.12/site-packages/dynamo/frontend/vllm_processor.py`, around line 383, add an `eos_token_id` guard to prevent generation from being cut short:

```diff
 for k, v in self.input_processor.generation_config_fields.items():
     if hasattr(sampling_params, k):
+        if k == "eos_token_id":
+            continue
         setattr(sampling_params, k, v)
```

---

## Launch the inference stack

```bash
# Hybrid offload — KV evicts to CPU pinned RAM as GPU fills (default)
./netpreme/coding_agents/launch_dynamo.sh

# XMem chip tier instead of CPU
./netpreme/coding_agents/launch_dynamo.sh --mtier

# Scenario 1 — all KV stays in HBM, no offloading
./netpreme/coding_agents/launch_dynamo.sh --no-offload

# Scenario 2 — always evict: GPU capped at 256 blocks, every block cycles through CPU
./netpreme/coding_agents/launch_dynamo.sh --force-offload

# Scenario 2 + XMem — always evict through chip tier
./netpreme/coding_agents/launch_dynamo.sh --mtier --force-offload
```

`FORCE_OFFLOAD_GPU_BLOCKS=N` overrides the 256-block default for `--force-offload`.

Ctrl-C stops both worker and frontend, frees GPU memory, and resets MTier if `--mtier` was used.

---

## Launch Claude Code

```bash
CLAUDE_CODE_ATTRIBUTION_HEADER=0 bash netpreme/prefix_caching/anthropic/_dynamo/run_claude_local.sh
```

Claude Code points at `http://localhost:8000` using the local model. Tool use (Bash, Glob, Grep, WebSearch, etc.) works via the `hermes` tool-call parser wired into the Dynamo frontend.

---

## Monitoring

```bash
# One-time install (Prometheus + Grafana)
./netpreme/coding_agents/monitoring/setup_monitoring.sh

# Start monitoring (wipes Prometheus data each run for a clean slate)
./netpreme/coding_agents/monitoring/start_monitoring.sh        # fresh start
./netpreme/coding_agents/monitoring/start_monitoring.sh --keep # preserve history
```

| URL | Service |
|-----|---------|
| `http://localhost:9090` | Prometheus |
| `http://localhost:3000` | Grafana — "vLLM + XMem — Unified" dashboard |
| `http://localhost:8081/metrics` | vLLM worker metrics (`vllm:*`, `dynamo_component_*`) |

The dashboard has three sections:

- **Dynamo / Router** — TTFT p50/p95/p99, KV hit rate, request throughput
- **vLLM Backend** — E2E latency, ITL, token throughput, GPU cache utilisation
- **XMem / KV Offload** — CPU↔GPU bandwidth (GB/s), cumulative bytes transferred

Prometheus scrapes `:8000` (vLLM + Dynamo metrics) and `:9091` (KV transfer exporter) at 1 s intervals.

### MTier quick commands (in any shell after sourcing `.bashrc`)

```bash
mtier-smi              # show MTier memory pool status
mtier-smi -l 1         # refresh every 1 s (like nvidia-smi -l 1)
mtier reset -y         # release all server-side allocations (no prompt)
```

---

## Key metrics

| Metric | What it measures |
|--------|-----------------|
| `dynamo_component_router_time_to_first_token_seconds_bucket` | TTFT histogram at the router (requires `--router-mode kv --no-router-kv-events`) |
| `vllm:time_to_first_token_seconds_bucket` | TTFT histogram from the vLLM engine |
| `vllm:e2e_request_latency_seconds_bucket` | End-to-end request latency |
| `vllm_kv_cpu_to_gpu_bandwidth_gbps` | CPU→GPU recall bandwidth |
| `vllm_kv_gpu_to_cpu_bandwidth_gbps` | GPU→CPU offload bandwidth |
| `dynamo_component_router_requests_total` | Request counter (use `rate([1m])` for QPS) |

**QPS at 500 ms TTFT SLO** (conservative lower bound using closest bucket):
```promql
rate(dynamo_component_router_time_to_first_token_seconds_bucket{le="0.47"}[1m])
```
