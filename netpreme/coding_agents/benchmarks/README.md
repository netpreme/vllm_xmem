# Benchmarks

Two scripts for two distinct purposes.

## `record_isl_osl.py` — ISL/OSL distribution recording

Runs SWE-bench problems **sequentially** (one at a time) and records per-turn ISL, OSL, tool calls, and cache metrics per problem. Use this to characterise real-world token distributions.

```bash
# Run all instances sequentially
python3 benchmarks/record_isl_osl.py --all

# Resume from where you left off
python3 benchmarks/record_isl_osl.py --all --skip-done

# Slice of the dataset
python3 benchmarks/record_isl_osl.py --all --start 0 --end 100

# Single instance by ID
python3 benchmarks/record_isl_osl.py --id django__django-11099
```

Requires a running server:
```bash
bash netpreme/coding_agents/start_server.sh --hybrid-cpu
```

Results saved to `coding_agents/results/isl_osl/<instance_id>_<timestamp>/metrics.json` — then run analysis:
```bash
python3 analysis/isl_osl_analysis.py
# → ISL/OSL distribution and cache hit figures per difficulty level
```

---

## `bench_concurrent_users.py` — Concurrent user simulation

Runs N sustained concurrent users on SWE-bench tasks. **Recording starts only after CPU/MTier external cache hit rate exceeds 1%** (i.e. once the KV cache is warm and actively serving offloaded hits). Use this to measure TTFT, throughput, and cache performance under real load.

```bash
# Run both setups back-to-back (default: hybrid-cpu then hybrid-mtier)
python3 benchmarks/bench_concurrent_users.py --concurrency 4 8 16

# Single setup — port and GPU are auto-assigned:
#   hybrid-cpu   → port 8001, GPU 1
#   hybrid-mtier → port 8002, GPU 0
python3 benchmarks/bench_concurrent_users.py --setup hybrid-cpu   --concurrency 4 8 16
python3 benchmarks/bench_concurrent_users.py --setup hybrid-mtier --concurrency 4 8 16
python3 benchmarks/bench_concurrent_users.py --setup cpu-only     --concurrency 4 8 16
python3 benchmarks/bench_concurrent_users.py --setup mtier-only   --concurrency 4 8 16

# Sustained mode (keep N users running for ~30 min)
python3 benchmarks/bench_concurrent_users.py --setup hybrid-cpu --concurrency 16 --sustained-mins 30

# Restrict SWE-bench slice
python3 benchmarks/bench_concurrent_users.py --setup hybrid-cpu --concurrency 16 --start 0 --end 50

# Adjust warmup threshold (default: 1% external cache hit rate)
python3 benchmarks/bench_concurrent_users.py --setup hybrid-cpu --concurrency 16 --warmup-cpu-hit-pct 5
```

Results saved to `results_benchmarks/bench_<setup>_p<port>_<timestamp>/`.

---

## Side-by-side Grafana comparison

Run both setups simultaneously in two terminals, then watch them live in Grafana:

**Terminal 1 — CPU (port 8001, GPU 1):**
```bash
python3 netpreme/coding_agents/benchmarks/bench_concurrent_users.py \
    --setup hybrid-cpu --concurrency 16 --sustained-mins 30
```

**Terminal 2 — MTier (port 8002, GPU 0):**
```bash
python3 netpreme/coding_agents/benchmarks/bench_concurrent_users.py \
    --setup hybrid-mtier --concurrency 16 --sustained-mins 30
```

Open the **CPU vs MTier — Live Comparison** dashboard:

```
http://localhost:3000/d/cpu-vs-mtier-v1
```

The dashboard shows CPU (blue, `localhost:8001`) vs MTier (orange, `localhost:8002`) side-by-side:

| Panel | What it shows |
|---|---|
| TTFT p50 / p99 | Time to first token |
| E2E latency | Full request latency |
| ITL | Inter-token latency |
| Queue wait | Scheduling queue delay |
| Cumulative ISL / OSL | Input / output token totals |
| Cache hit rate | GPU HBM + CPU/MTier hit % |
| KV cache usage % | GPU block utilisation |
| Output tok/s | Generation throughput |

Auto-refreshes every 5 s. Ctrl-C in either terminal cleanly stops the server and resets MTier.
