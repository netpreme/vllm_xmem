# Coding Agents — vLLM + Claude Benchmarking

Quick reference for what to run and when.

---

## Files at root (vLLM + Claude)

| File | Purpose |
|------|---------|
| `start_server.sh` | Start a vLLM server (hybrid-cpu or hybrid-mtier) |
| `run_claude.sh` | Run a single Claude Code session on a task |
| `parse_stream.py` | Parse Claude stream output → TTFT / ISL / OSL per turn |
| `watch_server.py` | Live tail of vLLM metrics (cache hit%, queue depth, tok/s) |

```bash
# Start hybrid-mtier on GPU 0, port 8000
CUDA_VISIBLE_DEVICES=0 PORT=8000 bash start_server.sh --hybrid-mtier # or --hybrid-cpu | --cpu-only | --mtier-only

# Ask claude to do a task automatically
bash run_claude.sh --auto "iterate through the vllm_xmem/netpreme codebase, open all the files, and give me a summary of each and write into /tmp/summary_{timestamp}.txt."
```

---

## benchmark/ — concurrent user load test

Run N Claude sessions in parallel against a live vLLM server.
Records TTFT, E2E, ITL, ISL, OSL per turn. Starts/stops the server automatically.

```bash
python3 benchmark/run_concurrent.py \
    --setup (hybrid-mtier | hybrid-cpu) \ 
    --concurrency 4 8 12 16 \
    --sustained --sustained-mins 10 \
    --gpus 0
```

Results land in `benchmark/results_benchmarks/bench_<setup>_<timestamp>/`.

`benchmark/tasks/` contains the SWE-bench task runners used as the workload.

---
## monitoring/ — Prometheus + Grafana

```bash
bash monitoring/start_monitoring.sh   # start Prometheus + Grafana
python3 monitoring/kv_exporter.py     # start KV bandwidth exporter (port 9091)
```

### Accessing Grafana

Open **http://localhost:3000** — default login is `admin / admin`.

Two dashboards are provisioned automatically:

**vLLM xmem** (main benchmark dashboard)
- TTFT p50/p95/p99 over time — both setups
- E2E request latency over time
- Queue wait time
- GPU HBM hit % and external cache hit %
- Output tokens/s and KV offload bandwidth (CPU↔GPU GB/s)

**MTier vs CPU — Live Demo** (side-by-side comparison)
- Use during `demo/` runs to watch MTier vs CPU in real time
- TTFT p50/p95, E2E p50/p95, queue wait, cache hit rates, throughput, KV recall bandwidth
- Blue = hybrid-cpu, orange = hybrid-mtier

### What each exporter provides

| Port | Source | Metrics |
|------|--------|---------|
| 8000 | vLLM frontend | TTFT, E2E, ITL, queue time, cache hits, token counts |
| 8081 | vLLM worker | prefill/decode breakdown, KV block usage |
| 9091 | `kv_exporter.py` | CPU↔GPU transfer bytes and bandwidth (GB/s) |

`kv_exporter.py` tails the Dynamo worker log for `KV Transfer metrics:` lines and exposes them as Prometheus counters/gauges. Run it alongside the server whenever you need the KV offload panels in Grafana.

