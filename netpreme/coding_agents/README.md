# Coding Agents — vLLM + Claude Benchmarking

Quick reference for what to run and when.

---

## Root files

| File | Purpose |
|------|---------|
| `start_server.sh` | Start a vLLM server in a given mode |
| `run_claude.sh` | Run a single Claude Code session on a task |
| `watch_server.py` | Live tail of vLLM metrics (cache hit%, queue depth, tok/s) |

```bash
# Start server — choose one mode:
bash start_server.sh --hybrid-cpu     # GPU HBM + CPU DRAM overflow  (port 8000)
bash start_server.sh --hybrid-mtier   # GPU HBM + MTier overflow      (port 8000)
bash start_server.sh --cpu-only       # CPU DRAM only (no GPU cache)
bash start_server.sh --mtier-only     # MTier only (no GPU cache)
bash start_server.sh --hbm-only       # GPU HBM only, no offload

# Optional overrides
PORT=8001 bash start_server.sh --hybrid-cpu
CUDA_VISIBLE_DEVICES=0 bash start_server.sh --hybrid-mtier

# Run a task with Claude
bash run_claude.sh --auto "fix the bug in vllm/v1/engine/core.py"
```

---

## benchmarks/ — two scripts, two purposes

See `benchmarks/README.md` for full usage.

### 1. `record_isl_osl.py` — ISL/OSL distribution recording

Runs SWE-bench problems **sequentially**, one at a time. Records per-turn ISL, OSL, tool calls, and cache metrics per problem. Use this to characterise real-world token distributions.

```bash
python3 benchmarks/record_isl_osl.py --all --skip-done
```

Results → `results/isl_osl/<instance_id>_<timestamp>/metrics.json`

Analysis → `python3 analysis/isl_osl_analysis.py`

### 2. `bench_concurrent_users.py` — Concurrent user simulation

Runs N sustained concurrent users. Recording starts only after external KV cache hit rate exceeds 1% (KV cache is warm). Starts and stops the server automatically.

```bash
# Use whatever server is running (default port 8000)
python3 benchmarks/bench_concurrent_users.py --concurrency 4 8 16 --sustained-mins 30

# Override port
python3 benchmarks/bench_concurrent_users.py --port 8001 --concurrency 4 8 16 --sustained-mins 30
python3 benchmarks/bench_concurrent_users.py --port 8002 --concurrency 4 8 16 --sustained-mins 30

# Or auto-start + kill the server (--cpu-hybrid | --mtier-hybrid | --cpu-only | --mtier-only)
python3 benchmarks/bench_concurrent_users.py --cpu-hybrid --concurrency 4 8 16 --sustained-mins 30
```

Results → `results/isl_osl/bench_<setup>_<timestamp>/`

Analysis → `python3 analysis/generate_metrics.py`

---

## analysis/ — figures

| Script | Reads from | Produces |
|--------|-----------|---------|
| `isl_osl_analysis.py` | `results/isl_osl/<instance>/metrics.json` | ISL/OSL distribution + cache hit figures per difficulty level |
| `generate_metrics.py` | `results/isl_osl/bench_combined_*/summary.csv` | TTFT/throughput/speedup comparison figures (CPU vs MTier) |

---

## demo/ — live MTier vs CPU comparison for TTFT

Side-by-side TTFT comparison of hybrid-cpu vs hybrid-mtier using SWE-bench problems.

```bash
# Start both servers (GPU 0 = MTier port 8002, GPU 1 = CPU port 8001)
bash demo/start_servers.sh

# Run the demo
python3 demo/demo_ttft_compare.py -c 4 -r 3
```

See `demo/README.md` for full details.

---

## monitoring/ — Prometheus + Grafana

```bash
bash monitoring/start_monitoring.sh   # Prometheus :9090, Grafana :3000, KV exporter :9091
```

Open **http://localhost:3000** — no login required.

Two dashboards auto-provisioned:
- **vLLM xmem** — TTFT, E2E, queue wait, cache hit%, KV offload bandwidth
- **MTier vs CPU — Live Demo** — side-by-side real-time comparison

| Port | Source | Metrics |
|------|--------|---------|
| 8000 | vLLM server | TTFT, E2E, ITL, queue time, cache hits, token counts |
| 9091 | `kv_exporter.py` | CPU↔GPU transfer bytes and bandwidth (GB/s) |

`monitoring/parse_stream.py` — parses `claude --output-format stream-json` output into per-turn ISL/OSL/tool tables. Used automatically by `run_claude.sh`.
