# Concurrent coding-agents benchmark

Drives **hybrid-mtier** (GPU0:8001) and **hybrid-cpu** (GPU1:8002) in parallel
against the same workload, snapshots Prometheus, and emits analysis figures.

## Setup (one time)

```bash
bash install.sh
```

## Live monitoring

`bench.sh` auto-starts the monitoring stack (Prometheus + Grafana + KV/GPU
exporters) on first invocation, and the bench reuses an already-running
Prometheus on subsequent runs. Open while a run is in flight:

| URL | What you see |
|-----|--------------|
| http://localhost:3000 | Grafana (no login). Two dashboards auto-provisioned: **vLLM xmem** (TTFT, E2E, queue p50/p95/p99, cache hit %, KV offload bandwidth) and **MTier vs CPU — Live** (side-by-side) |
| http://localhost:9090 | Prometheus query UI — ad-hoc PromQL |
| http://localhost:8001/metrics | vLLM mtier raw `/metrics` |
| http://localhost:8002/metrics | vLLM cpu raw `/metrics`   |

All runs are capped at **20 min** by default (`--sustained-mins` for run/record,
`--duration-cap-mins` for replay). Pass either flag to override.

## Run

```bash
./bench.sh --concurrency 16
```

## Run + record a trace

```bash
./bench.sh --concurrency 16 --sustained-mins 20 --save-trace
```

Trace data is written **inside the run's own folder** (alongside the
Prometheus snapshot + analysis figures), so one folder = everything from
one run.

## Run from a recorded trace

```bash
# OSL automatically pinned to each turn's recorded value
./bench.sh --from-trace results_benchmarks/bench_sweep_<ts>/c016/

# Override: force every turn's OSL to a constant (e.g. 1 — artificial)
./bench.sh --from-trace results_benchmarks/bench_sweep_<ts>/c016/ --osl 1
```

---

## System design

```
                 ┌──────────────────────────────────────────┐
                 │   bench.sh   (single user entrypoint)    │
                 └─────────────────────┬────────────────────┘
                                       │ flags
                                       ▼
                 ┌──────────────────────────────────────────┐
                 │   utils/bench.py → CodingAgents class    │
                 └────┬───────────────────────┬─────────────┘
                      │ run / record           │ from-trace
                      ▼                        ▼
   ┌─────────────────────────┐    ┌──────────────────────────┐
   │  utils/bench_concurrent │    │  utils/from_trace.py     │
   │       _users.py         │    │  + from_trace_session.py │
   │                         │    │                          │
   │  spawn N Claude Code    │    │  load recorded trace     │
   │  subprocesses per setup │    │  replay HTTP requests    │
   │  in parallel:           │    │  to each backend on the  │
   │                         │    │  recorded schedule.      │
   │  ┌─────────────────┐    │    │  OSL pinned to recorded  │
   │  │   Claude Code   │    │    │  value (or --osl N).     │
   │  │  (per session,  │    │    │                          │
   │  │   one per task) │    │    │                          │
   │  └────┬────────────┘    │    │                          │
   │       │ HTTP            │    │                          │
   │       ▼                 │    │                          │
   │  ┌───────────────────┐  │    │                          │
   │  │ record_proxy.py   │  │    │                          │
   │  │ (when --save-     │  │    │                          │
   │  │  trace is on)     │  │    │                          │
   │  │ tees /v1/messages │  │    │                          │
   │  │ + SSE to JSONL    │  │    │                          │
   │  └────┬──────────────┘  │    │                          │
   └───────┼─────────────────┘    └──────────┬───────────────┘
           │                                 │
           │  vLLM /v1/messages              │
           ▼                                 ▼
   ┌─────────────────────────────────────────────────────────┐
   │  hybrid-mtier vLLM (GPU0)    hybrid-cpu vLLM (GPU1)     │
   │  ├─ HBM prefix cache          ├─ HBM prefix cache        │
   │  └─ MTier-chip offload tier   └─ CPU DRAM offload tier   │
   └────────────────────────┬────────────────────────────────┘
                            │ metrics scraped per second
                            ▼
   ┌─────────────────────────────────────────────────────────┐
   │  Prometheus (admin API)  →  per-level TSDB snapshot     │
   │  analyze_snapshot.py     →  timeseries + offload PNGs   │
   │  extract_per_turn.py     →  per_turn.csv (record mode)  │
   └─────────────────────────────────────────────────────────┘
```

Determinism:
- `temperature=0`, `seed=42`, `VLLM_BATCH_INVARIANT=1` on both backends.
- In `--from-trace` mode, every request body is byte-identical between
  mtier and cpu, and OSL is pinned via `max_tokens=min_tokens=N+ignore_eos=True`
  (Anthropic adapter patched to forward these). KV-block allocation/eviction/
  offload sequence is then determined entirely by the recorded token stream.

---

## Expected output

Every run produces per concurrency level `<run_dir>/c<NN>/`:

| File | Contents |
|------|----------|
| `config.json` | concurrency, model, setups, t_start/end, per-setup counters |
| `prom_snapshot/` | full Prometheus TSDB at the moment the run finished |
| `analysis/timeseries.png` | HBM hit / Offload hit / Recompute / HBM util over time, per setup |
| `analysis/offload_dominant_mtier.png` | TTFT / E2E / throughput vs offload-share, mtier panel |
| `analysis/offload_dominant_cpu.png` | same, cpu panel |

When `--save-trace` is set, the same `c<NN>/` folder also contains:

| File | Contents |
|------|----------|
| `capture_meta.json` | model, n_sessions, level start/end timestamps |
| `sessions.jsonl` | one record per session-start (instance_id, t_session_start) |
| `per_turn.csv` | per-turn ISL / OSL / ISL_new / timings (auto-extracted) |
| `traces/<instance>.jsonl` | full captured `/v1/messages` request + SSE response per turn |

Aggregate inference metrics across runs (TTFT, ITL, E2E, queue p50/p95/p99,
HBM/offload hit rates, offload GB) are extractable via
`utils/extract_metrics.py --level-dir <c0NN>` → one CSV row per setup.
