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

## Run modes

`bench.sh` is the single entrypoint. All modes share the same flags;
which mode you get is decided by which flags you pass.

| Flag | Effect |
|------|--------|
| `--concurrency C` | Concurrent agents (capture) or replay workers (replay) per backend. Accepts a list `--concurrency 12 14 16` to sweep. |
| `--sustained-mins T` | Wall-clock cap per concurrency level. Default 20. |
| `--duration-cap-mins T` | Hard cap for `--from-trace` mode. Default 20. |
| `--save-trace` | Record per-session JSONL traces alongside the Prom snapshot. Mtier-only (since the trace is the source of truth for any later replay). |
| `--from-trace <dir>` | Replay a captured workload against both backends in parallel. |
| `--osl N` | In replay mode: override every turn's OSL to a fixed `N`. |
| `--deterministic` | Pin `VLLM_BATCH_INVARIANT=1`, `temperature=0`, `seed=42` on the vLLM server. In replay mode also pins OSL exactly via `min_tokens+ignore_eos`. **Orthogonal to every mode — append to any command.** |
| `--isl N`, `--isl-new K`, `--n-turns T`, `--n-sessions S` | Synthetic capture only — see below. |
| `--difficulty <set>` | Restrict SWE-bench tasks to a difficulty (easy/medium/hard/vhard). |
| `--start N`, `--end M` | SWE-bench dataset slice indices. |
| `--no-shuffle` | Disable dataset shuffling (use deterministic difficulty-sort order). |
| `--shuffle-seed N` | Pin the shuffle seed for reproducible task ordering. |
| `--model`, `--tp`, `--gpu-util`, `--max-num-seqs` | vLLM server overrides. |

### 1. Plain dual-backend run

Both backends drive SWE-bench tasks via Claude Code. No trace captured.
Useful when you only care about Prom metrics from a live workload.

```bash
./bench.sh --concurrency 16 --sustained-mins 20
./bench.sh --concurrency 16 --sustained-mins 20 --deterministic
```

### 2. Agent capture (record SWE-bench + Claude workload)

Runs Claude Code agents against SWE-bench, mtier-only, and tees every
`/v1/messages` request + SSE response to per-session JSONL. The captured
trace can be replayed against both backends later, byte-identical.

```bash
./bench.sh --concurrency 16 --sustained-mins 20 --save-trace
./bench.sh --concurrency 16 --sustained-mins 20 --save-trace --deterministic
```

Trace data is written **inside the run's own folder** (alongside the
Prometheus snapshot + analysis figures), so one folder = everything from
one run.

### 3. Synthetic capture (controlled ISL / ISL_new / OSL)

No GPU, no agents — fabricates a `/v1/messages` trace with a controlled
per-turn profile. The conversation grows monotonically: turn N's prompt
is turn N-1's prompt + the prior assistant's recorded `OSL` tokens + a
new user message of `ISL_new` tokens. Initial turn has `ISL` total tokens.

```bash
./bench.sh --save-trace \
           --isl 27000 --osl 110 --isl-new 500 \
           --n-turns 50 --n-sessions 30
```

Output goes to `results_benchmarks/bench_sweep_synth_<ts>_isln<K>_osl<M>/c<n_sessions:03d>/`.
Replay it like any other capture (see mode 4). Useful for isolating the
effect of `(ISL, ISL_new, OSL)` from agent-driven variance.

| Flag | Default | Meaning |
|------|---------|---------|
| `--isl N` | required | Target initial input tokens (system + first user message). |
| `--isl-new K` | 500 | Target uncached input tokens per follow-up turn. |
| `--osl M` | 110 | Output tokens per turn. Replay must use `--deterministic` to enforce exactly. |
| `--n-turns T` | 50 | Turns per session. |
| `--n-sessions S` | 30 | Sessions per capture. |

### 4. Replay a recorded trace (agent or synthetic)

Replays the recorded request bodies against both backends in parallel on
the captured timing schedule. Closed-loop floor: `max(captured_gap, response_time)`.

```bash
# OSL pinned to each turn's recorded value (model-decided sampling):
./bench.sh --from-trace results_benchmarks/bench_sweep_<ts>/c016/ --concurrency 16

# Deterministic + exact OSL (byte-identical workload on both backends):
./bench.sh --from-trace <dir> --concurrency 16 --deterministic

# Force every turn's OSL to a constant (e.g. 1 — artificial micro-benchmark):
./bench.sh --from-trace <dir> --concurrency 16 --osl 1

# Sweep concurrencies with the same recorded workload:
./bench.sh --from-trace <dir> --concurrency 12 14 16 18 --deterministic
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
