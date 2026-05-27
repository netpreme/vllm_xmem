# Benchmarking Coding Agents - Optimizing TTFT

Our goal is to see if the offload tier can help optimize inference metrics - latency and throughput. 

Simulating a real-life coding environment in production, multiple concurrent agents solve independent coding problems in parallel using the local GPU connected to vLLM and claude code. We compare the metrics using CPU offload and Mtier offload.

All coding agents solve a unique problem from SWE Bench Verified maintaining the number of concurrent agents in the GPU pool. 
The agents are orchestrated using the harness (claude code), which can use tool calls and makes requests to vLLM turn by turn to reach a solution. The assigned problems are from SWE Bench Verified.

## Modes
1. Given a setup with Mtier (claude code + vllm + Mtier offload) and CPU, have N concurrent coding agents solving problems one by one from the identical problem queue for T minutes.
2. Given a setup with Mtier and CPU, first, run 1. from above to obtain the traces. Then use the traces to trigger coding agent runs for both Mtier and CPU setup. Traces have OSL, ISL_new (uncached tokens) and timestamps to trigger agentic runs

Use 1. to profile and get the isolated end-to-end measurements between Mtier and CPU. Comparing Mtier and CPU may diverge in the HBM usage and offloading behavior.
Use 2. to compare apples-to-apples comparison between Mtier and CPU fairly. Both Mtier and CPU recapitulated the HBM usage and offloading behavior.


# How to run

## Setup (one time)

Install necessary dependencies in `coding_agents/`:
```bash
bash ../setup.sh
```

## Run modes

For mode 1. (live concurrent agents on both backends — no trace):
```bash
./benchmark.sh --concurrency 16 --sustained-mins 20
```

For mode 2. (capture once on mtier, then replay the same trace against both backends):
```bash
# Step 2a: capture a trace by running mode 1 with --save-trace (mtier only)
./benchmark.sh --concurrency 16 --sustained-mins 20 --save-trace

# Step 2b: replay the captured trace against mtier + cpu in parallel.
#          --deterministic pins OSL + sampling so the workload is byte-identical.
./benchmark.sh --from-trace results_benchmarks/bench_sweep_<ts>/c016/ \
               --concurrency 16 --deterministic
```

`benchmark.sh` is the single entrypoint. All modes share the same flags;
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


## Live monitoring

`benchmark.sh` auto-starts the monitoring stack (Prometheus + Grafana + KV/GPU
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

### 1. Plain dual-backend run

Both backends drive SWE-bench tasks via Claude Code. No trace captured.
Useful when you only care about Prom metrics from a live workload.

```bash
./benchmark.sh --concurrency 16 --sustained-mins 20
./benchmark.sh --concurrency 16 --sustained-mins 20 --deterministic
```

### 2. Agent capture (record SWE-bench + Claude workload)

Runs Claude Code agents against SWE-bench, mtier-only, and tees every
`/v1/messages` request + SSE response to per-session JSONL. The captured
trace can be replayed against both backends later, byte-identical.

```bash
./benchmark.sh --concurrency 16 --sustained-mins 20 --save-trace
./benchmark.sh --concurrency 16 --sustained-mins 20 --save-trace --deterministic
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
./benchmark.sh --save-trace \
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
./benchmark.sh --from-trace results_benchmarks/bench_sweep_<ts>/c016/ --concurrency 16

# Deterministic + exact OSL (byte-identical workload on both backends):
./benchmark.sh --from-trace <dir> --concurrency 16 --deterministic

# Force every turn's OSL to a constant (e.g. 1 — artificial micro-benchmark):
./benchmark.sh --from-trace <dir> --concurrency 16 --osl 1

# Sweep concurrencies with the same recorded workload:
./benchmark.sh --from-trace <dir> --concurrency 12 14 16 18 --deterministic
```

---

## System design

```
              ┌──────────────────────────────┐
              │  benchmark.sh  (entrypoint)  │
              └──────────────┬───────────────┘
                             │
                ┌────────────┴────────────┐
                ▼                         ▼
         ┌──────────────┐          ┌──────────────┐
         │  Mode 1:     │          │  Mode 2:     │
         │  Claude Code │          │  Replay a    │
         │  agents (N   │          │  recorded    │
         │  in parallel)│          │  trace       │
         └──────┬───────┘          └──────┬───────┘
                │                         │
                └────────────┬────────────┘
                             │ /v1/messages (Anthropic API)
                             ▼
       ┌──────────────────────────────────────────────────┐
       │                  vLLM dual-backend               │
       │  ┌─────────────────────┐  ┌─────────────────────┐│
       │  │  mtier  (GPU 0)     │  │   cpu  (GPU 1)      ││
       │  │  HBM + MTier offload│  │   HBM + CPU DRAM    ││
       │  └─────────────────────┘  └─────────────────────┘│
       └─────────────────────┬────────────────────────────┘
                             │ metrics scraped per second
                             ▼
       ┌──────────────────────────────────────────────────┐
       │  Prometheus  →  per-concurrency TSDB snapshot    │
       │  Grafana     →  live dashboards                  │
       │  Analysis    →  timeseries + offload figures     │
       └──────────────────────────────────────────────────┘
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
