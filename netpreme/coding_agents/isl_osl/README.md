# ISL/OSL distribution generation

Measures the input and output sequence length (ISL / OSL) distributions of a
real coding agent. Claude Code drives a local vLLM server through SWE-bench
Verified problems; two sidecar processes capture per-turn data:

- **`metrics_watcher`** polls vLLM's Prometheus `/metrics` endpoint and writes
  one row per turn with ISL / OSL / cache hits / TTFT / prefill / decode /
  ITL / queue / KV-usage — everything the model and scheduler can measure.
- **`agent_labeler`** is a thin reverse-proxy in front of vLLM that
  inspects each request body to classify it as main-agent vs Task-tool
  sub-agent (by system-prompt size), then appends a label that the watcher
  merges into the same row.

Single-GPU setup (1× NVIDIA GPU) required.

## Setup

Harness:    Claude Code

Server:     vLLM

Model:      [Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8)

Dataset:    [SWE Bench Verified](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified)

## System design

```
   ┌──────────────────────┐
   │   SWE-bench problem  │   GitHub repo cloned locally @ base_commit,
   │   (Verified / Pro)   │   used as the sandbox repo
   └──────────┬───────────┘
              │ problem statement
              ▼
   ┌──────────────────────┐
   │     Claude Code      │   coding agent: reads files, edits, runs tests,
   │   (claude -p prompt) │   loops tool calls until the bug is fixed
   └──────────┬───────────┘
              │ POSTs /v1/messages, one per turn
              ▼
   ┌──────────────────────┐
   │   agent_labeler      │   inspects system-prompt size →
   │   (reverse proxy)    │   appends "main" or "sub" record
   └──────────┬───────────┘     to .agent_labels (FIFO queue)
              │ forwards unmodified
              ▼
   ┌──────────────────────┐
   │  vLLM (Qwen3-Coder)  │ ─────────► /metrics  (Prometheus endpoint)
   └──────────────────────┘                  ▲
              │ response streamed             │ scraped every 100 ms
              ▼                               │
   ┌──────────────────────┐                   │
   │      claude-cli      │              ┌────┴───────────────────┐
   │  (next turn, repeat) │              │   metrics_watcher       │
   └──────────────────────┘              │   on each detected      │
                                          │   completion:           │
                                          │     · pop one label     │
                                          │       from .agent_labels│
                                          │     · merge with vLLM   │
                                          │       counter deltas    │
                                          │     · write CSV row to  │
                                          │       per_problem/*.csv │
                                          └─────────────────────────┘
                                                       │
                                                       ▼
                                          analyze.sh ─► data.npz + figures
```

**Where each metric comes from.** vLLM's `/metrics` is authoritative for
anything the GPU or scheduler can measure: token counts, prefill/decode/ITL
timings, queue time, prefix-cache stats, KV utilization. At concurrency=1,
every per-request `_sum` histogram updates atomically when a request
completes, so the delta between two scrapes that bracket one completion is
exactly that request's contribution. The labeler owns what only the request
body can tell us: whether the call is claude's main agent loop (~27k char
system prompt) or a Task-tool sub-agent (~3k char system prompt), plus
incidentals like `num_tool_defs` and `num_messages`. The two streams are
joined in the watcher by FIFO ordering — at concurrency=1, the N-th label
written corresponds to the N-th completion observed.

What gets saved per run (`runs/<stamp>/`):

- `config.json` — resolved config (model, dataset, caps, machine)
- `problems.jsonl` — the SWE-bench rows fed to claude
- `per_problem/<id>.csv` — one row per assistant turn (the watcher's output)
- `per_problem/<id>.summary.json` — per-problem totals from claude's `result` event
- `solved.txt` — completed instance IDs
- `.active_instance` — control file the watcher reads to attribute rows
- `.agent_labels` — FIFO queue the labeler appends to and the watcher pops from
  (truncated at the start of every problem)
- `.labeler.log` / `.watcher.log` — sidecar stderr. The watcher log will contain
  `scrape error:` lines during each `reset_vllm.sh` window (the server is
  briefly down between problems) — that's expected.
- `data.npz` — canonical per-turn structured array (built by `analyze.sh`)
- `analysis/*.png` — figures from `analyze.sh` (all read from `data.npz`)

## Code layout

```
pipeline/
  agent_labels.py       on-disk label-record format + Writer + Reader
                        (the contract between labeler and watcher)
  agent_labeler.py      the reverse-proxy. Pure classification logic at the
                        top of the file; HTTP transport at the bottom.
  metrics_watcher.py    polls /metrics, pops labels, writes per-problem CSV
  solve_problem.py      drives one `claude -p` invocation for one problem
  fetch_dataset.py      pulls SWE-bench rows into problems.jsonl
  reset_vllm.sh         cold-restarts vLLM between problems

analysis/
  build_data.py         CSVs + problems.jsonl  →  data.npz
  data.py               small helpers shared by build_data + plots
  plot_*.py             one figure each; every script reads only data.npz

run.sh                  orchestrator: starts sidecars, loops problems
analyze.sh              orchestrator: builds data.npz, renders all figures
```

The modular split means each file has one job. The labeler doesn't know
about timing; the watcher doesn't know about HTTP bodies; the plot scripts
don't know about CSVs. Changing any layer is a localized edit.

## Quick start

```bash
# Start the vLLM server (Qwen3-Coder), then drive a SWE-bench run:
bash ../server.sh > /tmp/vllm.log 2>&1 &
./run.sh                                            # Verified, all 500 problems
./analyze.sh runs/<stamp>                           # build data.npz + figures
```

Other dataset / sample sizes:

```bash
./run.sh --dataset pro                              # SWE-bench Pro instead
./run.sh --limit 50                                 # first 50 problems
./run.sh --random 100 --seed 0                      # random sample of 100
./run.sh --no-analysis                              # skip analyze.sh at the end
```

## Results

![Aggregate OSL / ISL / ISL_uncached distributions](results/analysis_dist_agg.png)

`analysis_dist_agg.png` — Same three histograms collapsed across all 500
problems (no difficulty split). Same semantic region bands as the grid
version: OSL is bucketed into `tool calls / plan / code edits` and
ISL_uncached into `small tool result / file read / large read /
system prompt or compaction`. ISL panel marks the **claude-code
baseline** (~27k tokens total — system prompt ~6.4k + 28 tool schemas ~19.5k +
task statement) as a red reference line.

![Cache hit rate per turn](results/analysis_cache.png)

`analysis_cache.png` — Cache-hit-rate trajectory per turn, by difficulty.
Turn 1 (always 0% cold-start) and auto-compaction turns (`cache_hit < 50%`
AND `isl_new > 50k` — the ~143k full-recompute events) are excluded from
both the per-turn line and the aggregate distribution. y-axis clipped to
60–100%. **Takeaways:** from turn 2 onward, cache hit is already ~75–85%
and climbs to 95–98% steady-state by turn ~5 for all difficulty buckets.
Harder problems just run for many more turns at that steady state.

![Turns per problem](results/analysis_turns.png)

`analysis_turns.png` — distribution of turns-per-problem (substantive turns
only; empty/init rows dropped). Leftmost panel aggregates all 500 problems;
the next three split by difficulty with a shared y-axis for direct
comparison. **Takeaways:** median 29 turns/problem overall, with a clear
monotone shift by difficulty (`<15min` median 26 → `15min–1h` median 30 →
`1+h` median 34). The long tail reaches 471 turns (`sympy__sympy-24443`) —
these are the cases where the model loops without converging on a fix.

![Per-turn KV cache + time breakdown — matplotlib-23412 (142 turns)](results/samples/kv_matplotlib__matplotlib-23412.png)

`samples/kv_matplotlib__matplotlib-23412.png` — example per-turn breakdown
for one representative problem (142 turns). **Top panel**: stacked KV cache
in GB per turn — blue (cached prefix reused), red (recompute), green
(decode). **Bottom panel**: per-turn wall time in ms, decomposed the same
way.
