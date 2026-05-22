# ISL/OSL distribition generation

Measures the input and output sequence length (ISL / OSL) distributions of a
real coding agent. Claude Code drives a model — local via vLLM, or hosted via
the Anthropic API — through SWE-bench Verified problems, and a logging proxy
records ISL/OSL plus cache and timing on every turn. Single-GPU setup
(1× NVIDIA GPU) required.

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
              │ many turns of /v1/messages, each carrying
              │   prompt history + tool results
              ▼
   ┌──────────────────────┐   tap captures, per turn:
   │  measurement tap     │   ISL, OSL, ISL_new, ISL_cached,
   │  (logging proxy)     │   cache_hit_rate, ttft_ms, decode_ms,
   └──────────┬───────────┘   itl_ms, category (text / tool / mixed)
              │ forwards request unchanged
              ▼
   ┌─────────────────────────────────────────┐
   │              Model server               │
   │  ┌─────────────────┐  ┌───────────────┐ │
   │  │  vLLM (local)   │  │ Anthropic API │ │
   │  │  Qwen3-Coder    │  │ Claude Opus   │ │
   │  └─────────────────┘  └───────────────┘ │
   └─────────────────────────────────────────┘
              │ response streamed back through the proxy to claude
              ▼
   ┌──────────────────────┐
   │  per-problem CSV +   │   analyze.sh consolidates everything into one
   │  full text JSONL     │   data.npz, then renders figures from it
   └──────────────────────┘
```

What gets saved per run (`runs/<stamp>/`):

- `config.json` — resolved config (model, dataset, caps, machine)
- `problems.jsonl` — the SWE-bench rows fed to claude
- `per_problem/<id>.csv` — one row per assistant turn (the proxy log)
- `per_problem/<id>.summary.json` — per-problem totals from claude's `result` event
- `transcripts/<id>.jsonl` — full per-turn request + response text
- `solved.txt` — completed instance IDs
- `data.npz` — canonical per-turn structured array (built by `analyze.sh`)
- `analysis/*.png` — figures from `analyze.sh` (all read from `data.npz`)

## Quick start

```bash
# Local vLLM (Qwen3-Coder) — start the server, then run:
bash ../server.sh > /tmp/vllm.log 2>&1 &
./run.sh                                            # claude code + vllm + swe bench verfied
./analyze.sh runs/<stamp>                           # build data.npz + figures
```

To run SWE-bench Pro, use:

```bash
bash ../server.sh > /tmp/vllm.log 2>&1 &
./run.sh --dataset pro
./analyze.sh runs/<stamp>
```

To use the hosted Anthropic API (no local vLLM needed; uses your existing
`claude login` credentials), use:

```bash
./run.sh --backend anthropic --model claude-opus-4-7
./analyze.sh runs/<stamp>
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
comparison. **Takeaways:** median ~31 turns/problem overall, with a clear
monotone shift by difficulty (`<15min` median 26 → `15min–1h` median 32 →
`1+h` median 39). One outlier at 618 turns lives in the easy bucket
(probably a loop the model couldn't escape).

![Per-turn KV cache + time breakdown — matplotlib-24637 (171 turns)](results/samples/kv_matplotlib__matplotlib-24637.png)

`samples/kv_matplotlib__matplotlib-24637.png` — example per-turn breakdown
for one representative problem (171 turns). **Top panel**: stacked KV cache
in GB per turn — blue (cached prefix reused), red (recompute), green
(decode). **Bottom panel**: per-turn wall time in ms, decomposed the same
way. **Hatched bars** mark Task-tool sub-agent turns; solid bars are the main agent. 
