# ISL/OSL benchmark — current status & runbook

Last update: 2026-05-10 03:25 UTC

## Working baselines

| Combo                                | Status |
|--------------------------------------|--------|
| claude × Verified × qwen3-coder-30b  | full 500-problem run launched (`runs/20260510_032201/`) |
| codex × Verified × qwen3-coder-30b   | smoke OK (1 turn, isl=59796 cached=55824 osl=276) — full run not yet launched |
| claude × Pro × qwen3-coder-30b       | infra ready, not launched |
| codex × Pro × qwen3-coder-30b        | infra ready, not launched |

## Architecture

```
SWE-bench problem
   │
   ▼
fetch_dataset.py   → problems.jsonl (one row per problem)
   │
   ▼
run.sh / run_codex.sh   (loops over problems.jsonl)
   │
   ▼
run_one.py / run_one_codex.py   (per-problem, clones repo @ base_commit)
   │
   ▼
claude -p --bare          OR        codex exec --json
   │                                   │
   ▼                                   ▼
proxy.py (port 9001)              vllm /v1/responses (port 8000)
  Anthropic /v1/messages
   │
   ▼
vllm /v1/messages (port 8000, qwen3-coder-30b FP8)
```

vLLM is patched in `vllm/entrypoints/anthropic/serving.py` to populate
`cache_read_input_tokens` from `prompt_tokens_details.cached_tokens` (both
non-streaming and streaming paths).

## Data emitted per turn

`usage.jsonl` rows (proxy-logged for claude, agent-logged for codex):

```
{ ts, instance_id, agent, turn, category,
  isl, osl, isl_new, isl_cached, cache_hit_rate,
  input_tokens, cached_input_tokens / cache_read_input_tokens,
  output_tokens, stop_reason, num_tool_calls, num_text_blocks, ... }
```

Categories: `text_only` | `tool_only` | `mixed` | `empty`.

Mirror CSV at `usage.csv` (claude only — codex doesn't go through proxy).

## How to fire each remaining combo

All commands assume vLLM is up at `http://localhost:8000` with the
qwen3-coder-30b model. If not: `bash netpreme/coding_agents/server.sh`.

### Codex × Verified
```bash
cd /root/vllm_xmem/netpreme/coding_agents/isl_osl
bash run_codex.sh                                  # all 500
SWE_LIMIT=20 bash run_codex.sh                     # short smoke
```

### Claude × Pro
```bash
cd /root/vllm_xmem/netpreme/coding_agents/isl_osl
SWE_DATASET=ScaleAI/SWE-bench_Pro bash run.sh
```

### Codex × Pro
```bash
cd /root/vllm_xmem/netpreme/coding_agents/isl_osl
SWE_DATASET=ScaleAI/SWE-bench_Pro bash run_codex.sh
```

## Switching to a different model

Two-step: (a) edit `netpreme/coding_agents/.env`, (b) restart vLLM.

```bash
# stop vllm
pkill -f "vllm serve"; sleep 5
# edit .env: MODEL_NAME, SERVED_MODEL_NAME, TOOL_CALL_PARSER, MAX_MODEL_LEN
# (and adjust GPU_MEMORY_UTILIZATION if different precision)
$EDITOR /root/vllm_xmem/netpreme/coding_agents/.env
# bring it back up
bash /root/vllm_xmem/netpreme/coding_agents/server.sh > /tmp/server.log 2>&1 &
# wait until ready
until curl -fsS http://localhost:8000/v1/models >/dev/null; do sleep 5; done
```

After vllm is healthy, run any of the four combos above.

## Candidate single-A100 coding models (verify tool calls before running)

Sorted by VRAM ascending. Set MODEL_NAME=<id>, TOOL_CALL_PARSER=<parser> in .env.

| HF model id | Approx VRAM | parser | ctx | Notes |
|---|---|---|---|---|
| `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8` | 32 GB FP8 | `qwen3_coder` | 256K | **Current baseline.** XML tool format. |
| `mistralai/Devstral-Small-2-24B-Instruct-2512` (or `stelterlab/Devstral-Small-2507-FP8`) | 14–24 GB | `mistral` | 128K | Mistral's agentic SWE coder. |
| `openai/gpt-oss-20b` | ~13 GB MXFP4 | `openai` (Harmony) | 128K | OAI open-weights; needs `openai` parser, not `hermes`. |
| `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` | 32 GB BF16 | none official; try `hermes` | 128K (YaRN) | Tool-call reliability is weak. |
| `cpatonn/GLM-4.5-Air-AWQ-4bit` | 58–62 GB | `glm45` (+ `--reasoning-parser glm45`) | 128K | MoE; keep ctx ≤ 64K. |
| `RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic` (or `hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4`) | 40–70 GB | `llama3_json` | 128K | General — not code-specific; AWQ-INT4 preferred on A100. |
| `openai/gpt-oss-120b` | ~63 GB MXFP4 | `openai` | 128K | Tight on KV cache. |

### Quick tool-call sanity probe (substitute MODEL_NAME first)

```bash
curl -fsS http://localhost:8000/v1/messages \
  -H 'content-type: application/json' \
  -d '{"model":"<served-name>","max_tokens":256,
       "messages":[{"role":"user","content":"What is 17*23? Use the calc tool."}],
       "tools":[{"name":"calc","description":"eval expr",
                 "input_schema":{"type":"object",
                   "properties":{"expr":{"type":"string"}},
                   "required":["expr"]}}]}'
```

Look for `"type":"tool_use"` in the response content. If yes → tool calling works,
proceed with the run. If `"type":"text"` only → swap parser or skip the model.

## Open issues / gotchas

1. **`--bare` is mandatory for claude.** Without it, claude's auto-memory loads
   accumulated session state from `~/.claude/projects/`, ballooning user
   messages to 8 MB+ and overflowing context. Already wired into run_one.py.

2. **Periodic clean of `~/.claude/projects/-root-vllm-xmem-*` recommended**
   between long runs (each session leaves ~17 MB). One-liner:
   `for d in /root/.claude/projects/*isl-osl*; do rm -rf "$d"; done`.

3. **`MAX_TOKENS_CAP=4096`** in proxy is required for claude. claude requests
   `max_tokens=20000–32000` by default; combined with a 200K+ conversation
   that overflows vLLM's `prompt + max_tokens <= max_model_len` check.

4. **Codex 0.130+ dropped `wire_api="chat"`.** Must use `wire_api="responses"`
   in `~/.codex/config.toml`. vLLM's `/v1/responses` works with qwen3-coder.

5. **Disk for repos.** Each problem clones the repo (~800 MB for astropy).
   `rmtree`d after each problem, so steady-state disk is small. With 500
   problems @ avg ~200 MB each, peak transient disk is bounded by single repo.

6. **Per-problem timeout 600 s, MAX_TURNS=15.** Tunable in `.env`. Lower turns
   if context overflows for huge-file problems.

## What's done vs. left

- [x] vllm patched for cache_read_input_tokens
- [x] proxy.py (streaming, gzip-aware, max_tokens clamp, body-dump diag)
- [x] claude run_one.py with `--bare`
- [x] codex install + config + run_one_codex.py + run_codex.sh
- [x] summarize.py works on both claude and codex schemas
- [x] SWE-bench Verified verified working end-to-end (smoke)
- [x] SWE-bench Pro dataset confirmed downloadable (`ScaleAI/SWE-bench_Pro`)
- [x] Candidate model list with parsers
- [ ] **claude × Verified full 500** — IN FLIGHT (`runs/20260510_032201/`)
- [ ] codex × Verified full
- [ ] claude × Pro full
- [ ] codex × Pro full
- [ ] Per-candidate-model tool-call verification + Verified runs

## Estimated remaining compute

- claude × Verified 500: 30–50h (~4–6 min/problem at MAX_TURNS=15)
- codex × Verified 500: similar
- × Pro 731: ~50–75h each
- 6 candidate models × 2 agents × Verified 500: 360–600h

Realistic plan: do Verified+claude (in flight) + Verified+codex back-to-back
(~50–100h total), then evaluate whether to expand. The candidate-model list
gives a path forward but each model swap is multi-day.
