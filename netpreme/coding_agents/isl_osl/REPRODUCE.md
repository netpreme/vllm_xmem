# ISL/OSL benchmark — reproduce guide

Documents the exact configuration used for each run in `runs/`, the live
configuration in `.env`, and the commands needed to reproduce or extend the
measurement.

---

## 1. Hardware / software baseline

| Component | Value |
|---|---|
| GPU | NVIDIA A100-SXM4-80GB |
| NVIDIA driver | 580.126.09 |
| vLLM | `0.1.dev15152+g47dafbbe9` (this fork; `vllm_xmem` branch `isl-osl-analysis`) |
| Python | 3.12 (uv venv at `/root/vllm_xmem/.venv`) |
| Claude Code CLI | 2.1.139 |
| Model | `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8` (served as `qwen3-coder-30b`) |
| Max model length | 262,144 tokens |
| GPU memory utilization | 0.92 |
| Tool-call parser | `qwen3_coder` |

The vLLM serving binary is patched (uncommitted) in
`vllm/entrypoints/anthropic/serving.py` to populate
`cache_read_input_tokens` from `prompt_tokens_details.cached_tokens` on both
streaming and non-streaming paths. The patch matters for the cache-hit-rate
columns in `usage.jsonl`.

---

## 2. Current `.env` (effective for all new runs)

Path: `/root/vllm_xmem/netpreme/coding_agents/.env`

```
MODEL_NAME=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8
SERVED_MODEL_NAME=qwen3-coder-30b
HOST=0.0.0.0
PORT=8000
TENSOR_PARALLEL_SIZE=1
MAX_MODEL_LEN=262144
GPU_MEMORY_UTILIZATION=0.92
TOOL_CALL_PARSER=qwen3_coder
SWE_LIMIT=500
SWE_DATASET=princeton-nlp/SWE-bench_Verified
SWE_SPLIT=test
CLAUDE_MAX_TURNS=999          # effectively unlimited
CLAUDE_TIMEOUT_SECS=86400     # 24h per problem
MAX_TOKENS_CAP=4096           # per-response cap (proxy)
```

Note: `CLAUDE_MAX_TURNS` / `CLAUDE_TIMEOUT_SECS` were raised from 15/600 on
2026-05-12. Earlier runs use the old (capped) values — see §5.

---

## 3. Bring vLLM up

```bash
cd /root/vllm_xmem
source .venv/bin/activate

# starts vllm serve with the .env settings
bash netpreme/coding_agents/server.sh > /tmp/vllm_server.log 2>&1 &

# wait until ready
until curl -fsS http://localhost:8000/v1/models >/dev/null; do sleep 5; done
```

The vLLM CLI it actually executes (for reference):

```
vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
    --host 0.0.0.0 --port 8000 \
    --served-model-name qwen3-coder-30b \
    --tensor-parallel-size 1 \
    --max-model-len 262144 \
    --gpu-memory-utilization 0.92 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --enable-prompt-tokens-details
```

`--enable-prompt-tokens-details` is what lets vLLM emit per-request
`cached_tokens`, which the patched Anthropic serving path forwards as
`cache_read_input_tokens`.

---

## 4. Launching a run

`run.sh` boots/reuses vLLM, starts the logging proxy on port 9001, fetches
the dataset, and runs `claude -p` once per problem. Each problem is one
`claude -p` invocation — claude itself iterates tool calls inside that single
process up to `CLAUDE_MAX_TURNS`.

### SWE-bench Verified (500 problems)

```bash
cd /root/vllm_xmem/netpreme/coding_agents/isl_osl
bash run.sh                                                    # .env defaults
SWE_LIMIT=20 bash run.sh                                       # smoke
```

### SWE-bench Pro (500 of 731)

```bash
cd /root/vllm_xmem/netpreme/coding_agents/isl_osl
SWE_DATASET=ScaleAI/SWE-bench_Pro bash run.sh                  # 500 (SWE_LIMIT)
SWE_DATASET=ScaleAI/SWE-bench_Pro SWE_LIMIT=731 bash run.sh    # all 731
```

### Important: `run.sh` env-override fix

`run.sh` was patched on 2026-05-12 to preserve command-line env overrides
when sourcing `.env`. Before the patch, `SWE_DATASET=ScaleAI/SWE-bench_Pro`
on the command line was overwritten by the `SWE_DATASET=princeton-...` line
in `.env`. The fixed block (lines 27–37) snapshots overrides before
sourcing and re-exports them after. Run `20260512_012629` was launched with
the buggy version and actually contains Verified data despite its
intended-Pro launch command.

---

## 5. Run lineage

| Dir | Dataset (actual) | SWE_LIMIT | MAX_TURNS | TIMEOUT | Outcome |
|---|---|---|---|---|---|
| `_smoke/` | Verified | 1 (smoke) | 15 | 600 | OK |
| `20260510_042605/` | Verified | 500 | 15 | 600 | 486/500 completed (early stop) |
| `20260511_010738/` | Verified | 500 | 15 | 600 | **500/500 completed** |
| `20260512_012629/` | Verified (mislabeled) | 500 | 15 | 600 | **500/500 completed.** Intended Pro but env bug returned Verified |
| `20260512_154746/` | Pro (real) | 500 | 15 | 600 | **Killed at 7/500.** Superseded by uncapped re-run |

All preserved — none deleted.

---

## 6. Per-turn schema (`usage.jsonl`)

One line per assistant turn:

```
ts                       # ISO timestamp
instance_id              # SWE-bench id
elapsed_ms               # inference wall-time for this /v1/messages call
stream                   # boolean — was it a streaming call
input_tokens             # tokens charged to input on this turn
cache_creation_input_tokens
cache_read_input_tokens  # tokens served from prefix cache (the patch)
output_tokens
stop_reason
category                 # text_only | tool_only | mixed | empty
num_tool_calls
num_text_blocks
isl                      # total input tokens this turn (= input + cache_read)
osl                      # output tokens this turn
isl_new                  # input not from cache
isl_cached               # input from cache
cache_hit_rate           # isl_cached / isl
```

Mirror CSV is at `usage.csv`.

---

## 7. Summarize + plot

```bash
# numeric summary (per-category percentiles, cache hit rates)
/root/vllm_xmem/.venv/bin/python3 \
    netpreme/coding_agents/isl_osl/summarize.py \
    --usage runs/<STAMP>/usage.jsonl \
    --out   runs/<STAMP>/summary.json

# OSL / ISL / ISL_new distributions, 3×4 grid by difficulty
# (Verified only — Pro has no `difficulty` field; use `repo_language` for Pro)
/root/vllm_xmem/.venv/bin/python3 \
    netpreme/coding_agents/isl_osl/plot_dist_grid.py \
    --run-dir runs/<STAMP> \
    --out     analysis/analysis_dist_grid.png \
    --title-suffix "SWE-bench Verified (claude × qwen3-coder-30b)"
```

---

## 8. Gotchas

1. **`claude --bare` is mandatory.** Without it, claude auto-loads accumulated
   session state from `~/.claude/projects/`, which can balloon to multi-MB
   user messages and overflow context. Already wired into `run_one.py:189`.

2. **Clean `~/.claude/projects/-root-vllm-xmem-*` between long runs:**
   ```bash
   for d in /root/.claude/projects/*isl-osl*; do rm -rf "$d"; done
   ```

3. **`MAX_TOKENS_CAP=4096`** in proxy clamps claude's per-call `max_tokens`
   request from its default (~20-32K) down to 4K so the
   `prompt + max_tokens <= max_model_len` check doesn't fail on long
   conversations.

4. **`SWE_LIMIT=731`** to do the full Pro test split (default 500).

5. **One-claude-per-problem.** The runs **don't** start multiple claude
   sessions per problem. Each problem is one `claude -p <prompt>` invocation
   that internally loops tool calls. The proxy logs every internal
   `/v1/messages` round-trip as a `usage.jsonl` row.

6. **Repo clone overhead** dominates per-problem wall time for some
   datasets — Pro repos (NodeBB, ansible, openlibrary, teleport, etc.) are
   much larger than Verified's Python projects, so per-problem wall time
   is 5–10× slower on Pro at the same caps.
