# SWE-bench HBM Fill → Spill → Recall Demo

Compares **hybrid-cpu** (DRAM offload) vs **hybrid-mtier** (MTier chip offload) vLLM servers by running real [SWE-bench Verified](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified) problems through a three-phase KV cache stress test.

## What it does

| Phase | Turn | Prompt size | Purpose |
|-------|------|-------------|---------|
| Fill-HBM | 1 | ~64K tokens | Loads KV cache into GPU HBM |
| Spill-to-offload | 2 | +64–127K tokens | Overflows HBM → spills KV to external memory |
| Recall | 3+ | Short (~10 tokens) | Short query that forces a full KV reload from external memory |

The recall turns isolate the KV transfer cost: CPU must DMA over PCIe (~32 GB/s), MTier fetches over GPU fabric (~300 GB/s). The speedup is visible in TTFT at recall turns.

## Setup

```bash
# Install deps
pip install datasets httpx

# Optional: set GitHub token for faster repo file fetching (5000 req/hr vs 60)
export GITHUB_TOKEN=ghp_...
```

## Usage

```bash
# 1. Start both servers (GPU 0 = MTier, GPU 1 = CPU)
bash netpreme/coding_agents/demo/start_servers.sh

# 2. Run the demo
python3 netpreme/coding_agents/demo/swe_spill_recall.py

# Options
python3 swe_spill_recall.py -c 4   # 4 concurrent sessions (default 8)
python3 swe_spill_recall.py -r 5   # 5 recall turns (default 3)
```

## Output

Live terminal display (4 sessions shown simultaneously, all sessions tracked):

```
━━━ Turn 3  (8 sessions)  [recall-t3]
  [H]  sympy/sympy-13091
       cpu     5594ms  ████████████████████████████████████████████
       mtier   320ms   ████████                                      1.10×
  [H]  sympy/sympy-13878
       cpu    12414ms  ████████████████████████████████████████████
       mtier  11200ms  ████████████████████████████████████          1.11×
  ...

  Summary [recall-t3]
  hybrid-cpu      p50    5594ms  p95   12414ms
  hybrid-mtier    p50     320ms  p95     450ms

  MTier vs CPU p50:  1.10×  MTier faster
```

Final recall summary across all recall turns:
```
━━━ Recall Summary (turns 3+) ━━━
  hybrid-cpu     p50    5200ms  p95   12000ms
  hybrid-mtier   p50     310ms  p95     430ms

  Overall speedup  p50:  1.10×  p95:  1.08×  MTier faster
  KV per session : 1572 MB  (~32K tokens)
```

## GPU topology (important)

MTier **must** run on physical GPU 0. `cuMemCreate` allocates on GPU 0; running on GPU 1 routes transfers over NVLink causing a ~3.7× slowdown.

```
GPU 0 → MTier server (port 8002)
GPU 1 → CPU server  (port 8001)
```

This is handled automatically by `start_servers.sh`.

## Caching

Repo source files are cached to `~/.cache/swe_demo/` after first fetch. Delete to force a re-fetch.
