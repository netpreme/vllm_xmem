# Coding-agent reproduction on vLLM

Reproducible coding-agent workloads across vLLM offload tiers (MTier vs CPU DRAM).
**To run a benchmark, see [`benchmarks/`](benchmarks/README.md).**

| Folder | What it does |
|--------|--------------|
| `benchmarks/` | Run / record / replay coding-agent workloads on dual vLLM (mtier + cpu) — primary entrypoint |
| `analysis/` | Snapshot post-processing — generates time-series + offload-dominance figures |
| `monitoring/` | Prometheus + Grafana stack — live dashboards on `:9090` and `:3000` |

| File | What it does |
|------|--------------|
| `start_server.sh` | Bring up one vLLM in a chosen KV mode (`--hybrid-mtier` / `--hybrid-cpu` / etc.) — invoked by the benchmark for each backend |
