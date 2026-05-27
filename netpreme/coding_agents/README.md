# Coding-agent reproduction on vLLM

Reproducible coding-agent workloads across vLLM offload tiers (MTier vs CPU DRAM).
**To run a benchmark, see [`benchmarks/`](benchmarks/README.md).**

## First-time install

```bash
bash setup.sh
```

Installs everything not already provided by vLLM's `pip install -e .`:
Prometheus + psmisc (apt), Grafana (tarball), and `matplotlib pandas datasets` (uv pip).

| Folder | What it does |
|--------|--------------|
| `benchmarks/` | Run / record / replay coding-agent workloads on dual vLLM (mtier + cpu) — primary entrypoint |
| `analysis/` | Snapshot post-processing — generates time-series + offload-dominance figures |
| `monitoring/` | Prometheus + Grafana stack — live dashboards on `:9090` and `:3000` |

| File | What it does |
|------|--------------|
| `setup.sh` | One-time install for the whole stack — delegates to monitoring + analysis setups, adds `psmisc` + `datasets` |
| `server.sh` | Bring up one vLLM in a chosen KV mode (`--hybrid-mtier` / `--hybrid-cpu` / etc.) — invoked by the benchmark for each backend |
