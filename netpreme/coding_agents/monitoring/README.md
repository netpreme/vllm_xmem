# Monitoring

Prometheus + Grafana stack for vLLM + MTier observability.

## First-time setup

Run once to install Prometheus and Grafana:
```bash
bash monitoring/setup_monitoring.sh
```

## Start monitoring

```bash
bash monitoring/start_monitoring.sh
```

Starts three processes:
| Process | Port | Purpose |
|---------|------|---------|
| Prometheus | 9090 | Scrapes vLLM + KV exporter metrics |
| Grafana | 3000 | Dashboards (auto-provisioned, no login needed) |
| `kv_exporter.py` | 9091 | KV offload bandwidth metrics |

Open **http://localhost:3000** — two dashboards auto-load:
- **vLLM xmem** — TTFT, E2E, queue wait, cache hit%, KV offload bandwidth over time
- **MTier vs CPU — Live Demo** — side-by-side real-time comparison during `demo/` runs

SSH tunnel if running on a remote machine:
```bash
ssh -L 3000:localhost:3000 -L 9090:localhost:9090 ubuntu@<host>
```

---

## Scripts

### `start_monitoring.sh`
Starts Prometheus, Grafana, and `kv_exporter.py`. Wipes Prometheus data on each start for a fresh slate. Use `--keep` to retain existing data:
```bash
bash monitoring/start_monitoring.sh --keep
```

### `kv_exporter.py`
Tails the Dynamo worker log and exposes KV offload metrics (CPU↔GPU bytes transferred, TTFT) as Prometheus counters on port 9091. Also scrapes the vLLM worker for prefix cache hit rate.
```bash
python3 monitoring/kv_exporter.py --log /tmp/dynamo_worker_8000.log --port 9091
```

### `watch_requests.py`
Live per-request terminal table. Polls Prometheus every 0.5s and prints one row per completed request — ISL, OSL, TTFT, ITL, E2E, GPU/CPU cache hit rates, and KV transfer bytes. Useful for single-user debugging.
```bash
python3 monitoring/watch_requests.py
```

### `parse_stream.py`
Parses `claude --output-format stream-json` from stdin into a per-turn ISL/OSL/tool table. Called automatically by `run_claude.sh` — not meant to be run directly.

### `prometheus.yml`
Scrape config: vLLM frontend (:8000), vLLM worker (:8081), KV exporter (:9091).

### `grafana_provisioning/`
Auto-provisioned Grafana datasources and dashboards. Changes here take effect on next `start_monitoring.sh`.
