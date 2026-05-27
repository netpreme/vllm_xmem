# Monitoring

Prometheus + Grafana stack for vLLM + MTier observability.

## First-time setup

Installed by the top-level setup script (see `coding_agents/setup.sh`):
```bash
bash ../setup.sh
```

## Start monitoring

```bash
bash monitoring/run.sh
```

Starts three processes:
| Process | Port | Purpose |
|---------|------|---------|
| Prometheus | 9090 | Scrapes vLLM + GPU metrics |
| Grafana | 3000 | Dashboards (auto-provisioned, no login needed) |
| `gpu_metrics_recorder.py` | 9092 | Per-GPU utilization, memory, power, temperature |

Open **http://localhost:3000** — dashboards auto-load with TTFT, E2E, queue wait, cache hit%, and per-GPU utilization/memory over time.

SSH tunnel if running on a remote machine:
```bash
ssh -L 3000:localhost:3000 -L 9090:localhost:9090 ubuntu@<host>
```

---

## Scripts

### `run.sh`
Starts Prometheus, Grafana, and `gpu_metrics_recorder.py`. Wipes Prometheus data on each start for a fresh slate. Use `--keep` to retain existing data:
```bash
bash monitoring/run.sh --keep
```

### `gpu_metrics_recorder.py`
Polls `nvidia-smi` and exposes per-GPU utilization, memory, and power as Prometheus gauges on port 9092. Started by `run.sh` and also spawned directly by the benchmark utilities.
```bash
python3 monitoring/gpu_metrics_recorder.py --port 9092 --interval 1.0
```

### `prometheus.yml`
Scrape config: vLLM frontend (:8000), vLLM worker (:8081), GPU metrics recorder (:9092).

### `grafana_provisioning/`
Auto-provisioned Grafana datasources and dashboards. Changes here take effect on next `run.sh`.
