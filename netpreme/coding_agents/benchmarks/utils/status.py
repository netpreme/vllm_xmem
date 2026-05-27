"""Live per-setup status thread + end-of-level averages.

LiveStatus runs a background thread that polls Prometheus every interval
seconds and prints one row per setup (TTFT/ITL/E2E/out tok/s + cache hit
rates). On a TTY it uses ANSI cursor-up to overwrite in place; on a file
log it emits a fresh block every 10s.

print_averages() prints the end-of-level summary block.
"""
import sys
import threading
import time
from urllib.parse import urlparse

import requests

from utils.colors import DIM, OFF, setup_color, setup_short
from lifecycle import PROM_URL


def prom_query(query: str) -> float | None:
    """Single-value Prometheus instant query. Returns float or None on failure."""
    try:
        r = requests.get(f"{PROM_URL}/api/v1/query",
                         params={"query": query}, timeout=3)
        d = r.json().get("data", {}).get("result", [])
        if not d:
            return None
        v = d[0]["value"][1]
        return None if v in ("NaN", "+Inf", "-Inf") else float(v)
    except Exception:
        return None


def query_setup_metrics(spec: dict, window: str = "30s") -> dict:
    """Query all live metrics for one setup over `window`."""
    sel = '{instance="localhost:%s"}' % urlparse(spec["base_url"]).port
    return {
        "ttft":    prom_query(f'histogram_quantile(0.5,  sum by (le) (rate(vllm:time_to_first_token_seconds_bucket{sel}[{window}])))'),
        "itl":     prom_query(f'histogram_quantile(0.5,  sum by (le) (rate(vllm:inter_token_latency_seconds_bucket{sel}[{window}])))'),
        "e2e":     prom_query(f'histogram_quantile(0.5,  sum by (le) (rate(vllm:e2e_request_latency_seconds_bucket{sel}[{window}])))'),
        "out_tps": prom_query(f'sum(rate(vllm:generation_tokens_total{sel}[{window}]))'),
        "hbm_use": prom_query(f'avg(vllm:kv_cache_usage_perc{sel})'),
        "hbm_hit": prom_query(f'rate(vllm:prefix_cache_hits_total{sel}[{window}]) / clamp_min(rate(vllm:prefix_cache_queries_total{sel}[{window}]), 1e-9)'),
        "off_hit": prom_query(f'rate(vllm:external_prefix_cache_hits_total{sel}[{window}]) / clamp_min(rate(vllm:prefix_cache_queries_total{sel}[{window}]), 1e-9)'),
        "turns":   prom_query(f'sum(vllm:e2e_request_latency_seconds_count{sel})'),
    }


# ── formatting helpers ────────────────────────────────────────────────────────
def _ms(v):  return f"{int(v*1000)}ms" if v is not None else "-"
def _s(v):   return f"{v:.1f}s"        if v is not None else "-"
def _tps(v): return f"{v:.0f}"         if v is not None else "-"
def _pct(v): return f"{v*100:.1f}%"    if v is not None else "-"


def _fmt_status_line(spec: dict, m: dict, task_counters: dict) -> str:
    """`[GPU0+Mtier]  tasks=4 ok=2 turns=37 ttft=89ms itl=12ms e2e=2.3s ...`"""
    color = setup_color(spec["setup"])
    gpu   = f"GPU{spec.get('gpus', '?')}"
    name  = setup_short(spec["setup"])
    turns = int(m["turns"]) if m["turns"] is not None else 0
    return (
        f"{color}[{gpu}+{name:5s}]{OFF}  "
        f"tasks={task_counters.get('n_active',0):>2}/{task_counters.get('n_done',0):>3}  "
        f"turns={turns:>5}  "
        f"ttft={_ms(m['ttft']):>6}  itl={_ms(m['itl']):>5}  e2e={_s(m['e2e']):>6}  "
        f"out={_tps(m['out_tps']):>4}t/s  "
        f"HBM_use={_pct(m['hbm_use']):>6}  HBM_hit={_pct(m['hbm_hit']):>6}  "
        f"off_hit={_pct(m['off_hit']):>5}"
    )


class LiveStatus:
    """Thread that prints per-setup metrics every `interval` seconds.

    Usage:
        status = LiveStatus(setup_specs, shared_state, t_start_mono).start()
        ... do work ...
        status.stop()
    """

    def __init__(
        self,
        setup_specs: list[dict],
        shared_state: dict,
        t_start_mono: float,
        interval: float = 1.0,
    ):
        self.setup_specs   = setup_specs
        self.shared_state  = shared_state
        self.t_start_mono  = t_start_mono
        self.interval      = interval
        self._stop_evt     = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> "LiveStatus":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop_evt.set()
        time.sleep(0.5)  # let the last redraw drain

    def _run(self) -> None:
        is_tty = sys.stdout.isatty()
        interval = self.interval if is_tty else 10.0
        n = len(self.setup_specs)
        first = True
        while not self._stop_evt.wait(interval):
            elapsed = int(time.monotonic() - self.t_start_mono)
            if is_tty:
                if not first:
                    sys.stdout.write(f"\033[{n+1}A")
                first = False
                sys.stdout.write(
                    f"\033[2K\r  {DIM}── live  t={elapsed:>3}s  "
                    f"(window=30s rolling){OFF}\n"
                )
                for spec in self.setup_specs:
                    m  = query_setup_metrics(spec)
                    tc = self.shared_state.get(spec["setup"], {})
                    sys.stdout.write(f"\033[2K\r  {_fmt_status_line(spec, m, tc)}\n")
                sys.stdout.flush()
            else:
                print(f"\n  ── live  t={elapsed:>3}s  (window=30s rolling)", flush=True)
                for spec in self.setup_specs:
                    m  = query_setup_metrics(spec)
                    tc = self.shared_state.get(spec["setup"], {})
                    print(f"  {_fmt_status_line(spec, m, tc)}", flush=True)


def print_averages(
    setup_specs: list[dict],
    t_start_unix: float,
    t_end_unix: float,
    per_setup: dict,
) -> None:
    """End-of-level summary: per-setup averages over the full run window."""
    duration_s = max(15, int(t_end_unix - t_start_unix))
    window = f"{duration_s}s"
    print(f"\n  {DIM}── Averages  (over {duration_s}s window){OFF}", flush=True)
    for spec in setup_specs:
        sel = '{instance="localhost:%s"}' % urlparse(spec["base_url"]).port
        avg_ttft = prom_query(f'increase(vllm:time_to_first_token_seconds_sum{sel}[{window}]) / clamp_min(increase(vllm:time_to_first_token_seconds_count{sel}[{window}]), 1)')
        avg_itl  = prom_query(f'increase(vllm:inter_token_latency_seconds_sum{sel}[{window}]) / clamp_min(increase(vllm:inter_token_latency_seconds_count{sel}[{window}]), 1)')
        avg_e2e  = prom_query(f'increase(vllm:e2e_request_latency_seconds_sum{sel}[{window}]) / clamp_min(increase(vllm:e2e_request_latency_seconds_count{sel}[{window}]), 1)')
        avg_tps  = prom_query(f'rate(vllm:generation_tokens_total{sel}[{window}])')
        avg_hbm  = prom_query(f'avg_over_time(vllm:kv_cache_usage_perc{sel}[{window}])')
        hbm_hit  = prom_query(f'increase(vllm:prefix_cache_hits_total{sel}[{window}]) / clamp_min(increase(vllm:prefix_cache_queries_total{sel}[{window}]), 1)')
        off_hit  = prom_query(f'increase(vllm:external_prefix_cache_hits_total{sel}[{window}]) / clamp_min(increase(vllm:prefix_cache_queries_total{sel}[{window}]), 1)')
        turns    = prom_query(f'sum(increase(vllm:e2e_request_latency_seconds_count{sel}[{window}]))')
        n_ok     = per_setup.get(spec["setup"], {}).get("n_tasks_completed", 0)
        color = setup_color(spec["setup"])
        gpu   = f"GPU{spec.get('gpus', '?')}"
        name  = setup_short(spec["setup"])
        print(
            f"  {color}[{gpu}+{name:5s}]{OFF}  "
            f"tasks_done={n_ok:>3}  turns={int(turns) if turns else 0:>5}  "
            f"avg ttft={_ms(avg_ttft):>6}  itl={_ms(avg_itl):>5}  e2e={_s(avg_e2e):>6}  "
            f"out={_tps(avg_tps):>4}t/s  "
            f"HBM_use={_pct(avg_hbm):>6}  HBM_hit={_pct(hbm_hit):>6}  "
            f"off_hit={_pct(off_hit):>5}",
            flush=True,
        )
