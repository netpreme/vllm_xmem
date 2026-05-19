#!/usr/bin/env python3
"""Plot capture (source) vs replay (duplicate) time-series for each concurrency
in a single 5-row × 2-column grid.

Rows: concurrency levels c=16, 15, 14, 13, 12 (descending top→bottom).
Cols: (left) capture run — mtier-only;  (right) replay run — mtier + cpu.

Metrics per panel:
    HBM hit       — green
    Offload hit   — purple
    Recompute     — red
    HBM KV util   — gray

Replay panels overlay both setups: mtier = solid line, cpu = dashed line.
"""
from __future__ import annotations

import csv
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests


RESULTS = Path("/home/ubuntu/vllm_xmem/netpreme/coding_agents/benchmarks/results_benchmarks")
MASTER  = RESULTS / "replay_metrics_master.csv"
OUTFILE = RESULTS / "capture_vs_replay_grid.png"


# ── throwaway Prometheus over a snapshot dir ────────────────────────────────
def start_prom(snapshot: Path, port: int) -> subprocess.Popen:
    cfg = Path(tempfile.mkstemp(prefix="prom_grid_", suffix=".yml")[1])
    cfg.write_text("global:\n  scrape_interval: 60s\n")
    log = f"/tmp/prom_grid_{port}.log"
    proc = subprocess.Popen(
        ["prometheus", f"--config.file={cfg}", f"--storage.tsdb.path={snapshot}",
         f"--web.listen-address=:{port}", "--storage.tsdb.retention.time=10y"],
        stdout=open(log, "w"), stderr=subprocess.STDOUT, start_new_session=True,
    )
    t0 = time.time()
    while time.time() - t0 < 30:
        try:
            if requests.get(f"http://localhost:{port}/-/ready", timeout=2).status_code == 200:
                return proc
        except Exception:
            pass
        time.sleep(0.3)
    raise RuntimeError(f"prometheus did not become ready on :{port}")


def stop_prom(proc):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM); proc.wait(timeout=5)
    except Exception:
        try: proc.kill()
        except Exception: pass


def query_range(prom_url: str, query: str, t_start: float, t_end: float,
                step: int = 5) -> pd.Series:
    r = requests.get(f"{prom_url}/api/v1/query_range",
                     params={"query": query, "start": t_start, "end": t_end,
                             "step": step}, timeout=60)
    data = r.json().get("data", {}).get("result", [])
    if not data:
        return pd.Series(dtype=float)
    pts = data[0]["values"]
    idx = [float(p[0]) for p in pts]
    vals = [float(p[1]) if p[1] not in ("NaN","+Inf","-Inf") else float("nan") for p in pts]
    return pd.Series(vals, index=idx)


def metrics_for_setup(prom_url: str, setup_label: str, t_start: float, t_end: float,
                      step: int = 5) -> dict[str, pd.Series]:
    """Use the same metric basis as analyze_snapshot.fig1_timeseries:
    per-prompt-token source breakdown (sums to 100%) from
    `vllm:prompt_tokens_by_source_total`. Selected by `setup` label."""
    win = "30s"
    def pct(source: str) -> str:
        return (
            f'sum(rate(vllm:prompt_tokens_by_source_total'
            f'{{setup="{setup_label}",source="{source}"}}[{win}])) / '
            f'clamp_min(sum(rate(vllm:prompt_tokens_by_source_total'
            f'{{setup="{setup_label}"}}[{win}])), 1e-9)'
        )
    out = {}
    out["hbm_hit"]   = query_range(prom_url, pct("local_cache_hit"),      t_start, t_end, step)
    out["off_hit"]   = query_range(prom_url, pct("external_kv_transfer"), t_start, t_end, step)
    out["recompute"] = query_range(prom_url, pct("local_compute"),        t_start, t_end, step)
    out["hbm_use"]   = query_range(prom_url,
        f'avg_over_time(vllm:kv_cache_usage_perc{{setup="{setup_label}"}}[{win}])',
        t_start, t_end, step)
    return out


def gather() -> dict:
    """Return nested dict keyed by (concurrency, kind). kind ∈ {'capture','replay'}.
    Each entry: { setup_label: {'hbm_hit':Series, 'off_hit':..., ...}, 't0': ..., 't_end':... }"""
    rows = list(csv.DictReader(open(MASTER)))
    pairs: dict[int, dict] = {}
    for r in rows:
        c = int(r["concurrency"])
        if c in pairs:
            continue
        cap_meta_path = Path(r["capture_dir"]) / "capture_meta.json"
        cap_meta = json.loads(cap_meta_path.read_text())
        pairs[c] = {
            "capture_snap": Path(cap_meta["sweep_run_dir"]) / "prom_snapshot",
            "capture_setup":     cap_meta["setup"],
            "capture_port":      cap_meta["vllm_port"],
            "capture_t_start":   float(cap_meta["t_level_start_unix"]),
            "capture_t_end":     float(cap_meta["t_level_end_unix"]),
            "replay_snap":       Path(r["level_dir"]) / "prom_snapshot",
            "replay_t_start":    float(r["t_start_unix"]),
            "replay_t_end":      float(r["t_end_unix"]),
        }

    out: dict = {}
    for c in sorted(pairs):
        p = pairs[c]
        out[c] = {"capture": {}, "replay": {}}

        # capture (mtier only, selected by setup label)
        print(f"  c={c}  capture snap → starting prom ...", flush=True)
        proc = start_prom(p["capture_snap"], port=9105)
        try:
            metrics = metrics_for_setup(f"http://localhost:9105", p["capture_setup"],
                                         p["capture_t_start"], p["capture_t_end"], step=5)
            out[c]["capture"][p["capture_setup"]] = {
                **metrics,
                "t_start": p["capture_t_start"], "t_end": p["capture_t_end"]
            }
        finally:
            stop_prom(proc)

        # replay: both setups distinguished by `setup` label
        print(f"  c={c}  replay snap → starting prom ...", flush=True)
        proc = start_prom(p["replay_snap"], port=9106)
        try:
            for setup in ("hybrid-mtier", "hybrid-cpu"):
                metrics = metrics_for_setup(f"http://localhost:9106", setup,
                                             p["replay_t_start"], p["replay_t_end"], step=5)
                out[c]["replay"][setup] = {
                    **metrics,
                    "t_start": p["replay_t_start"], "t_end": p["replay_t_end"],
                }
        finally:
            stop_prom(proc)

    return out


# ── plotting ────────────────────────────────────────────────────────────────
_COLOR = {
    "hbm_hit":   "#2ca02c",   # green
    "off_hit":   "#6a51a3",   # purple
    "recompute": "#d62728",   # red
    "hbm_use":   "#7f7f7f",   # gray
}
_LABEL = {
    "hbm_hit":   "HBM hit",
    "off_hit":   "Offload hit",
    "recompute": "Recompute",
    "hbm_use":   "HBM util",
}


def plot_panel(ax, series_by_setup: dict, t0_anchor: float, title: str,
               max_xs: int = 1500) -> None:
    """Plot all metrics for one cell. If multiple setups, distinguish by linestyle."""
    setup_linestyle = {"hybrid-mtier": "-", "hybrid-cpu": "--"}
    plotted_any = False
    xs_max_seen = 0
    for setup, m in series_by_setup.items():
        ls = setup_linestyle.get(setup, "-")
        for key in ("hbm_hit", "off_hit", "recompute", "hbm_use"):
            s = m[key]
            if s.empty:
                continue
            xs = [t - t0_anchor for t in s.index]
            ys = [v * 100 for v in s.values]  # convert to %
            ax.plot(xs, ys,
                    color=_COLOR[key],
                    linestyle=ls,
                    linewidth=1.2,
                    label=f"{_LABEL[key]} ({'mtier' if setup=='hybrid-mtier' else 'cpu' if setup=='hybrid-cpu' else setup})",
                    alpha=0.95 if ls == "-" else 0.85)
            plotted_any = True
            if xs:
                xs_max_seen = max(xs_max_seen, max(xs))
    ax.set_ylim(-2, 102)
    ax.set_xlim(0, min(max_xs, xs_max_seen + 30) if xs_max_seen else max_xs)
    ax.grid(True, alpha=0.25)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("Percent (%)", fontsize=9)


def main():
    print("Gathering metrics from snapshots ...", flush=True)
    data = gather()

    concurrencies = sorted(data.keys(), reverse=True)  # c=16 top → c=12 bottom
    n_rows = len(concurrencies)
    fig, axes = plt.subplots(n_rows, 3, figsize=(22, 3.8 * n_rows), sharex=False)

    for i, c in enumerate(concurrencies):
        cap = data[c]["capture"]
        rep = data[c]["replay"]

        # col 0: original capture (mtier-only)
        cap_setup = next(iter(cap))
        t0_cap = cap[cap_setup]["t_start"]
        plot_panel(axes[i][0], cap, t0_cap,
                   f"c={c}  ORIGINAL trace  (mtier, 20-min sustained)",
                   max_xs=1300)

        # col 1: reproduction mtier (replay against mtier backend)
        t0_rep = next(iter(rep.values()))["t_start"]
        mtier_only = {"hybrid-mtier": rep["hybrid-mtier"]} if "hybrid-mtier" in rep else {}
        plot_panel(axes[i][1], mtier_only, t0_rep,
                   f"c={c}  REPRODUCTION on mtier",
                   max_xs=1600)

        # col 2: reproduction cpu (replay against cpu backend)
        cpu_only = {"hybrid-cpu": rep["hybrid-cpu"]} if "hybrid-cpu" in rep else {}
        plot_panel(axes[i][2], cpu_only, t0_rep,
                   f"c={c}  REPRODUCTION on cpu",
                   max_xs=1600)

        for j in range(3):
            axes[i][j].set_xlabel("Time (s, from level start)", fontsize=10)

    # Shared legend (4 metrics; no dash distinction needed since mtier/cpu are split)
    legend_handles = [
        plt.Line2D([], [], color=_COLOR["hbm_hit"],   lw=2, label="HBM hit"),
        plt.Line2D([], [], color=_COLOR["off_hit"],   lw=2, label="Offload hit"),
        plt.Line2D([], [], color=_COLOR["recompute"], lw=2, label="Recompute"),
        plt.Line2D([], [], color=_COLOR["hbm_use"],   lw=2, label="HBM KV util"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=4, fontsize=11,
               bbox_to_anchor=(0.5, 0.997), frameon=True)
    fig.suptitle("Original (capture) vs Reproduction (replay) — KV / offload time-series per concurrency",
                 fontsize=14, y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.965])

    OUTFILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTFILE, dpi=110, bbox_inches="tight")
    print(f"Wrote {OUTFILE}")
    plt.close(fig)


if __name__ == "__main__":
    main()
