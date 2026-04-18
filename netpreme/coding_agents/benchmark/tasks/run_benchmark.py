#!/usr/bin/env python3
"""
Concurrent SWE-bench benchmark: measures TTFT, E2E, ITL and throughput
at varying concurrency levels against a running vLLM / Dynamo server.

For each concurrency level C:
  1. Pre-build git workspaces (outside the measurement window)
  2. Fire C tasks simultaneously via ThreadPoolExecutor
  3. After all tasks finish, query Prometheus for latency percentiles
     over the exact run window
  4. Save per-concurrency and summary results

Prometheus is the source of truth for latency percentiles (p50/p90/p95/p99).
Client-side timing covers per-task wall time and token totals.

Usage:
    python3 run_benchmark.py                             # C=1,2,4,8,16,32,64,128, 1 batch each
    python3 run_benchmark.py --concurrency 1 2 4 8      # specific levels only
    python3 run_benchmark.py --tasks-per-level 8        # run 8 tasks per level (more stats)
    python3 run_benchmark.py --label tp4 --concurrency 1 2 4 8 16
    python3 run_benchmark.py --start 0 --end 50         # restrict SWE-bench slice
    python3 run_benchmark.py --no-clone                 # skip repo setup, run in cwd

Results saved to:
    tasks/results/bench_<label>_<timestamp>/
        c001.json, c002.json, ...   per-concurrency detail
        summary.json                all levels consolidated
        summary.csv                 flat table for plotting / import
"""

import argparse
import concurrent.futures
import csv
import json
import os
import queue as _queue
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR     = Path(__file__).resolve().parent
BENCHMARKS_DIR = SCRIPT_DIR.parent
ENV_FILE       = BENCHMARKS_DIR.parent / ".env"
WORKSPACE_ROOT = Path("/tmp/swe_workspaces")
RESULTS_DIR    = SCRIPT_DIR / "results"

DEFAULT_MODEL        = "qwen/qwen3-coder-30b-a3b-instruct-fp8"
DEFAULT_PORT         = "8000"
DEFAULT_PROM         = "http://localhost:9090"
PROM_SCRAPE_BUFFER_S = 5   # seconds to wait after run before querying Prometheus
CLONE_WORKERS        = 16   # parallel git-clone threads (safe for GitHub rate limits)


# ── environment ───────────────────────────────────────────────────────────────
def load_env(path: Path) -> None:
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if key and key not in os.environ:
                os.environ[key] = value.strip()


# ── Prometheus helpers ────────────────────────────────────────────────────────
def prom_scalar(prom: str, q: str) -> Optional[float]:
    try:
        r = requests.get(f"{prom}/api/v1/query", params={"query": q}, timeout=5)
        results = r.json()["data"]["result"]
        if results:
            v = results[0]["value"][1]
            return None if v in ("NaN", "Inf", "-Inf") else float(v)
    except Exception:
        pass
    return None


def query_percentiles(prom: str, bucket_metric: str, duration_s: float,
                      unit_multiplier: float = 1000.0) -> dict[str, Optional[float]]:
    """
    Query p50/p90/p95/p99 from a Prometheus histogram over the last duration_s.
    unit_multiplier=1000 converts seconds → ms; use 1.0 for token counts.
    """
    d = max(15, int(duration_s))   # minimum 15s window so rate() has data
    result: dict[str, Optional[float]] = {}
    for q, label in [(0.5, "p50"), (0.9, "p90"), (0.95, "p95"), (0.99, "p99")]:
        promql = f"histogram_quantile({q}, sum(rate({bucket_metric}[{d}s])) by (le))"
        v = prom_scalar(prom, promql)
        result[label] = round(v * unit_multiplier, 1) if v is not None else None
    return result


def query_rate(prom: str, metric: str, duration_s: float) -> Optional[float]:
    d = max(15, int(duration_s))
    v = prom_scalar(prom, f"sum(rate({metric}[{d}s]))")
    return round(v, 3) if v is not None else None


def query_hit_rate_pct(prom: str, hits_metric: str, queries_metric: str,
                       duration_s: float) -> Optional[float]:
    """Compute hit-rate % from a hits/queries counter pair over the run window."""
    d = max(15, int(duration_s))
    hits    = prom_scalar(prom, f"increase({hits_metric}[{d}s])")
    queries = prom_scalar(prom, f"increase({queries_metric}[{d}s])")
    if hits is None or queries is None or queries == 0:
        return None
    return round(hits / queries * 100, 1)


# ── client-side percentiles ───────────────────────────────────────────────────
def _percentile(data: list[float], p: float) -> Optional[float]:
    if not data:
        return None
    data = sorted(data)
    idx = (len(data) - 1) * p
    lo  = int(idx)
    hi  = min(lo + 1, len(data) - 1)
    return round(data[lo] + (data[hi] - data[lo]) * (idx - lo), 2)


def _pct_dict(data: list[float]) -> dict[str, Optional[float]]:
    return {
        "p50": _percentile(data, 0.50),
        "p90": _percentile(data, 0.90),
        "p95": _percentile(data, 0.95),
        "p99": _percentile(data, 0.99),
    }


# ── dataset ───────────────────────────────────────────────────────────────────
def load_dataset():
    from datasets import load_dataset
    print("Loading SWE-bench Verified ...", flush=True)
    return load_dataset("princeton-nlp/SWE-bench_Verified", split="test")


# ── workspace ─────────────────────────────────────────────────────────────────
def setup_workspace(instance: dict) -> Path:
    instance_id = instance["instance_id"]
    repo        = instance["repo"]
    base_commit = instance["base_commit"]
    workspace   = WORKSPACE_ROOT / instance_id

    if workspace.exists():
        subprocess.run(["git", "checkout", "-f", base_commit],
                       cwd=workspace, check=True, capture_output=True)
        subprocess.run(["git", "clean", "-fd"],
                       cwd=workspace, check=True, capture_output=True)
    else:
        WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth=50",
                        f"https://github.com/{repo}.git", str(workspace)], check=True)
        subprocess.run(["git", "fetch", "--depth=50", "origin", base_commit],
                       cwd=workspace, check=True, capture_output=True)
        subprocess.run(["git", "checkout", base_commit],
                       cwd=workspace, check=True, capture_output=True)
    return workspace


# ── single task runner ────────────────────────────────────────────────────────
def run_task(instance: dict, workdir: Path, model: str, base_url: str) -> dict:
    """
    Run one SWE-bench task.  Returns per-task timing and token summary.
    Called concurrently — must be thread-safe (subprocess + independent state only).
    """
    instance_id = instance["instance_id"]
    problem     = instance["problem_statement"]
    t0 = time.monotonic()

    env = {
        **os.environ,
        "ANTHROPIC_BASE_URL":             base_url,
        "ANTHROPIC_API_KEY":              "dummy",
        "ANTHROPIC_AUTH_TOKEN":           "dummy",
        "ANTHROPIC_DEFAULT_OPUS_MODEL":   model,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL":  model,
    }

    try:
        proc = subprocess.run(
            ["claude", "--model", model,
             "--dangerously-skip-permissions",
             "--verbose",
             "--output-format", "stream-json",
             "-p", problem],
            cwd=str(workdir),
            env=env,
            capture_output=True,
            text=True,
            timeout=900,   # 15 min hard cap per task
        )
        returncode = proc.returncode
        stdout     = proc.stdout
    except subprocess.TimeoutExpired:
        return {
            "instance_id": instance_id,
            "wall_time_s": round(time.monotonic() - t0, 2),
            "status": "timeout",
            "n_turns": 0,
            "input_tokens": None, "output_tokens": None, "cache_read_tokens": None,
        }
    except Exception as e:
        return {
            "instance_id": instance_id,
            "wall_time_s": round(time.monotonic() - t0, 2),
            "status": f"error:{e}",
            "n_turns": 0,
            "input_tokens": None, "output_tokens": None, "cache_read_tokens": None,
        }

    wall_time_s = round(time.monotonic() - t0, 2)

    # Parse stream-json for turn count and final token usage
    n_turns     = 0
    seen_isls: set = set()
    final_usage: dict = {}

    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        etype = event.get("type")
        if etype == "assistant":
            isl = event.get("message", {}).get("usage", {}).get("input_tokens")
            if isl not in seen_isls:
                seen_isls.add(isl)
                n_turns += 1
        elif etype == "result":
            final_usage = event.get("usage", {})

    return {
        "instance_id":      instance_id,
        "wall_time_s":      wall_time_s,
        "status":           "ok" if returncode == 0 else f"exit:{returncode}",
        "n_turns":          n_turns,
        "input_tokens":     final_usage.get("input_tokens"),
        "output_tokens":    final_usage.get("output_tokens"),
        "cache_read_tokens": final_usage.get("cache_read_input_tokens"),
    }


# ── concurrency-level runner ──────────────────────────────────────────────────
def run_level(
    concurrency: int,
    initial_specs: list[dict],          # first C specs, already cloned
    extra_queue: "_queue.Queue[dict | None]",  # remaining specs fed by bg cloner; None = sentinel
    n_total: int,                        # total expected tasks (for display)
    model: str,
    base_url: str,
    prom: str,
    sustained_min_s: float = 0,
) -> dict:
    sustained = sustained_min_s > 0
    mode = f"sustained (stop after {sustained_min_s:.0f}s or all {n_total} tasks done)" if sustained else "batch"
    print(f"\n  ── Concurrency {concurrency:>3}  ({n_total} tasks, {mode}) ──")

    t_start      = time.monotonic()
    task_results = []

    spec_iter  = iter(initial_specs)
    queue_done = False

    def _next_spec() -> "dict | None":
        nonlocal queue_done
        s = next(spec_iter, None)
        if s is not None:
            return s
        if queue_done:
            return None
        try:
            s = extra_queue.get_nowait()
            if s is None:
                queue_done = True
                return None
            return s
        except _queue.Empty:
            return None  # background cloner hasn't finished yet

    ex     = concurrent.futures.ThreadPoolExecutor(max_workers=concurrency)
    active: dict = {}

    def _fill():
        while len(active) < concurrency:
            spec = _next_spec()
            if spec is None:
                break
            f = ex.submit(run_task, spec["instance"], spec["workdir"], model, base_url)
            active[f] = spec

    _fill()

    try:
        stop = False
        while active and not stop:
            done, _ = concurrent.futures.wait(
                list(active.keys()), timeout=1.0,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                res = future.result()
                del active[future]
                task_results.append(res)
                elapsed = time.monotonic() - t_start
                status_icon = "✓" if res["status"] == "ok" else "✗"
                print(f"    {status_icon} {res['instance_id']:<45}"
                      f"  {res['wall_time_s']:>6.0f}s"
                      f"  {res['n_turns']:>3} turns"
                      f"  out={res['output_tokens'] or '?'}"
                      f"  [{res['status']}]  elapsed={elapsed:.0f}s")

                if sustained and elapsed >= sustained_min_s:
                    n_done = len(task_results)
                    print(f"    Stop condition met: {elapsed:.0f}s elapsed, {n_done} tasks done. Cancelling queued tasks.")
                    for f in list(active.keys()):
                        f.cancel()
                    active.clear()
                    stop = True
                    break

            if not stop:
                # Fill empty slots — picks up any new workspaces the bg cloner deposited
                _fill()
    finally:
        ex.shutdown(wait=False, cancel_futures=True)

    t_end     = time.monotonic()
    duration_s = t_end - t_start

    # Wait for Prometheus to scrape the last metrics before querying
    print(f"    Waiting {PROM_SCRAPE_BUFFER_S}s for Prometheus scrape ...", flush=True)
    time.sleep(PROM_SCRAPE_BUFFER_S)

    # ── Latency percentiles from Prometheus histograms ────────────────────────
    ttft_ms    = query_percentiles(prom, "vllm:time_to_first_token_seconds_bucket", duration_s)
    e2e_ms     = query_percentiles(prom, "vllm:e2e_request_latency_seconds_bucket", duration_s)
    itl_ms     = query_percentiles(prom, "vllm:inter_token_latency_seconds_bucket", duration_s)
    queue_ms   = query_percentiles(prom, "vllm:request_queue_time_seconds_bucket",  duration_s)
    prefill_ms = query_percentiles(prom, "vllm:request_prefill_time_seconds_bucket", duration_s)

    # ── Token length distributions (histograms, unit = tokens) ───────────────
    prompt_tok = query_percentiles(prom, "vllm:request_prompt_tokens_bucket",     duration_s, unit_multiplier=1.0)
    gen_tok    = query_percentiles(prom, "vllm:request_generation_tokens_bucket", duration_s, unit_multiplier=1.0)

    # ── Throughput from Prometheus ────────────────────────────────────────────
    req_rps = query_rate(prom, "vllm:e2e_request_latency_seconds_count", duration_s)
    out_tps        = query_rate(prom, "vllm:generation_tokens_total", duration_s)
    total_itl_s    = prom_scalar(prom, f"increase(vllm:inter_token_latency_seconds_sum[{max(15,int(duration_s))}s])")
    total_osl      = prom_scalar(prom, f"increase(vllm:generation_tokens_total[{max(15,int(duration_s))}s])")

    # ── Cache metrics ────────────────────────────────────────────────────────
    d = max(15, int(duration_s))
    # GPU (local) hit rate: blocks found already resident in GPU KV cache
    gpu_hit_rate = query_hit_rate_pct(
        prom, "vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total", duration_s)
    # CPU (external) hit rate: blocks found in Dynamo's offloaded CPU tier
    cpu_hit_rate = query_hit_rate_pct(
        prom, "vllm:external_prefix_cache_hits_total", "vllm:external_prefix_cache_queries_total", duration_s)
    # Combined hit rate: prompt tokens served from any cache tier / total prompt tokens
    combined_hit_rate = query_hit_rate_pct(
        prom, "vllm:prompt_tokens_cached_total", "vllm:prompt_tokens_total", duration_s)
    kv_cache_used = prom_scalar(prom, f"avg_over_time(vllm:kv_cache_usage_perc[{d}s])")

    # ── CPU↔GPU KV offload bytes ──────────────────────────────────────────────
    kv_cpu_to_gpu_bytes = prom_scalar(prom, f"increase(vllm_kv_cpu_to_gpu_bytes_total[{d}s])")
    kv_gpu_to_cpu_bytes = prom_scalar(prom, f"increase(vllm_kv_gpu_to_cpu_bytes_total[{d}s])")
    preemptions         = prom_scalar(prom, f"increase(vllm:num_preemptions_total[{d}s])")

    # ── Client-side aggregates ────────────────────────────────────────────────
    ok   = [r for r in task_results if r["status"] == "ok"]
    n_ok = len(ok)

    def safe_avg(vals):
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 1) if vals else None

    # Client-side per-task distributions
    wall_s_list   = [r["wall_time_s"] for r in ok]
    n_turns_list  = [r["n_turns"]     for r in ok if r["n_turns"] > 0]
    tpt_list      = [r["wall_time_s"] / r["n_turns"] for r in ok if r["n_turns"] > 0]

    level = {
        "concurrency":  concurrency,
        "n_tasks":      n_total,
        "n_ok":         n_ok,
        "wall_time_s":  round(duration_s, 1),

        "throughput": {
            "tasks_per_sec":        round(n_ok / duration_s, 4) if n_ok else 0,
            "requests_per_sec":     req_rps,    # from Prometheus (more accurate)
            "output_tokens_per_sec": out_tps,
        },

        "per_task": {
            "avg_wall_time_s":     safe_avg(r["wall_time_s"]   for r in ok),
            "avg_n_turns":         safe_avg(r["n_turns"]        for r in ok),
            "avg_output_tokens":   safe_avg(r["output_tokens"]  for r in ok),
            "avg_input_tokens":    safe_avg(r["input_tokens"]   for r in ok),
            "total_turns":         sum(r["n_turns"]        or 0  for r in ok),
            "total_output_tokens": sum(r["output_tokens"]  or 0  for r in ok),
            "total_input_tokens":  sum(r["input_tokens"]   or 0  for r in ok),
            "total_itl_s":         round(total_itl_s, 1)  if total_itl_s is not None else None,
            "total_osl":           round(total_osl)        if total_osl    is not None else None,
        },

        # Prometheus-sourced latency distributions (per vLLM request = per turn)
        "ttft_ms":    ttft_ms,
        "prefill_ms": prefill_ms,
        "queue_ms":   queue_ms,
        "itl_ms":     itl_ms,
        "e2e_ms":     e2e_ms,

        # Token length distributions (per vLLM request = per turn)
        "prompt_tok_len": prompt_tok,
        "gen_tok_len":    gen_tok,

        # Cache & KV offload
        "cache": {
            "combined_hit_rate_pct": combined_hit_rate,
            "gpu_hit_rate_pct":      gpu_hit_rate,
            "cpu_hit_rate_pct":      cpu_hit_rate,
            "kv_usage_pct":          round(kv_cache_used * 100, 1) if kv_cache_used is not None else None,
            "cpu_to_gpu_bytes":  round(kv_cpu_to_gpu_bytes)    if kv_cpu_to_gpu_bytes is not None else None,
            "gpu_to_cpu_bytes":  round(kv_gpu_to_cpu_bytes)    if kv_gpu_to_cpu_bytes is not None else None,
            "preemptions":       round(preemptions)             if preemptions is not None else None,
        },

        # Client-side per-task distributions (across tasks, not per-request)
        "task_wall_s":     _pct_dict(wall_s_list),
        "task_n_turns":    _pct_dict(n_turns_list),
        "task_wall_per_turn_s": _pct_dict(tpt_list),

        "tasks": task_results,
    }

    _print_level_summary(level)
    return level


def _fmt_pct(d: dict, suffix: str = "") -> str:
    def _f(v): return f"{v}{suffix}" if v is not None else "-"
    return f"p50={_f(d['p50'])}  p90={_f(d['p90'])}  p95={_f(d['p95'])}  p99={_f(d['p99'])}"


def _print_level_summary(level: dict) -> None:
    c  = level["concurrency"]
    tp = level["throughput"]
    ca = level.get("cache", {})
    W  = 65

    def _f(v, unit=""): return f"{v}{unit}" if v is not None else "-"

    print(f"\n    ┌─ C={c} summary {'─' * (W - 12)}")
    pt = level["per_task"]
    print(f"    │  tasks       : {level['n_ok']}/{level['n_tasks']} ok"
          f"  wall={level['wall_time_s']:.0f}s")
    print(f"    │  throughput  : {tp['tasks_per_sec']:.3f} tasks/s"
          f"  {_f(tp['requests_per_sec'])} req/s"
          f"  {_f(tp['output_tokens_per_sec'])} out-tok/s")
    print(f"    │  totals      : turns={_f(pt.get('total_turns'))}"
          f"  OSL={_f(pt.get('total_osl'))} tok"
          f"  ITL={_f(pt.get('total_itl_s'))} s")

    print(f"    │")
    print(f"    │  ── Latency per vLLM request (= per turn) {'─' * 22}")
    print(f"    │  TTFT    (ms): {_fmt_pct(level['ttft_ms'],    'ms')}")
    print(f"    │  Queue   (ms): {_fmt_pct(level['queue_ms'],   'ms')}  ← scheduler wait")
    print(f"    │  Prefill (ms): {_fmt_pct(level['prefill_ms'], 'ms')}  ← prompt compute")
    print(f"    │  ITL     (ms): {_fmt_pct(level['itl_ms'],     'ms')}  ← decode/tok")
    print(f"    │  E2E     (ms): {_fmt_pct(level['e2e_ms'],     'ms')}")

    print(f"    │")
    print(f"    │  ── Token lengths per request {'─' * 33}")
    print(f"    │  Input  (tok): {_fmt_pct(level['prompt_tok_len'])}")
    print(f"    │  Output (tok): {_fmt_pct(level['gen_tok_len'])}")

    print(f"    │")
    print(f"    │  ── KV cache & CPU↔GPU offload {'─' * 31}")

    def _gb(b): return f"{b/1e9:.2f} GB" if b is not None else "-"
    print(f"    │  Cache hit     : combined={_f(ca.get('combined_hit_rate_pct'), '%')}  GPU={_f(ca.get('gpu_hit_rate_pct'), '%')}  CPU={_f(ca.get('cpu_hit_rate_pct'), '%')}  KV used={_f(ca.get('kv_usage_pct'), '%')}")
    print(f"    │  CPU→GPU      : {_gb(ca.get('cpu_to_gpu_bytes'))}")
    print(f"    │  GPU→CPU      : {_gb(ca.get('gpu_to_cpu_bytes'))}")
    print(f"    │  Preemptions  : {_f(ca.get('preemptions'))}")

    print(f"    │")
    print(f"    │  ── Per-task client-side (across {level['n_ok']} tasks) {'─' * 18}")
    print(f"    │  Wall    (s) : {_fmt_pct(level['task_wall_s'], 's')}")
    print(f"    │  N turns     : {_fmt_pct(level['task_n_turns'])}")
    print(f"    │  Wall/turn(s): {_fmt_pct(level['task_wall_per_turn_s'], 's')}")
    print(f"    └{'─' * W}")


# ── CSV summary ───────────────────────────────────────────────────────────────
def write_csv(levels: list[dict], path: Path) -> None:
    rows = []
    for lv in levels:
        tp = lv["throughput"]
        ca = lv.get("cache", {})
        row = {
            "concurrency":           lv["concurrency"],
            "n_ok":                  lv["n_ok"],
            "wall_time_s":           lv["wall_time_s"],
            "tasks_per_sec":         tp["tasks_per_sec"],
            "requests_per_sec":      tp["requests_per_sec"],
            "output_tokens_per_sec": tp["output_tokens_per_sec"],
            "avg_wall_time_s":       lv["per_task"]["avg_wall_time_s"],
            "avg_n_turns":           lv["per_task"]["avg_n_turns"],
            "avg_output_tokens":     lv["per_task"]["avg_output_tokens"],
            "total_turns":           lv["per_task"]["total_turns"],
            "total_osl":             lv["per_task"]["total_osl"],
            "total_itl_s":           lv["per_task"]["total_itl_s"],
            # TTFT (ms)
            "ttft_p50":    lv["ttft_ms"]["p50"],
            "ttft_p90":    lv["ttft_ms"]["p90"],
            "ttft_p95":    lv["ttft_ms"]["p95"],
            "ttft_p99":    lv["ttft_ms"]["p99"],
            # Queue wait (ms)
            "queue_p50":   lv["queue_ms"]["p50"],
            "queue_p90":   lv["queue_ms"]["p90"],
            "queue_p95":   lv["queue_ms"]["p95"],
            "queue_p99":   lv["queue_ms"]["p99"],
            # Prefill time (ms)
            "prefill_p50": lv["prefill_ms"]["p50"],
            "prefill_p90": lv["prefill_ms"]["p90"],
            "prefill_p95": lv["prefill_ms"]["p95"],
            "prefill_p99": lv["prefill_ms"]["p99"],
            # ITL / decode time per token (ms)
            "itl_p50":     lv["itl_ms"]["p50"],
            "itl_p90":     lv["itl_ms"]["p90"],
            "itl_p95":     lv["itl_ms"]["p95"],
            "itl_p99":     lv["itl_ms"]["p99"],
            # E2E (ms)
            "e2e_p50":     lv["e2e_ms"]["p50"],
            "e2e_p90":     lv["e2e_ms"]["p90"],
            "e2e_p95":     lv["e2e_ms"]["p95"],
            "e2e_p99":     lv["e2e_ms"]["p99"],
            # Token lengths
            "prompt_tok_p50": lv["prompt_tok_len"]["p50"],
            "prompt_tok_p90": lv["prompt_tok_len"]["p90"],
            "prompt_tok_p95": lv["prompt_tok_len"]["p95"],
            "prompt_tok_p99": lv["prompt_tok_len"]["p99"],
            "gen_tok_p50":    lv["gen_tok_len"]["p50"],
            "gen_tok_p90":    lv["gen_tok_len"]["p90"],
            "gen_tok_p95":    lv["gen_tok_len"]["p95"],
            "gen_tok_p99":    lv["gen_tok_len"]["p99"],
            # Cache & offload
            "cache_hit_combined_pct": ca.get("combined_hit_rate_pct"),
            "gpu_cache_hit_pct":      ca.get("gpu_hit_rate_pct"),
            "cpu_cache_hit_pct":      ca.get("cpu_hit_rate_pct"),
            "kv_cache_used_pct":  ca.get("kv_usage_pct"),
            "kv_cpu_to_gpu_bytes": ca.get("cpu_to_gpu_bytes"),
            "kv_gpu_to_cpu_bytes": ca.get("gpu_to_cpu_bytes"),
            "preemptions":         ca.get("preemptions"),
            # Per-task client-side distributions (s)
            "task_wall_p50": lv["task_wall_s"]["p50"],
            "task_wall_p90": lv["task_wall_s"]["p90"],
            "task_wall_p95": lv["task_wall_s"]["p95"],
            "task_wall_p99": lv["task_wall_s"]["p99"],
            "task_turns_p50": lv["task_n_turns"]["p50"],
            "task_turns_p90": lv["task_n_turns"]["p90"],
            "task_turns_p95": lv["task_n_turns"]["p95"],
            "task_turns_p99": lv["task_n_turns"]["p99"],
            "wall_per_turn_p50": lv["task_wall_per_turn_s"]["p50"],
            "wall_per_turn_p90": lv["task_wall_per_turn_s"]["p90"],
            "wall_per_turn_p95": lv["task_wall_per_turn_s"]["p95"],
            "wall_per_turn_p99": lv["task_wall_per_turn_s"]["p99"],
        }
        rows.append(row)

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--concurrency", type=int, default=1,
                    help="Number of concurrent users/tasks (default: 1)")
    ap.add_argument("--tasks-per-level", type=int, default=None,
                    help="Tasks to run at each level (default: C in batch mode, 3×C in sustained)")
    ap.add_argument("--sustained", action="store_true",
                    help="Sustained mode: sustain concurrent N users. When one user completes, immediately start a new user, sustaining N users. Run until 3N users complete")
    ap.add_argument("--start",    type=int, default=0,
                    help="SWE-bench dataset start index")
    ap.add_argument("--end",      type=int, default=None,
                    help="SWE-bench dataset end index (exclusive)")
    ap.add_argument("--prom",     default=DEFAULT_PROM,
                    help=f"Prometheus URL (default: {DEFAULT_PROM})")
    ap.add_argument("--port",     default=None,
                    help="vLLM port (default from .env or 8000)")
    ap.add_argument("--model",    default=None,
                    help="Model name override")
    ap.add_argument("--label",    default="",
                    help="Tag for this run (e.g. tp4, tp2, no-offload)")
    ap.add_argument("--no-clone", action="store_true",
                    help="Skip repo setup; run claude in cwd")
    args = ap.parse_args()

    load_env(ENV_FILE)
    model    = args.model or os.environ.get("MODEL", DEFAULT_MODEL)
    port     = args.port  or os.environ.get("PORT",  DEFAULT_PORT)
    base_url = f"http://localhost:{port}"
    label    = args.label
    prom     = args.prom

    # ── check server is alive ─────────────────────────────────────────────────
    try:
        requests.get(f"{base_url}/health", timeout=5)
    except Exception:
        print(f"ERROR: vLLM server not reachable at {base_url}/health", file=sys.stderr)
        sys.exit(1)

    # ── check Prometheus is alive ─────────────────────────────────────────────
    try:
        requests.get(f"{prom}/api/v1/query", params={"query": "1"}, timeout=5)
    except Exception:
        print(f"WARNING: Prometheus not reachable at {prom} — latency percentiles will be None",
              file=sys.stderr)

    ds = load_dataset()
    total = len(ds)
    end   = args.end if args.end is not None else total

    c         = args.concurrency
    sustained = args.sustained
    # Sustained: keep C tasks running in parallel; stop after 5 minutes OR when all
    # queued tasks finish, whichever comes first. Pre-load enough tasks to stay busy.
    n_needed  = args.tasks_per_level or ((5 * c) if sustained else c)
    SUSTAINED_MIN_S  = 300   # 5-minute wall-clock cap
    SUSTAINED_MAX_N  = 5 * c # unused in stop logic; controls queue depth

    candidate_indices = list(range(args.start, min(end, total)))
    if len(candidate_indices) < n_needed:
        print(f"WARNING: need {n_needed} tasks but only {len(candidate_indices)} available.",
              file=sys.stderr)

    # ── workspaces ────────────────────────────────────────────────────────────
    instances    = [ds[i] for i in candidate_indices[:n_needed]]
    first_batch  = instances[:c]         # needed before inference can start
    rest_batch   = instances[c:]         # cloned in background while inference runs

    extra_queue: _queue.Queue = _queue.Queue()

    if args.no_clone:
        initial_specs = [{"instance": inst, "workdir": Path.cwd()} for inst in first_batch]
        for inst in rest_batch:
            extra_queue.put({"instance": inst, "workdir": Path.cwd()})
        extra_queue.put(None)  # sentinel
    else:
        # Clone first C workspaces in parallel (blocking — needed to start inference)
        print(f"\nSetting up first {len(first_batch)} workspaces ...")
        initial_specs: list[dict] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(CLONE_WORKERS, len(first_batch))) as cex:
            future_map = {cex.submit(setup_workspace, inst): inst for inst in first_batch}
            for f in concurrent.futures.as_completed(future_map):
                inst = future_map[f]
                try:
                    wd = f.result()
                    print(f"  ✓ {inst['instance_id']}")
                except Exception as e:
                    print(f"  ✗ {inst['instance_id']}: {e}")
                    wd = Path.cwd()
                initial_specs.append({"instance": inst, "workdir": wd})

        # Clone remaining workspaces in background, feeding into extra_queue
        if rest_batch:
            print(f"  Cloning {len(rest_batch)} remaining workspaces in background ...", flush=True)

            def _bg_clone():
                with concurrent.futures.ThreadPoolExecutor(max_workers=CLONE_WORKERS) as cex:
                    fmap = {cex.submit(setup_workspace, inst): inst for inst in rest_batch}
                    for f in concurrent.futures.as_completed(fmap):
                        inst = fmap[f]
                        try:
                            wd = f.result()
                            print(f"  ✓ [bg] {inst['instance_id']}", flush=True)
                        except Exception as e:
                            print(f"  ✗ [bg] {inst['instance_id']}: {e}", flush=True)
                            wd = Path.cwd()
                        extra_queue.put({"instance": inst, "workdir": wd})
                extra_queue.put(None)  # sentinel

            threading.Thread(target=_bg_clone, daemon=True).start()
        else:
            extra_queue.put(None)  # no rest — sentinel immediately

    # ── output directory ──────────────────────────────────────────────────────
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"bench_{label}_{ts}" if label else f"bench_{ts}"
    out_dir  = RESULTS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═'*70}")
    print(f"  Benchmark : {run_name}")
    print(f"  Model     : {model}")
    print(f"  Server    : {base_url}")
    print(f"  Prometheus: {prom}")
    print(f"  Concurrency: {c}  ({'sustained: replace on finish, stop after 2N+1=' + str(2*c+1) if sustained else 'batch: all start together, stop when all done'})")
    print(f"  Tasks     : {n_needed}")
    print(f"  Results   : {out_dir}/")
    print(f"{'═'*70}")

    started_at = datetime.now().isoformat()
    result     = run_level(c, initial_specs, extra_queue, n_needed, model, base_url, prom,
                           sustained_min_s=SUSTAINED_MIN_S if sustained else 0)
    result["label"] = label

    out_path = out_dir / f"c{c:03d}.json"
    out_path.write_text(json.dumps(result, indent=2))

    summary = {
        "run_name":    run_name,
        "label":       label,
        "model":       model,
        "server":      base_url,
        "prometheus":  prom,
        "concurrency": c,
        "sustained":   sustained,
        "started_at":  started_at,
        "finished_at": datetime.now().isoformat(),
        **result,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_csv([result], out_dir / "summary.csv")

    # ── print final table ─────────────────────────────────────────────────────
    tp = result["throughput"]
    ca = result.get("cache", {})
    def _f(v, u=""): return f"{v}{u}" if v is not None else "-"
    print(f"\n{'═'*70}")
    pt = result["per_task"]
    print(f"  C={c}  tasks/s={tp['tasks_per_sec']:.4f}"
          f"  req/s={_f(tp['requests_per_sec'])}"
          f"  out-tok/s={_f(tp['output_tokens_per_sec'])}")
    print(f"  Totals: turns={_f(pt.get('total_turns'))}"
          f"  OSL={_f(pt.get('total_osl'))} tok"
          f"  ITL={_f(pt.get('total_itl_s'))} s")
    print(f"  TTFT  p50={_f(result['ttft_ms']['p50'],'ms')}"
          f"  p90={_f(result['ttft_ms']['p90'],'ms')}"
          f"  p95={_f(result['ttft_ms']['p95'],'ms')}"
          f"  p99={_f(result['ttft_ms']['p99'],'ms')}")
    print(f"  Queue   p50={_f(result['queue_ms']['p50'],'ms')}"
          f"  p90={_f(result['queue_ms']['p90'],'ms')}"
          f"  p95={_f(result['queue_ms']['p95'],'ms')}"
          f"  p99={_f(result['queue_ms']['p99'],'ms')}")
    print(f"  Prefill p50={_f(result['prefill_ms']['p50'],'ms')}"
          f"  p90={_f(result['prefill_ms']['p90'],'ms')}"
          f"  p95={_f(result['prefill_ms']['p95'],'ms')}"
          f"  p99={_f(result['prefill_ms']['p99'],'ms')}")
    print(f"  ITL     p50={_f(result['itl_ms']['p50'],'ms')}"
          f"  p90={_f(result['itl_ms']['p90'],'ms')}"
          f"  p95={_f(result['itl_ms']['p95'],'ms')}"
          f"  p99={_f(result['itl_ms']['p99'],'ms')}")
    print(f"  E2E     p50={_f(result['e2e_ms']['p50'],'ms')}"
          f"  p90={_f(result['e2e_ms']['p90'],'ms')}"
          f"  p95={_f(result['e2e_ms']['p95'],'ms')}"
          f"  p99={_f(result['e2e_ms']['p99'],'ms')}")
    print(f"  Input tok  p50={_f(result['prompt_tok_len']['p50'])}"
          f"  p90={_f(result['prompt_tok_len']['p90'])}"
          f"  p95={_f(result['prompt_tok_len']['p95'])}"
          f"  p99={_f(result['prompt_tok_len']['p99'])}")
    print(f"  Output tok p50={_f(result['gen_tok_len']['p50'])}"
          f"  p90={_f(result['gen_tok_len']['p90'])}"
          f"  p95={_f(result['gen_tok_len']['p95'])}"
          f"  p99={_f(result['gen_tok_len']['p99'])}")
    def _gb(b): return f"{b/1e9:.2f} GB" if b is not None else "-"
    print(f"  Cache hit combined={_f(ca.get('combined_hit_rate_pct'),'%')}"
          f"  GPU={_f(ca.get('gpu_hit_rate_pct'),'%')}"
          f"  CPU={_f(ca.get('cpu_hit_rate_pct'),'%')}"
          f"  KV used={_f(ca.get('kv_usage_pct'),'%')}"
          f"  preemptions={_f(ca.get('preemptions'))}")
    print(f"  CPU→GPU={_gb(ca.get('cpu_to_gpu_bytes'))}"
          f"  GPU→CPU={_gb(ca.get('gpu_to_cpu_bytes'))}")
    print(f"  Task wall  p50={_f(result['task_wall_s']['p50'],'s')}"
          f"  p90={_f(result['task_wall_s']['p90'],'s')}"
          f"  p95={_f(result['task_wall_s']['p95'],'s')}"
          f"  p99={_f(result['task_wall_s']['p99'],'s')}")
    print(f"  N turns    p50={_f(result['task_n_turns']['p50'])}"
          f"  p90={_f(result['task_n_turns']['p90'])}"
          f"  p95={_f(result['task_n_turns']['p95'])}"
          f"  p99={_f(result['task_n_turns']['p99'])}")
    print(f"  Wall/turn  p50={_f(result['task_wall_per_turn_s']['p50'],'s')}"
          f"  p90={_f(result['task_wall_per_turn_s']['p90'],'s')}"
          f"  p95={_f(result['task_wall_per_turn_s']['p95'],'s')}"
          f"  p99={_f(result['task_wall_per_turn_s']['p99'],'s')}")
    print(f"  → {out_dir}/")
    print(f"{'═'*70}")


if __name__ == "__main__":
    main()
