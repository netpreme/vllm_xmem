#!/usr/bin/env python3
"""Coding-agent benchmark runner — entry point.

Orchestrates the per-problem loop; the actual work lives elsewhere. Each
problem is wrapped in four context managers, each driving one thing:

    VLLMServer     fresh vLLM for this problem (empty cache; killed on exit)
    MetricsScraper polls vLLM's Prometheus /metrics once per turn
                   → results/<stamp>/telemetry/<iid>/vllm.jsonl
    Proxy          sits between claude-cli and vLLM (with --capture); normalizes
                   requests and tees per-turn rows
                   → results/<stamp>/telemetry/<iid>/proxy.jsonl
    Sandbox        throwaway repo checkout + wall-clock timer for this problem

The agent only solves; analysis is a single separate pass at the end
(analysis/report.py builds data.npz + figures). Per-problem metadata
(results/<stamp>/telemetry/<iid>/meta.json) doubles as the resume ledger.

    for task in dataset:
        with VLLMServer(...) as server, MetricsScraper(...), \\
             Proxy(...) as proxy, Sandbox(...) as sandbox:
            exit_code = coding_agent(task, sandbox.dir, server.model, proxy.base_url)
        write_meta(...)
    report.run(save_dir)                                 # once, at the end
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from loguru import logger
from tqdm import tqdm

# Make `from pipeline import ...` / `from analysis...` work when this script
# is invoked directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis.report import run as run_report
from pipeline import datasets
from pipeline.agent import Sandbox, coding_agent, write_meta
from pipeline.proxy import Proxy
from pipeline.server import VLLMServer
from pipeline.vllm_metrics import MetricsScraper

HERE = Path(__file__).resolve().parent

# Ports are env-overridable so multiple TP1 shards can run side by side
# on the same host (one shard per GPU, distinct port pairs).
VLLM_PORT = int(os.environ.get("VLLM_PORT", "8000"))
PROXY_PORT = int(os.environ.get("PROXY_PORT", "8001"))
VLLM_URL = f"http://localhost:{VLLM_PORT}"
WAIT_TIME = 0.3
DATASET = "princeton-nlp/SWE-bench_Verified"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--capture",
        action="store_true",
        help="run the proxy to capture per-turn "
        "agentic metadata (system_prompt_chars, tool calls, stop reason)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="reuse <SAVE_DIR>, skipping problems already solved in it",
    )
    # vLLM launch knobs — each falls back to server.sh's .env/default if unset.
    parser.add_argument(
        "--model", default=None, help="model to serve (default: server.sh's .env)"
    )
    parser.add_argument(
        "--tensor-parallel", dest="tensor_parallel_size", type=int, default=None
    )
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--tool-call", dest="tool_call_parser", default=None)
    args = parser.parse_args()

    stamp = (
        args.resume.name if args.resume else datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    save_dir = HERE / "results" / stamp
    save_dir.mkdir(parents=True, exist_ok=True)

    # Resume = skip problems already solved here, inferred from their
    # meta.json (exit_code == 0). No separate ledger file needed.
    solved_ids = set()
    for meta_path in (save_dir / "telemetry").glob("*/meta.json"):
        meta = json.loads(meta_path.read_text())
        if meta.get("exit_code") == 0:
            solved_ids.add(meta["instance_id"])

    dataset = datasets.get_dataset(
        name=DATASET,
        random=args.random,
        seed=args.seed,
        solved_ids=solved_ids,
    )
    logger.info("{} problems pending → {}", len(dataset), save_dir)

    sandbox_root = Path(f"/tmp/swe_sandboxes/{save_dir.name}")
    started_at = time.time()
    for task in tqdm(dataset, desc="solving", unit="problem"):
        instance_id = task["instance_id"]
        logger.info("{} ({} @ {})", instance_id, task["repo"], task["base_commit"][:5])
        # Each component drives one thing: server, watcher, proxy, sandbox.
        with (
            VLLMServer(
                url=VLLM_URL,
                model=args.model,
                tensor_parallel_size=args.tensor_parallel_size,
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
                tool_call_parser=args.tool_call_parser,
            ) as server,
            MetricsScraper(url=VLLM_URL, save_dir=save_dir, instance_id=instance_id),
            Proxy(
                save_dir=save_dir,
                instance_id=instance_id,
                url=VLLM_URL,
                proxy_port=PROXY_PORT,
                capture=args.capture,
            ) as proxy,
            Sandbox(root=sandbox_root, prefix=f"{instance_id}.") as sandbox,
        ):
            model = server.model
            time.sleep(WAIT_TIME)  # let the proxy/watcher settle on this instance_id
            exit_code = coding_agent(
                task=task,
                sandbox_dir=sandbox.dir,
                model=server.model,
                base_url=proxy.base_url,
            )
        write_meta(
            save_dir=save_dir,
            task=task,
            started_at=sandbox.started,
            ended_at=sandbox.ended,
            exit_code=exit_code,
        )

    (save_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "model": model,
                "vllm_url": VLLM_URL,
                "dataset": DATASET,
                "capture": args.capture,
                "started_at": round(started_at, 3),
                "ended_at": round(time.time(), 3),
            },
            indent=2,
        )
        + "\n"
    )

    run_report(save_dir=save_dir)
    return 0


def _claude_version() -> str:
    """claude-cli version string, e.g. '2.1.156 (Claude Code)'."""
    try:
        return subprocess.run(
            ["claude", "--version"], capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return ""


def _vllm_version() -> str:
    """Installed vLLM distribution version (cheap — no package import)."""
    try:
        return version("vllm")
    except PackageNotFoundError:
        return ""


def _gpu_info() -> dict:
    """GPU name / count / total-memory (MiB) via nvidia-smi; {} if unavailable."""
    try:
        lines = (
            subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            .stdout.strip()
            .splitlines()
        )
    except (subprocess.SubprocessError, OSError):
        return {}
    names = [ln.split(",")[0].strip() for ln in lines if ln.strip()]
    if not names:
        return {}
    try:
        memory_mib = int(lines[0].split(",")[1])
    except (ValueError, IndexError):
        memory_mib = 0
    return {"name": names[0], "count": len(names), "memory_mib": memory_mib}


if __name__ == "__main__":
    sys.exit(main())
