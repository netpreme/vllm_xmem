"""Coding-agent benchmark runner — entry point.

Orchestrates the per-problem loop; the actual work lives elsewhere. Each
problem is wrapped in four context managers, each driving one thing:

    Server         fresh vLLM for this problem (empty cache; killed on exit)
    MetricsScraper polls vLLM's Prometheus /metrics once per turn
                   → results/<stamp>/telemetry/<iid>/vllm_metrics.jsonl
    Proxy          sits between claude-cli and vLLM (with --capture); normalizes
                   requests and tees the per-turn raw text trace
                   → results/<stamp>/telemetry/<iid>/vllm_traces.jsonl
    Sandbox        throwaway repo checkout + wall-clock timer for this problem

The agent only solves; analysis is a separate pass run on demand (it derives
Anthropic transcript telemetry when needed, then builds its arrays and figures).
Per-problem metadata (results/<stamp>/telemetry/<iid>/session_config.json)
doubles as the resume ledger.

    for task in dataset:
        with Server(...) as server, MetricsScraper(...), \\
             Proxy(...) as proxy, Sandbox(...) as sandbox:
            exit_code = coding_agent(task, sandbox.dir, server.model, proxy.base_url)
        save_session_metadata(...)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import claude
from pipeline.agent import coding_agent
from pipeline.datasets import DATASETS, Sandbox, get_dataset
from pipeline.proxy import Proxy
from pipeline.utils.metadata import save_session_metadata, write_config
from pipeline.vllm_server import Server
from pipeline.vllm_metrics import MetricsScraper

HERE = Path(__file__).resolve().parent

SERVER_PORT = int(os.environ.get("SERVER_PORT", "8000"))
PROXY_PORT = int(os.environ.get("PROXY_PORT", "8001"))
SERVER_URL = f"http://localhost:{SERVER_PORT}"
ANTHROPIC_URL = "https://api.anthropic.com"
WAIT_TIME = 0.3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=["vllm", "anthropic"],
        default="vllm",
        help="vllm = serve --model locally and capture /metrics (default); "
        "anthropic = send to api.anthropic.com (no vLLM) and copy Claude "
        "transcripts for analysis. Requires --model.",
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        default="pro",
        help="benchmark dataset to run (default: %(default)s)",
    )
    parser.add_argument(
        "--capture",
        nargs="?",
        const="raw",
        default=None,
        choices=["raw"],
        help="run the proxy and tee the per-turn raw text trace "
        "(isl/isl_new/osl as text) → vllm_traces.jsonl. Omit the flag to skip "
        "the proxy entirely. `--capture` and `--capture raw` are equivalent.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="run at most N (pending) problems this invocation (e.g. --limit 1)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="reuse <SAVE_DIR>, skipping problems already solved in it",
    )
    parser.add_argument(
        "--model", default=None, help="model to serve (default: server.sh's .env)"
    )
    parser.add_argument(
        "--tensor-parallel", dest="tensor_parallel_size", type=int, default=None
    )
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--tool-call", dest="tool_call_parser", default=None)
    parser.add_argument(
        "--agent-timeout",
        dest="agent_timeout_s",
        type=float,
        default=claude.DEFAULT_TIMEOUT_S,
        help="wall-clock cap (seconds) per claude session; on timeout the "
        "process tree is killed and the problem recorded as unsolved "
        f"(default: {claude.DEFAULT_TIMEOUT_S})",
    )
    args = parser.parse_args()

    remote = args.backend == "anthropic"
    if remote and args.model is None:
        parser.error("--model is required with --backend anthropic")
    if not remote and args.tool_call_parser is None:
        parser.error(
            "--tool-call is required with --backend vllm (must match the model "
            "family, e.g. qwen3_coder for Qwen, openai for GPT-OSS)"
        )

    if args.resume is not None:
        save_dir = args.resume.expanduser().resolve()
    else:
        save_dir = HERE / "results" / datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Resume skips problems already solved here (session_config exit_code == 0).
    solved_ids = set()
    for session_path in (save_dir / "telemetry").glob("*/session_config.json"):
        session = json.loads(session_path.read_text())
        if session.get("exit_code") == 0:
            solved_ids.add(session["instance_id"])

    dataset_name = DATASETS[args.dataset]
    dataset = get_dataset(name=dataset_name, solved_ids=solved_ids)
    if args.limit is not None:
        dataset = dataset[: args.limit]
    logger.info("{} problems pending → {}", len(dataset), save_dir)

    backend_url = ANTHROPIC_URL if remote else SERVER_URL
    capture = not remote and args.capture is not None
    sandbox_root = Path(f"/tmp/swe_sandboxes/{save_dir.name}")
    started_at = time.time()

    server_kwargs = dict(
        url=backend_url,
        remote=remote,
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tool_call_parser=args.tool_call_parser,
    )

    # Snapshot the overall config once, up front (serving knobs resolve from
    # args/env/.env without a running server).
    write_config(
        save_dir=save_dir,
        args=args,
        server=Server(**server_kwargs),
        dataset=dataset,
        dataset_name=dataset_name,
        solved_ids=solved_ids,
        proxy_port=PROXY_PORT,
        started_at=started_at,
    )

    for task in tqdm(dataset, desc="solving", unit="problem"):
        instance_id = task["instance_id"]
        logger.info("{} ({} @ {})", instance_id, task["repo"], task["base_commit"][:5])
        with (
            Server(**server_kwargs) as server,
            MetricsScraper(
                url=backend_url,
                save_dir=save_dir,
                instance_id=instance_id,
                enabled=not remote,
            ),
            Proxy(
                save_dir=save_dir,
                instance_id=instance_id,
                url=server.url,
                proxy_port=PROXY_PORT,
                capture=capture,
                raw=capture,
            ) as proxy,
            Sandbox(root=sandbox_root, prefix=f"{instance_id}.") as sandbox,
        ):
            time.sleep(WAIT_TIME)  # let the proxy/watcher settle on this instance_id
            exit_code = coding_agent(
                task=task,
                sandbox_dir=sandbox.dir,
                model=server.model,
                base_url=proxy.base_url,  # anthropic url, ignored under oauth
                timeout_s=args.agent_timeout_s,
                oauth=remote,
                telemetry_dir=(save_dir / "telemetry") if remote else None,
            )
        save_session_metadata(
            save_dir=save_dir,
            task=task,
            server=server,
            started_at=sandbox.started,
            ended_at=sandbox.ended,
            exit_code=exit_code,
        )

    # Analysis is a separate pass run on demand.
    return 0


if __name__ == "__main__":
    sys.exit(main())
