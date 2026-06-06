"""Coding-agent benchmark runner — entry point.

Orchestrates the per-problem loop; the actual work lives elsewhere. Each
problem is wrapped in four context managers, each driving one thing:

    Server         fresh vLLM for this problem (empty cache; killed on exit)
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
        with Server(...) as server, MetricsScraper(...), \\
             Proxy(...) as proxy, Sandbox(...) as sandbox:
            exit_code = coding_agent(task, sandbox.dir, server.model, proxy.base_url)
        write_meta(...)
    report.run(save_dir)  # once, at the end
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis.report import run as run_report
from pipeline import claude
from pipeline.agent import coding_agent
from pipeline.datasets import DATASETS, Sandbox, get_dataset
from pipeline.proxy import Proxy
from pipeline.utils.metadata import write_meta, write_run_config
from pipeline.vllm_server import Server, vllm_version
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
        "anthropic = send to api.anthropic.com (no vLLM), deriving "
        "isl/osl/isl_new from Anthropic's usage. Needs ANTHROPIC_API_KEY set.",
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        default="verified",
        help="benchmark dataset to run (default: %(default)s)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--capture",
        nargs="?",
        const="raw",
        default=None,
        choices=["raw"],
        help="run the proxy and capture per-turn data: agentic metadata "
        "(system_prompt_chars, tool calls, stop reason) → proxy.jsonl, AND the "
        "raw text + exact token-id traces (isl/isl_new/osl) → raw.jsonl. "
        "Omit the flag to skip the proxy entirely. `--capture` and "
        "`--capture raw` are equivalent.",
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
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
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

    stamp = (
        args.resume.name if args.resume else datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    save_dir = HERE / "results" / stamp
    save_dir.mkdir(parents=True, exist_ok=True)

    # Resume skips problems already solved here (meta.json exit_code == 0).
    solved_ids = set()
    for meta_path in (save_dir / "telemetry").glob("*/meta.json"):
        meta = json.loads(meta_path.read_text())
        if meta.get("exit_code") == 0:
            solved_ids.add(meta["instance_id"])

    dataset_name = DATASETS[args.dataset]
    dataset = get_dataset(name=dataset_name, solved_ids=solved_ids)
    if args.limit is not None:
        dataset = dataset[: args.limit]
    logger.info("{} problems pending → {}", len(dataset), save_dir)

    # anthropic backend: claude-cli talks DIRECTLY to Anthropic with its own
    # subscription OAuth (no proxy, no vLLM); claude.solve parses per-turn
    # usage from its stream-json stdout into vllm.jsonl.
    remote = args.backend == "anthropic"
    model = args.model  # resolved to server.model per-problem on the vllm backend
    sandbox_root = Path(f"/tmp/swe_sandboxes/{save_dir.name}")
    started_at = time.time()
    for task in tqdm(dataset, desc="solving", unit="problem"):
        instance_id = task["instance_id"]
        logger.info("{} ({} @ {})", instance_id, task["repo"], task["base_commit"][:5])
        server = None
        with contextlib.ExitStack() as stack:
            if remote:
                model = args.model
                base_url = ANTHROPIC_URL  # unused under oauth (claude-cli's own creds)
            else:
                server = stack.enter_context(
                    Server(
                        url=SERVER_URL,
                        model=args.model,
                        tensor_parallel_size=args.tensor_parallel_size,
                        max_model_len=args.max_model_len,
                        gpu_memory_utilization=args.gpu_memory_utilization,
                        tool_call_parser=args.tool_call_parser,
                    )
                )
                stack.enter_context(
                    MetricsScraper(
                        url=SERVER_URL, save_dir=save_dir, instance_id=instance_id
                    )
                )
                proxy = stack.enter_context(
                    Proxy(
                        save_dir=save_dir,
                        instance_id=instance_id,
                        url=SERVER_URL,
                        proxy_port=PROXY_PORT,
                        capture=args.capture is not None,
                        raw=args.capture is not None,
                    )
                )
                model = server.model
                base_url = proxy.base_url
            sandbox = stack.enter_context(
                Sandbox(root=sandbox_root, prefix=f"{instance_id}.")
            )
            time.sleep(WAIT_TIME)  # let the proxy/watcher settle on this instance_id
            exit_code = coding_agent(
                task=task,
                sandbox_dir=sandbox.dir,
                model=model,
                base_url=base_url,
                timeout_s=args.agent_timeout_s,
                oauth=remote,
                telemetry_dir=(save_dir / "telemetry") if remote else None,
                raw=remote and args.capture is not None,
            )
        write_meta(
            save_dir=save_dir,
            task=task,
            started_at=sandbox.started,
            ended_at=sandbox.ended,
            exit_code=exit_code,
        )
        # Snapshot run inputs once (vLLM backend only — needs the server config).
        if server is not None and not (save_dir / "run_config.json").exists():
            write_run_config(
                save_dir=save_dir,
                args=args,
                server=server,
                dataset=dataset,
                dataset_name=dataset_name,
                solved_ids=solved_ids,
                proxy_port=PROXY_PORT,
                started_at=started_at,
            )

    (save_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "model": model,
                "vllm_url": SERVER_URL,
                "dataset": dataset_name,
                "capture": args.capture,
                "versions": {
                    "claude": claude.claude_version(),
                    "vllm": vllm_version(),
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                },
                "started_at": round(started_at, 3),
                "ended_at": round(time.time(), 3),
            },
            indent=2,
        )
        + "\n"
    )

    run_report(save_dir=save_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
