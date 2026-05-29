#!/usr/bin/env python3
"""Coding-agent benchmark runner.

The agent is given a task + a server URL and *solves* — nothing else.
The runner wraps each solve in two context managers, in order: first
``initialize_server`` cold-restarts vLLM (empty cache), then ``Proxy``
attaches the telemetry/proxy on top of it. Each problem's raw per-turn
telemetry is saved; nothing is analysed inline. Analysis is a single
separate pass at the end over the saved directory (analyze.sh).

    agent = CodingAgent()
    for task in dataset:
        with initialize_server(VLLM_URL) as model:      # 1. reset vllm
            with Proxy(save_dir, iid, ...) as proxy:     # 2. attach proxy
                agent.solve(task=task, save_dir=save_dir,
                            model=model, base_url=proxy.base_url)
    analyze.sh <save_dir>                                # once, at the end
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

from loguru import logger
from tqdm import tqdm

# Make `from pipeline import ...` work when this script is invoked directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import claude, datasets, git_repo
from pipeline.proxy import Proxy
from pipeline.server import initialize_server

HERE = Path(__file__).resolve().parent
ANALYZE_SH = HERE / "analyze.sh"

# Ports are env-overridable so multiple TP1 shards can run side by side
# on the same host (one shard per GPU, distinct port pairs).
VLLM_PORT = int(os.environ.get("VLLM_PORT", "8000"))
PROXY_PORT = int(os.environ.get("PROXY_PORT", "8001"))
VLLM_URL = f"http://localhost:{VLLM_PORT}"
QUIESCE_S = 0.3
DATASET = "princeton-nlp/SWE-bench_Verified"


# ---------------------------------------------------------------------------
# Coding agent. Given a task + a served model + a base URL, it solves —
# it does not restart vLLM, attach proxies, or analyse anything.
# ---------------------------------------------------------------------------


class CodingAgent:
    """Solves SWE-bench problems one at a time — nothing else."""

    def solve(self, task: dict, save_dir: Path, model: str, base_url: str) -> int:
        """Clone the target repo and drive claude-cli over `task`.

        Writes per_problem/<iid>.meta.json, cleans up the workspace, and
        returns the exit code."""
        iid = task["instance_id"]
        per_problem_dir = save_dir / "per_problem"
        per_problem_dir.mkdir(parents=True, exist_ok=True)
        workdirs = Path(f"/tmp/swe_workdirs/{save_dir.name}")
        workdirs.mkdir(parents=True, exist_ok=True)
        workdir = Path(tempfile.mkdtemp(prefix=f"{iid}.", dir=workdirs))

        time.sleep(QUIESCE_S)
        t_start = time.time()
        try:
            repo = git_repo.clone(task, workdir)
            exit_code = claude.solve(task, repo, model=model, base_url=base_url)
        except subprocess.SubprocessError as exc:
            logger.error("{}: clone/solve failed: {!r}", iid, exc)
            exit_code = 1
        t_end = time.time()

        meta = {
            "instance_id": iid,
            "difficulty": task.get("difficulty"),
            "repo": task.get("repo"),
            "base_commit": task.get("base_commit"),
            "started_at": round(t_start, 3),
            "ended_at": round(t_end, 3),
            "exit_code": exit_code,
        }
        (per_problem_dir / f"{iid}.meta.json").write_text(
            json.dumps(meta, indent=2) + "\n"
        )

        shutil.rmtree(workdir, ignore_errors=True)
        return exit_code


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--random", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--capture",
        action="store_true",
        help="run the proxy to capture per-turn "
        "agentic metadata (system_prompt_chars, tool calls, stop reason)",
    )
    p.add_argument(
        "--resume",
        type=Path,
        help="reuse <SAVE_DIR>, skipping problems already solved in it",
    )
    args = p.parse_args()

    stamp = (
        args.resume.name if args.resume else datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    save_dir = HERE / "save" / stamp
    save_dir.mkdir(parents=True, exist_ok=True)

    # Resume = skip problems already solved here, inferred from their
    # meta.json (exit_code == 0). No separate ledger file needed.
    solved_ids = set()
    for meta_path in (save_dir / "per_problem").glob("*.meta.json"):
        meta = json.loads(meta_path.read_text())
        if meta.get("exit_code") == 0:
            solved_ids.add(meta["instance_id"])

    dataset = datasets.get_dataset(
        DATASET,
        random=args.random,
        seed=args.seed,
        solved_ids=solved_ids,
    )
    logger.info("{} problems pending → {}", len(dataset), save_dir)

    agent = CodingAgent()
    model = None
    started_at = time.time()
    for task in tqdm(dataset, desc="solving", unit="problem"):
        iid = task["instance_id"]
        logger.info("{} ({} @ {})", iid, task["repo"], task["base_commit"][:8])
        # 1. reset vLLM, 2. attach the proxy on top of the fresh server.
        with initialize_server(VLLM_URL) as served, Proxy(
            save_dir,
            iid,
            vllm_url=VLLM_URL,
            proxy_port=PROXY_PORT,
            capture=args.capture,
        ) as proxy:
            model = served
            agent.solve(
                task=task, save_dir=save_dir, model=served, base_url=proxy.base_url
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

    subprocess.run(["bash", str(ANALYZE_SH), str(save_dir)], check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
