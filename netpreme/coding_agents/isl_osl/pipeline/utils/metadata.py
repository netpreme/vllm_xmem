"""Run metadata writers.

Two JSON snapshots of a run:

- ``write_run_config`` — once per run, ``run_config.json``: the full CLI args,
  the running server's resolved serving knobs (arg → os env → .env, incl. the
  actually-served model), the raw .env, the dataset selection, ports, versions.
- ``write_meta`` — once per problem, ``telemetry/<iid>/meta.json``: the
  problem's repo/commit, timing and exit code. Doubles as the resume ledger
  (main.py reads ``exit_code == 0`` to skip solved problems).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from loguru import logger

from pipeline.claude import claude_version
from pipeline.utils.jsonl import instance_dir
from pipeline.vllm_server import Server, _read_env_file, gpu_info, vllm_version


def write_run_config(
    *,
    save_dir: Path,
    args: argparse.Namespace,
    server: Server,
    dataset: list,
    dataset_name: str,
    solved_ids: set,
    proxy_port: int,
    started_at: float,
) -> None:
    """Snapshot every input/knob for this run to ``run_config.json``."""
    config = {
        "stamp": save_dir.name,
        "started_at": round(started_at, 3),
        "command": " ".join(sys.argv),
        "args": vars(args),
        "serving_config": server.serving_config(),
        "dotenv": _read_env_file(),
        "dataset": {
            "name": dataset_name,
            "random": args.random,
            "seed": args.seed,
            "pending": len(dataset),
            "skipped_solved": len(solved_ids),
            "instance_ids": [task["instance_id"] for task in dataset],
        },
        "ports": {"vllm": server.port, "proxy": proxy_port, "vllm_url": server.url},
        "versions": {
            "claude": claude_version(),
            "vllm": vllm_version(),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "gpu": gpu_info(),
    }
    # default=str so Path args (e.g. --resume) serialize cleanly.
    (save_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, default=str) + "\n"
    )
    logger.info("wrote run config → {}", save_dir / "run_config.json")


def write_meta(
    save_dir: Path, task: dict, started_at: float, ended_at: float, exit_code: int
) -> None:
    """Record one problem's run metadata under telemetry/<iid>/meta.json."""
    iid = task["instance_id"]
    problem_dir = instance_dir(save_dir / "telemetry", iid)
    problem_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "instance_id": iid,
        "difficulty": task.get("difficulty"),
        "repo": task.get("repo"),
        "base_commit": task.get("base_commit"),
        "started_at": round(started_at, 3),
        "ended_at": round(ended_at, 3),
        "exit_code": exit_code,
    }
    (problem_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
