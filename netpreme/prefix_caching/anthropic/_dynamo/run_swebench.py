#!/usr/bin/env python3
"""
SWE-bench Verified runner using Claude Code as the agent.

Loads tasks from princeton-nlp/SWE-bench_Verified, checks out each repo,
runs Claude Code (claude -p) to generate a fix, and records the patch.
TTFT per LLM call is captured by the ttft_proxy running on :8001.

Prerequisites:
    pip install swebench datasets
    python3 ttft_proxy.py &          # start proxy first
    ./launch_dynamo.sh               # Dynamo + vLLM must be running

Usage:
    python3 run_swebench.py                         # 10 tasks, proxy on :8001
    python3 run_swebench.py --n 20                  # 20 tasks
    python3 run_swebench.py --filter django          # only django tasks
    python3 run_swebench.py --tag cpu               # label run in logs (e.g. cpu / chip)
    python3 run_swebench.py --no-proxy               # skip proxy (no TTFT capture)

Results written to:
    results/{tag}/                   # one dir per run
        tasks.jsonl                  # per-task outcome (resolved, patch size, timing)
        ttft_log.jsonl               # symlink → proxy's log (or copy if --no-proxy)
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from datetime import datetime

# ── constants ─────────────────────────────────────────────────────────

PROXY_URL   = "http://localhost:8001"
DYNAMO_URL  = "http://localhost:8000"
MODEL_ENV   = "Qwen/Qwen2.5-Coder-32B-Instruct"

# Repos with large codebases — better KV cache pressure
PREFERRED_REPOS = {
    'django/django',
    'astropy/astropy',
    'scikit-learn/scikit-learn',
    'sympy/sympy',
    'matplotlib/matplotlib',
    'sphinx-doc/sphinx',
    'pytest-dev/pytest',
    'pallets/flask',
    'psf/requests',
    'pydata/xarray',
}

CLAUDE_TASK_PROMPT = """\
You are an expert software engineer. You have been given a GitHub issue to fix.

The repository has already been checked out in your current working directory.

ISSUE:
{problem_statement}

INSTRUCTIONS:
1. Read the relevant source files to understand the codebase.
2. Identify the root cause of the issue.
3. Make the minimal necessary code changes to fix the issue.
4. Do not add tests unless the issue specifically requires it.
5. Do not modify unrelated files.

Fix the issue now.
"""

# ── helpers ───────────────────────────────────────────────────────────

def load_tasks(n: int, filter_repo: str | None = None) -> list[dict]:
    """Load n tasks from SWE-bench Verified (HuggingFace)."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing swebench and datasets...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                               'swebench', 'datasets', '-q'])
        from datasets import load_dataset

    print("Loading SWE-bench Verified dataset...")
    ds = load_dataset('princeton-nlp/SWE-bench_Verified', split='test')
    tasks = list(ds)

    # Prefer repos with large codebases for better KV cache stress
    preferred = [t for t in tasks
                 if t.get('repo', '') in PREFERRED_REPOS]
    others    = [t for t in tasks
                 if t.get('repo', '') not in PREFERRED_REPOS]
    ordered   = preferred + others

    if filter_repo:
        ordered = [t for t in ordered
                   if filter_repo.lower() in t.get('repo', '').lower()]

    selected = ordered[:n]
    print(f"Selected {len(selected)} tasks "
          f"({sum(1 for t in selected if t.get('repo') in PREFERRED_REPOS)} preferred repos)")
    return selected


def clone_repo(repo: str, base_commit: str, workdir: Path) -> bool:
    """Clone repo and checkout base_commit into workdir."""
    url = f"https://github.com/{repo}.git"
    try:
        # --filter=blob:none fetches all commits but downloads file content
        # on demand — fast and works for any base_commit depth.
        # --depth=N would fail for commits older than N revisions.
        subprocess.run(
            ['git', 'clone', '--filter=blob:none', url, str(workdir)],
            check=True, capture_output=True, timeout=300,
        )
        subprocess.run(
            ['git', 'checkout', base_commit],
            cwd=workdir, check=True, capture_output=True, timeout=60,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"  clone failed: {e.stderr.decode()[:200]}")
        return False
    except subprocess.TimeoutExpired:
        print("  clone timeout")
        return False


def set_proxy_context(task_id: str, turn: int, proxy_url: str):
    """Tell the proxy which task/turn is active."""
    try:
        payload = json.dumps({'task_id': task_id, 'turn': turn}).encode()
        req = urllib.request.Request(
            f"{proxy_url}/x-benchmark-context",
            data=payload,
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        urllib.request.urlopen(req, timeout=3)
    except Exception:
        pass   # proxy context is best-effort


def run_claude(prompt: str, workdir: Path, proxy_url: str,
               model: str, timeout: int = 600) -> tuple[str, int, float]:
    """
    Run claude -p in workdir, returning (stdout, returncode, elapsed_sec).
    Each internal LLM call goes through the proxy for TTFT capture.
    """
    env = os.environ.copy()
    env['ANTHROPIC_BASE_URL']              = proxy_url
    env['ANTHROPIC_API_KEY']               = 'dummy'
    env['ANTHROPIC_AUTH_TOKEN']            = 'dummy'
    env['ANTHROPIC_DEFAULT_OPUS_MODEL']    = model
    env['ANTHROPIC_DEFAULT_SONNET_MODEL']  = model
    env['ANTHROPIC_DEFAULT_HAIKU_MODEL']   = model
    env['CLAUDE_CODE_ATTRIBUTION_HEADER']  = '0'

    cmd = [
        'claude', '-p', prompt,
        '--model', model,
        '--output-format', 'text',
        '--no-session-persistence',
        '--dangerously-skip-permissions',   # auto-approve file edits in batch mode
    ]

    t0 = time.perf_counter()
    result = subprocess.run(
        cmd,
        cwd=workdir,
        env=env,
        capture_output=True,
        timeout=timeout,
        text=True,
    )
    elapsed = time.perf_counter() - t0

    # Surface what Claude actually said/did for debugging
    if result.stdout.strip():
        # Print last 300 chars so we can see the outcome without flooding logs
        tail = result.stdout.strip()[-300:]
        print(f"         agent output: ...{tail}")
    if result.returncode != 0 and result.stderr.strip():
        print(f"         stderr: {result.stderr.strip()[-200:]}")

    return result.stdout, result.returncode, elapsed


def get_patch(workdir: Path) -> str:
    """Return git diff of changes made by the agent."""
    try:
        r = subprocess.run(
            ['git', 'diff'],
            cwd=workdir, capture_output=True, text=True, timeout=10,
        )
        return r.stdout
    except Exception:
        return ''


def patch_size_lines(patch: str) -> tuple[int, int]:
    """Return (added_lines, removed_lines) from a unified diff."""
    added = sum(1 for line in patch.splitlines()
                if line.startswith('+') and not line.startswith('+++'))
    removed = sum(1 for line in patch.splitlines()
                  if line.startswith('-') and not line.startswith('---'))
    return added, removed


# ── main runner ───────────────────────────────────────────────────────

def run(args):
    # Resolve output dir
    tag        = args.tag or datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path('results') / tag
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks_log  = output_dir / 'tasks.jsonl'

    proxy_url  = PROXY_URL if not args.no_proxy else DYNAMO_URL
    model      = args.model

    # Verify services are up
    for name, url in [('proxy' if not args.no_proxy else 'dynamo',
                        proxy_url + '/health'),
                       ('dynamo', DYNAMO_URL + '/health')]:
        try:
            urllib.request.urlopen(url, timeout=5)
        except Exception:
            print(f"ERROR: {name} not reachable at {url}", file=sys.stderr)
            print("Start the stack first:  ./launch_dynamo.sh", file=sys.stderr)
            if not args.no_proxy:
                print("Start the proxy:         python3 ttft_proxy.py &",
                      file=sys.stderr)
            sys.exit(1)

    tasks = load_tasks(args.n, filter_repo=args.filter)
    print(f"\nRun tag     : {tag}")
    print(f"Output      : {output_dir}/")
    print(f"Proxy URL   : {proxy_url}")
    print(f"Model       : {model}")
    print(f"Tasks       : {len(tasks)}")
    print()

    workbase = Path(tempfile.mkdtemp(prefix='swebench_'))
    print(f"Workspace   : {workbase}/")
    print()

    summary = {'resolved': 0, 'failed_clone': 0, 'failed_run': 0,
               'patched': 0, 'total': len(tasks)}

    for i, task in enumerate(tasks):
        instance_id    = task['instance_id']
        repo           = task['repo']
        base_commit    = task['base_commit']
        problem        = task['problem_statement']
        # Some tasks have hints
        hints          = task.get('hints_text', '')
        if hints:
            problem += f"\n\nHINTS:\n{hints}"

        print(f"[{i+1:3d}/{len(tasks)}] {instance_id}")
        print(f"         repo   : {repo}@{base_commit[:8]}")

        workdir = workbase / instance_id
        workdir.mkdir(parents=True, exist_ok=True)

        # ── clone ─────────────────────────────────────────────────────
        print(f"         cloning...")
        if not clone_repo(repo, base_commit, workdir):
            summary['failed_clone'] += 1
            _write_task_result(tasks_log, {
                'instance_id': instance_id,
                'repo':        repo,
                'status':      'failed_clone',
            })
            continue

        # ── run agent ─────────────────────────────────────────────────
        prompt = CLAUDE_TASK_PROMPT.format(problem_statement=problem)
        set_proxy_context(instance_id, 0, proxy_url)

        print(f"         running agent...")
        t_start = time.time()
        try:
            stdout, returncode, elapsed = run_claude(
                prompt, workdir, proxy_url, model, timeout=args.timeout
            )
        except subprocess.TimeoutExpired:
            print(f"         TIMEOUT after {args.timeout}s")
            summary['failed_run'] += 1
            _write_task_result(tasks_log, {
                'instance_id': instance_id,
                'repo':        repo,
                'status':      'timeout',
                'elapsed_sec': args.timeout,
            })
            continue
        except Exception as e:
            print(f"         ERROR: {e}")
            summary['failed_run'] += 1
            _write_task_result(tasks_log, {
                'instance_id': instance_id,
                'repo':        repo,
                'status':      f'error: {e}',
            })
            continue

        # ── collect patch ──────────────────────────────────────────────
        patch    = get_patch(workdir)
        added, removed = patch_size_lines(patch)
        patched  = bool(patch.strip())

        patch_file = output_dir / f"{instance_id}.patch"
        if patched:
            patch_file.write_text(patch)
            summary['patched'] += 1
            print(f"         patch  : +{added}/-{removed} lines  ({elapsed:.0f}s)")
        else:
            print(f"         no changes made  ({elapsed:.0f}s)")

        summary['resolved'] += 1   # task ran to completion (may or may not be correct)

        _write_task_result(tasks_log, {
            'instance_id':   instance_id,
            'repo':          repo,
            'status':        'completed',
            'patched':       patched,
            'patch_lines_added':   added,
            'patch_lines_removed': removed,
            'elapsed_sec':   round(elapsed, 1),
            'returncode':    returncode,
        })

        # Clean up repo to save disk (keep patch)
        shutil.rmtree(workdir, ignore_errors=True)

    # ── summary ────────────────────────────────────────────────────────
    print(f"\n{'═'*55}")
    print(f"  RUN COMPLETE  [{tag}]")
    print(f"{'═'*55}")
    print(f"  Total tasks   : {summary['total']}")
    print(f"  Completed     : {summary['resolved']}")
    print(f"  Patched       : {summary['patched']}")
    print(f"  Clone failed  : {summary['failed_clone']}")
    print(f"  Run failed    : {summary['failed_run']}")
    print(f"\n  Task results  : {tasks_log}")
    print(f"  Patches       : {output_dir}/*.patch")
    if not args.no_proxy:
        print(f"  TTFT data     : ttft_log.jsonl  (analyze with analyze_results.py)")
    print(f"{'═'*55}\n")

    shutil.rmtree(workbase, ignore_errors=True)


def _write_task_result(log_path: Path, record: dict):
    with open(log_path, 'a') as f:
        f.write(json.dumps({**record, 'ts': time.time()}) + '\n')


# ── entry point ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--n',        type=int, default=10,
                        help='Number of tasks to run (default 10)')
    parser.add_argument('--filter',   default=None,
                        help='Only run tasks whose repo contains this string')
    parser.add_argument('--tag',      default=None,
                        help='Label for this run (e.g. "cpu" or "chip")')
    parser.add_argument('--model',    default=MODEL_ENV,
                        help='Model name')
    parser.add_argument('--timeout',  type=int, default=600,
                        help='Per-task timeout in seconds (default 600)')
    parser.add_argument('--no-proxy', action='store_true',
                        help='Skip TTFT proxy (use Dynamo directly)')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
