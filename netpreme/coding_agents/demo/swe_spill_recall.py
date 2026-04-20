#!/usr/bin/env python3
"""
swe_spill_recall.py — HBM fill → spill → recall using real SWE-bench problems

PHASES
──────
  Turn 1 (fill-HBM):       full context + issue, max_tokens=32  → fills GPU KV
  Turn 2 (spill-to-offload): "which files need to change?", max_tokens=64 → spills to external
  Turns 3+ (recall):        "in one word: what was the bug?", max_tokens=1 → pure KV fetch

Session 0 (hardest problem) streams tokens live to terminal each turn.
All sessions show a timing table with scaled, segmented bars.

USAGE
─────
  pip install datasets
  bash demo/start_servers.sh --no-cap
  python3 demo/swe_spill_recall.py
  python3 demo/swe_spill_recall.py -c 4 -r 3
"""

import argparse
import asyncio
import json as _json
import os
import re
import shutil
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL              = "qwen/qwen3-coder-30b-a3b-instruct-fp8"
CPU_URL            = "http://localhost:8001"
MTIER_URL          = "http://localhost:8002"
BLOCK_SIZE         = 16
KV_BYTES_PER_TOKEN = 49_152
TOKENS_PER_REPEAT  = 106

# ── Colours ─────────────────────────────────────────────────────────────────────
BLUE   = "\033[94m"
ORANGE = "\033[38;5;208m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RED    = "\033[91m"
GRAY   = "\033[90m"
RESET  = "\033[0m"

# bar segment colours: queue=AMBER  kv_load=CORNFLOWER_BLUE  prefill=SAGE_GREEN
BAR_QUEUE   = "\033[38;5;214m"  # amber/gold     — waiting/wasted time
BAR_KV      = "\033[38;5;75m"   # cornflower blue — data in flight
BAR_PREFILL = "\033[38;5;114m"  # sage green      — compute completing

# ── Table column widths (VISIBLE chars only — never include ANSI codes in widths)
PROB_W = 40   # problem name column
TTFT_W = 9    # "   2359ms"
BAR_W  = 20   # bar column — _segment_bar always returns exactly this many visible chars
SPD_W  = 7    # " 6.55× "

# ── GitHub context cache ────────────────────────────────────────────────────────
CACHE_DIR = Path.home() / ".cache" / "swe_demo"


# ── Data classes ────────────────────────────────────────────────────────────────
@dataclass
class SWEProblem:
    instance_id:   str
    repo:          str
    difficulty:    str
    statement:     str
    fill_context:  str   # turn-1 user message: problem + first ~tps tokens of code
    spill_context: str   # turn-2 user message: next ~tps tokens of code (forces overflow)
    token_est:     int


@dataclass
class Result:
    label:   str
    problem: str
    turn:    int
    ttft_ms: Optional[float] = None
    e2e_ms:  Optional[float] = None
    tokens:  int = 0
    error:   Optional[str] = None
    reply:   str = ""


@dataclass
class MetricsSnapshot:
    queue_sum:     float = 0.0
    queue_count:   float = 0.0
    prefill_sum:   float = 0.0
    prefill_count: float = 0.0
    kv_load_time:  float = 0.0   # kv_offload_total_time_total{CPU_to_GPU}


@dataclass
class SessionState:
    problem:  SWEProblem
    messages: list[dict] = field(default_factory=list)


# ── Live stream display ─────────────────────────────────────────────────────────
class LiveStream:
    """
    Split-screen live display: CPU tokens on the left, MTier tokens on the right.
    Uses ANSI cursor-up + clear-line to re-render a fixed block on every token.
    """
    CONTENT_ROWS = 8   # lines reserved for streamed token text

    def __init__(self, prob: SWEProblem):
        self.prob     = prob
        term_w        = shutil.get_terminal_size((120, 40)).columns
        # "  " prefix + col + "  │  " (5) + col  ≤ term_w
        self.col_w    = max(30, (term_w - 9) // 2)
        self.cpu_ttft: Optional[float] = None
        self.mt_ttft:  Optional[float] = None
        self.cpu_text  = ""
        self.mt_text   = ""
        self._active   = False
        # TOTAL_ROWS = 1 header + 1 divider + CONTENT_ROWS
        self._total    = 2 + self.CONTENT_ROWS

    # ── fixed-width TTFT badge: always 12 visible chars ─────────────────────
    @staticmethod
    def _badge(ttft: Optional[float]) -> str:
        if ttft is not None:
            return f"[{ttft:>8.0f}ms]"   # [ + 8 + ms] = 12 chars
        return "[  pending  ]"            # 12 chars

    def _side_header(self, color: str, label: str, ttft: Optional[float]) -> str:
        """Header for one column: label + badge, padded to col_w visible chars."""
        badge   = self._badge(ttft)
        content = f"{label}  {badge}"          # visible length
        pad     = max(0, self.col_w - len(label) - 2 - len(badge))
        return f"{color}{label}{RESET}  {DIM}{badge}{RESET}" + " " * pad

    def start(self) -> None:
        short = self.prob.instance_id.replace("__", "/")
        print(f"\n  {BOLD}{short}{RESET}  {DIM}({self.prob.repo}){RESET}")
        sys.stdout.write("\n" * self._total)
        sys.stdout.flush()
        self._active = True
        self._redraw()

    def _redraw(self) -> None:
        if not self._active:
            return

        cpu_lines = textwrap.wrap(self.cpu_text, self.col_w) if self.cpu_text else []
        mt_lines  = textwrap.wrap(self.mt_text,  self.col_w) if self.mt_text  else []

        sys.stdout.write(f"\033[{self._total}A")

        # ── header row ────────────────────────────────────────────────────────
        cpu_hdr = self._side_header(BLUE,   "hybrid-cpu",   self.cpu_ttft)
        mt_hdr  = self._side_header(ORANGE, "hybrid-mtier", self.mt_ttft)
        sys.stdout.write(f"\033[2K  {cpu_hdr}  │  {mt_hdr}\n")

        # ── divider ───────────────────────────────────────────────────────────
        sys.stdout.write(f"\033[2K  {'─' * self.col_w}  │  {'─' * self.col_w}\n")

        # ── content rows ──────────────────────────────────────────────────────
        for i in range(self.CONTENT_ROWS):
            cl = (cpu_lines[i] if i < len(cpu_lines) else "").ljust(self.col_w)[:self.col_w]
            ml = (mt_lines[i]  if i < len(mt_lines)  else "").ljust(self.col_w)[:self.col_w]
            sys.stdout.write(f"\033[2K  {BLUE}{cl}{RESET}  │  {ORANGE}{ml}{RESET}\n")

        sys.stdout.flush()

    def on_prefill(self, key: str, ttft_ms: float, token: str) -> None:
        if key == "cpu":
            self.cpu_ttft = ttft_ms
            self.cpu_text = token
        else:
            self.mt_ttft  = ttft_ms
            self.mt_text  = token
        self._redraw()

    def on_decode(self, key: str, token: str) -> None:
        if key == "cpu":
            self.cpu_text += token
        else:
            self.mt_text  += token
        self._redraw()

    def finish(self) -> None:
        self._redraw()
        print()   # blank line before the table


# ── Metrics ─────────────────────────────────────────────────────────────────────
async def fetch_metrics_snapshot(url: str) -> MetricsSnapshot:
    snap = MetricsSnapshot()
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(f"{url}/metrics", timeout=10)
        for line in r.text.splitlines():
            if line.startswith("#"):
                continue
            if "request_queue_time_seconds_sum{"   in line:
                snap.queue_sum    = float(line.split()[-1])
            elif "request_queue_time_seconds_count{" in line:
                snap.queue_count  = float(line.split()[-1])
            elif "request_prefill_time_seconds_sum{" in line:
                snap.prefill_sum  = float(line.split()[-1])
            elif "request_prefill_time_seconds_count{" in line:
                snap.prefill_count = float(line.split()[-1])
            elif "kv_offload_total_time_total" in line and "CPU_to_GPU" in line:
                snap.kv_load_time = float(line.split()[-1])
    except Exception:
        pass
    return snap


def breakdown_ms(before: MetricsSnapshot, after: MetricsSnapshot,
                 n: int) -> tuple[float, float, float]:
    """Return (avg_queue_ms, avg_kv_load_ms, avg_prefill_ms) per request."""
    dq  = after.queue_count   - before.queue_count
    dp  = after.prefill_count - before.prefill_count
    q   = (after.queue_sum    - before.queue_sum)   / dq * 1000 if dq > 0 else 0.0
    p   = (after.prefill_sum  - before.prefill_sum) / dp * 1000 if dp > 0 else 0.0
    kv  = (after.kv_load_time - before.kv_load_time) / max(n, 1) * 1000
    return q, kv, p


# ── Visualisation ────────────────────────────────────────────────────────────────
def _segment_bar(q_ms: float, kv_ms: float, p_ms: float,
                 ref_ms: float, width: int = BAR_W) -> str:
    """
    Returns exactly `width` VISIBLE characters (block chars + spaces).
    RED=queue  CYAN=kv_load  GREEN=prefill  trailing spaces = empty
    Never use a format-spec width on the returned string — it already has correct visible width.
    """
    total = q_ms + kv_ms + p_ms
    # scale bar length to ref_ms
    filled = max(1, round(total / max(ref_ms, 1.0) * width)) if total > 0 else 1
    filled = min(filled, width)
    if total > 0:
        q_w  = max(0, round(q_ms  / total * filled))
        p_w  = max(0, round(p_ms  / total * filled))
        kv_w = max(0, filled - q_w - p_w)
    else:
        q_w, kv_w, p_w = 0, filled, 0
    spaces = width - filled
    return BAR_QUEUE + "█" * q_w + BAR_KV + "█" * kv_w + BAR_PREFILL + "█" * p_w + RESET + " " * spaces


def _diff_tag(d: str) -> str:
    c = RED if d == "hard" else (YELLOW if d == "medium" else GREEN)
    return f"{c}[{d[0].upper()}]{RESET}"


def _ttft_str(ms: Optional[float]) -> str:
    return f"{ms:>7.0f}ms" if ms else "       ?"


def _pct(rs: list[Result], p: float) -> Optional[float]:
    v = sorted(r.ttft_ms for r in rs if r.ttft_ms and not r.error)
    if not v:
        return None
    idx = (len(v) - 1) * p / 100.0
    lo  = int(idx)
    hi  = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (idx - lo)


def _speedup_str(ct: float, mt: float) -> tuple[str, str]:
    if not (ct and mt):
        return "      ?", RESET
    s = ct / mt
    c = GREEN if s >= 2.0 else (YELLOW if s >= 1.1 else RESET)
    return f"{s:>6.2f}×", c


STACK_BAR_W = 44   # wider bar for stacked layout

def print_turn_header(turn: int, c: int, phase: str) -> None:
    print(f"\n{BOLD}━━━ Turn {turn}  ({c} sessions)  [{phase}]{RESET}")
    print(f"  {'':3}  {'Problem':<{PROB_W}}  {'ms':>{TTFT_W}}  bar ({STACK_BAR_W} chars = slowest request)")


def _make_bars(
    ct: float, mt: float, ref_ms: float,
    cpu_q: float, cpu_kv: float, cpu_p: float,
    mt_q:  float, mt_kv:  float, mt_p:  float,
    have_metrics: bool,
) -> tuple[str, str]:
    W = STACK_BAR_W
    if have_metrics and ct and mt:
        shared_p = (cpu_p + mt_p) / 2 if (cpu_p and mt_p) else (cpu_p or mt_p or 0)
        cpu_var  = max(0.0, ct - shared_p)
        mt_var   = max(0.0, mt - shared_p)
        cpu_qkv  = max(cpu_q + cpu_kv, 1.0)
        mt_qkv   = max(mt_q  + mt_kv,  1.0)
        cb = _segment_bar(cpu_var * cpu_q / cpu_qkv, cpu_var * cpu_kv / cpu_qkv, shared_p, ref_ms, W)
        mb = _segment_bar(mt_var  * mt_q  / mt_qkv,  mt_var  * mt_kv  / mt_qkv,  shared_p, ref_ms, W)
    elif ct and mt:
        cb = _segment_bar(0, ct, 0, ref_ms, W)
        mb = _segment_bar(0, mt, 0, ref_ms, W)
    else:
        cb = mb = " " * W
    return cb, mb


def print_problem_rows(
    problems:    list[SWEProblem],
    cpu_results: list[Result],
    mt_results:  list[Result],
    ref_ms:      float,
    cpu_q: float, cpu_kv: float, cpu_p: float,
    mt_q:  float, mt_kv:  float, mt_p:  float,
    have_metrics: bool,
) -> None:
    indent = " " * (3 + 2 + PROB_W + 2)   # aligns bar under problem name
    for i, prob in enumerate(problems):
        cr = cpu_results[i]
        mr = mt_results[i]
        ct = cr.ttft_ms
        mt = mr.ttft_ms

        cpu_bar, mt_bar = _make_bars(ct or 0, mt or 0, ref_ms,
                                     cpu_q, cpu_kv, cpu_p,
                                     mt_q,  mt_kv,  mt_p, have_metrics)
        sp_str, sp_col = _speedup_str(ct or 0, mt or 0)

        short_id = prob.instance_id.replace("__", "/")
        if len(short_id) > PROB_W:
            short_id = short_id[:PROB_W - 1] + "…"

        # Row 1: tag + problem name
        print(f"  {_diff_tag(prob.difficulty)}  {short_id:<{PROB_W}}")
        # Row 2: TTFT for both servers on one line + speedup
        print(f"  {indent}TTFT: {BLUE}cpu{_ttft_str(ct)}{RESET}  {ORANGE}mtier{_ttft_str(mt)}{RESET}  {sp_col}{sp_str}{RESET}")


def print_turn_summary(
    cpu_results: list[Result], mt_results: list[Result],
    cpu_q: float, cpu_kv: float, cpu_p: float,
    mt_q:  float, mt_kv:  float, mt_p:  float,
    have_metrics: bool, phase: str, turn: int = 0,
) -> None:
    ct50 = _pct(cpu_results, 50); ct95 = _pct(cpu_results, 95)
    mt50 = _pct(mt_results,  50); mt95 = _pct(mt_results,  95)
    if not (ct50 and mt50):
        return

    sp_str, sp_col = _speedup_str(ct50, mt50)
    direction = "MTier faster" if ct50 / mt50 > 1.05 else ("CPU faster" if ct50 / mt50 < 0.95 else "similar")
    ref = max(ct50, mt50)

    if have_metrics:
        shared_p = (cpu_p + mt_p) / 2 if (cpu_p and mt_p) else (cpu_p or mt_p or 0)
        cpu_var  = max(0.0, ct50 - shared_p)
        mt_var   = max(0.0, mt50 - shared_p)
        cpu_qkv  = max(cpu_q + cpu_kv, 1.0)
        mt_qkv   = max(mt_q  + mt_kv,  1.0)
        cpu_bar  = _segment_bar(cpu_var * cpu_q / cpu_qkv, cpu_var * cpu_kv / cpu_qkv, shared_p, ref, 32)
        mt_bar   = _segment_bar(mt_var  * mt_q  / mt_qkv,  mt_var  * mt_kv  / mt_qkv,  shared_p, ref, 32)
    else:
        cpu_bar = _segment_bar(0, ct50, 0, ref, 32)
        mt_bar  = _segment_bar(0, mt50, 0, ref, 32)

    def _p(v): return f"{v:>7.0f}ms" if v else "       ?"
    leg = f"  {DIM}legend: {BAR_QUEUE}█{RESET}{DIM} queue  {BAR_KV}█{RESET}{DIM} kv_load  {BAR_PREFILL}█{RESET}{DIM} prefill{RESET}"
    print(f"\n{leg}")
    print(f"  {DIM}  Summary [{phase}]{RESET}")
    print(f"  {BLUE}{'hybrid-cpu':<14}{RESET}   p50 {_p(ct50)}  p95 {_p(ct95)}  {cpu_bar}")
    print(f"  {ORANGE}{'hybrid-mtier':<14}{RESET}   p50 {_p(mt50)}  p95 {_p(mt95)}  {mt_bar}")
    print(f"\n  {BOLD}MTier vs CPU p50:  {sp_col}{sp_str}  {direction}{RESET}")
    if turn <= 2:
        print(f"  {DIM}ⓘ  Turns 1–2 are HBM fill/spill — no external KV transfers yet, results expected to be similar{RESET}")


# ── Request sender ───────────────────────────────────────────────────────────────
async def send_request(
    client:     httpx.AsyncClient,
    base_url:   str,
    label:      str,
    problem_id: str,
    turn:       int,
    messages:   list[dict],
    max_tokens: int,
    live:       Optional[LiveStream] = None,
    live_key:   Optional[str] = None,
) -> Result:
    r  = Result(label=label, problem=problem_id, turn=turn)
    t0 = time.monotonic()
    try:
        async with client.stream(
            "POST", f"{base_url}/v1/chat/completions",
            json={
                "model":       MODEL,
                "messages":    messages,
                "stream":      True,
                "max_tokens":  max_tokens,
                "temperature": 0.0,
                "seed":        42,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=600,
        ) as resp:
            resp.raise_for_status()
            first = True
            parts: list[str] = []
            async for raw in resp.aiter_lines():
                if not raw.startswith("data:"):
                    continue
                body = raw[5:].strip()
                if body == "[DONE]":
                    break
                try:
                    obj   = _json.loads(body)
                    delta = obj["choices"][0]["delta"].get("content", "")
                    if delta:
                        if first:
                            r.ttft_ms = (time.monotonic() - t0) * 1000
                            first = False
                            if live and live_key:
                                live.on_prefill(live_key, r.ttft_ms, delta)
                        else:
                            if live and live_key:
                                live.on_decode(live_key, delta)
                        parts.append(delta)
                        r.tokens += 1
                except Exception:
                    pass
            r.e2e_ms = (time.monotonic() - t0) * 1000
            r.reply  = "".join(parts)
    except Exception as exc:
        r.error  = str(exc)[:80]
        r.e2e_ms = (time.monotonic() - t0) * 1000
    return r


# ── Server probe ─────────────────────────────────────────────────────────────────
async def get_gpu_blocks(url: str) -> int:
    async with httpx.AsyncClient() as c:
        r = await c.get(f"{url}/metrics", timeout=10)
    for line in r.text.splitlines():
        if "num_gpu_blocks=" in line and "override" not in line.lower():
            m = re.search(r'num_gpu_blocks="(\d+)"', line)
            if m:
                return int(m.group(1))
    for line in r.text.splitlines():
        if "num_gpu_blocks=" in line:
            m = re.search(r'num_gpu_blocks="(\d+)"', line)
            if m:
                return int(m.group(1))
    raise RuntimeError("Could not read num_gpu_blocks from /metrics")


# ── Problem loading ───────────────────────────────────────────────────────────────
def _patch_difficulty(patch: str) -> str:
    n = len(patch.splitlines())
    return "hard" if n >= 150 else ("medium" if n >= 60 else "easy")


async def _gh_get(client: httpx.AsyncClient, url: str, token: Optional[str]) -> Optional[list]:
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"
    try:
        r = await client.get(url, headers=headers, timeout=15)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 403:
            print(f"\n  {YELLOW}GitHub rate limit. Set GITHUB_TOKEN env var for 5000 req/hr.{RESET}")
    except Exception:
        pass
    return None


async def _fetch_raw(client: httpx.AsyncClient, url: str) -> Optional[str]:
    try:
        r = await client.get(url, timeout=15, follow_redirects=True)
        return r.text if r.status_code == 200 else None
    except Exception:
        return None


async def _fetch_all_repo_files(row: dict, target_chars: int, token: Optional[str]) -> list[str]:
    """
    Fetch .py files from GitHub for this repo/commit until target_chars is reached.
    Returns a list of "# path\n<code>\n" strings in order of relevance.
    Cached to ~/.cache/swe_demo/{instance_id}.txt.
    """
    instance_id = row["instance_id"]
    repo        = row["repo"]
    commit      = row["base_commit"]
    patch       = row.get("patch", "")

    cache_path = CACHE_DIR / f"{instance_id}.txt"
    if cache_path.exists():
        cached = cache_path.read_text(encoding="utf-8", errors="replace")
        if len(cached) >= target_chars * 0.8:
            # Split back into individual file entries
            return [s + "\n" for s in cached.split("\n\n") if s.strip()]

    patch_files = list(dict.fromkeys(
        re.findall(r'^\+\+\+ b/(.+)$', patch, re.MULTILINE)
    ))
    raw_base = f"https://raw.githubusercontent.com/{repo}/{commit}"

    parts: list[str] = []
    seen:  set[str]  = set()
    total_chars = 0

    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:

        async def add_file(path: str) -> None:
            nonlocal total_chars
            if path in seen or total_chars >= target_chars:
                return
            if not path.endswith((".py", ".md", ".rst", ".txt", ".toml", ".cfg")):
                return
            seen.add(path)
            text = await _fetch_raw(client, f"{raw_base}/{path}")
            if text:
                entry = f"# {path}\n{text.rstrip()}\n"
                parts.append(entry)
                total_chars += len(entry)

        async def add_dir(api_dir: str) -> None:
            if total_chars >= target_chars:
                return
            url   = f"https://api.github.com/repos/{repo}/contents/{api_dir}?ref={commit}"
            items = await _gh_get(client, url, token)
            if not isinstance(items, list):
                return
            for p in sorted(x["path"] for x in items
                            if x.get("type") == "file" and x["path"].endswith(".py")):
                if total_chars >= target_chars:
                    break
                await add_file(p)

        # Round 1: patched files (most relevant)
        for p in patch_files:
            await add_file(p)

        # Round 2: siblings in same dirs
        if total_chars < target_chars:
            for d in list(dict.fromkeys(p.rsplit("/", 1)[0] for p in patch_files if "/" in p)):
                await add_dir(d)

        # Round 3: parent package dirs
        if total_chars < target_chars:
            for pd in list(dict.fromkeys(
                p.rsplit("/", 2)[0] for p in patch_files if p.count("/") >= 2
            )):
                if total_chars >= target_chars:
                    break
                url   = f"https://api.github.com/repos/{repo}/contents/{pd}?ref={commit}"
                items = await _gh_get(client, url, token)
                if isinstance(items, list):
                    for sd in sorted(x["path"] for x in items if x.get("type") == "dir"):
                        if total_chars >= target_chars:
                            break
                        await add_dir(sd)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("\n\n".join(p.rstrip() for p in parts), encoding="utf-8")
    return parts


async def _build_problem(row: dict, tps: int, token: Optional[str]) -> SWEProblem:
    """
    Build fill_context (~tps tokens) and spill_context (~tps tokens) from real repo files.

    fill_context  = problem statement + hints + first batch of code files
    spill_context = next batch of code files (sent as turn-2 user message)

    Combined conversation after turn 2:
      [sys, user(fill+Q1), asst(32), user(spill+Q2)]
      ≈ 2 × tps tokens  >>  HBM capacity  →  ~50% spills to external
    """
    stmt   = row["problem_statement"].strip()
    hints  = row.get("hints_text", "").strip()
    diff   = _patch_difficulty(row.get("patch", ""))
    # fetch 2× tps chars worth of code so we can split into fill and spill batches
    files  = await _fetch_all_repo_files(row, tps * 4 * 2, token)

    header = (
        f"Repository: {row['repo']}\n"
        f"Issue: {row['instance_id']}\n\n"
        f"Problem Statement:\n{stmt}\n\n"
        + (f"Hints:\n{hints}\n\n" if hints else "")
        + "Repository source files:\n\n"
    )
    header_chars = len(header)
    fill_target  = tps * 4 - header_chars   # chars remaining for code in fill turn

    fill_code  = ""
    spill_code = ""
    used_fill  = 0
    for f in files:
        if used_fill < fill_target:
            fill_code += f
            used_fill += len(f)
        else:
            spill_code += f

    fill_context  = header + fill_code
    spill_context = (
        f"Additional repository context for {row['instance_id']}:\n\n{spill_code}"
        if spill_code else
        f"Additional context: same repository, reviewing broader codebase for {row['instance_id']}.\n\n{fill_code}"
    )
    spill_context += f"\n\n{TURN2_Q}"

    return SWEProblem(
        instance_id   = row["instance_id"],
        repo          = row["repo"],
        difficulty    = diff,
        statement     = stmt,
        fill_context  = fill_context,
        spill_context = spill_context,
        token_est     = len(fill_context) // 4,
    )


async def load_swe_problems(n: int, tps: int) -> list[SWEProblem]:
    try:
        from datasets import load_dataset
    except ImportError:
        print(f"{RED}Missing: pip install datasets{RESET}", file=sys.stderr)
        sys.exit(1)

    token = os.environ.get("GITHUB_TOKEN")

    print(f"  Loading SWE-bench_Verified…", end="", flush=True)
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    print(f" {len(ds)} problems")

    rows = sorted(ds, key=lambda r: -len(r.get("patch", "").splitlines()))[:n]

    print(f"  Fetching repo source files (2× tps each, cached to {CACHE_DIR})…")
    problems = await asyncio.gather(*[_build_problem(r, tps, token) for r in rows])

    for p in problems:
        dc = RED if p.difficulty == "hard" else (YELLOW if p.difficulty == "medium" else GREEN)
        spill_k = len(p.spill_context) // 4 // 1000
        cached = "cache" if (CACHE_DIR / f"{p.instance_id}.txt").exists() else "live "
        print(f"  {dc}[{p.difficulty:6s}]{RESET}  {p.instance_id:<50}  "
              f"fill ~{p.token_est//1000}K + spill ~{spill_k}K tok  [{cached}]")

    return list(problems)


# ── Prompts ──────────────────────────────────────────────────────────────────────
TURN1_Q  = "/no_think In 2-3 sentences: what is the root cause of this bug?"
TURN2_Q  = "/no_think Which specific files and functions need to change?"
RECALL_Q = "/no_think In one short phrase: what component contains the bug?"


# ── Main ─────────────────────────────────────────────────────────────────────────
async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency",  "-c", type=int, default=8)
    parser.add_argument("--recall-turns", "-r", type=int, default=3)
    parser.add_argument("--gpu-blocks",         type=int, default=0)
    args = parser.parse_args()

    print(f"\n{BOLD}SWE-bench HBM Fill → Spill → Recall{RESET}")
    print(f"  {BLUE}blue   = hybrid-cpu   (port 8001){RESET}")
    print(f"  {ORANGE}orange = hybrid-mtier (port 8002){RESET}")

    async with httpx.AsyncClient() as c:
        for url, lbl in [(CPU_URL, "CPU (8001)"), (MTIER_URL, "MTier (8002)")]:
            try:
                (await c.get(f"{url}/health", timeout=5)).raise_for_status()
                print(f"  ✓ {lbl} healthy")
            except Exception as e:
                print(f"  {RED}✗ {lbl} not ready: {e}{RESET}", file=sys.stderr)
                sys.exit(1)

    num_gpu_blocks = args.gpu_blocks or await get_gpu_blocks(CPU_URL)
    gpu_token_cap  = num_gpu_blocks * BLOCK_SIZE
    tps            = min(95_000, max(500, gpu_token_cap // args.concurrency))  # tokens per session

    if num_gpu_blocks <= 2048:
        print(f"\n  {YELLOW}⚠  num_gpu_blocks={num_gpu_blocks} — restart with: bash demo/start_servers.sh --no-cap{RESET}")

    recall_kv_mb = 2 * tps * KV_BYTES_PER_TOKEN / 1e6
    print(f"\n  GPU KV capacity  : {gpu_token_cap:,} tokens  [{num_gpu_blocks} blocks × {BLOCK_SIZE}]")
    print(f"  Tokens/session   : ~{tps:,}  (fills HBM at c={args.concurrency})")
    print(f"  KV per recall    : {recall_kv_mb:.0f} MB/session")
    print(f"  PCIe ceiling     : {recall_kv_mb/32_000*1000:.0f} ms @ 32 GB/s (CPU)")
    print(f"  D2D ceiling      : {recall_kv_mb/300_000*1000:.0f} ms @ 300 GB/s (MTier est)")

    print(f"\n{BOLD}Loading SWE problems (hardest → easy)…{RESET}")
    problems = await load_swe_problems(args.concurrency, tps)

    import random
    sid = random.getrandbits(32)

    def sys_msg(p: SWEProblem) -> dict:
        return {"role": "system", "content":
                f"[s:{sid:08x}][{p.instance_id}] You are a senior software engineer. Be concise."}

    cpu_sessions = [SessionState(p, [sys_msg(p)]) for p in problems]
    mt_sessions  = [SessionState(p, [sys_msg(p)]) for p in problems]

    all_results: list[tuple[list[Result], list[Result]]] = []

    for turn in range(1, 2 + args.recall_turns + 1):
        if turn == 1:
            phase, max_tokens = "fill-HBM",         32
        elif turn == 2:
            phase, max_tokens = "spill-to-offload",  64
        else:
            phase, max_tokens = f"recall-t{turn}",   10

        # Build per-session user message content
        # Turn 1: fill_context (~tps tokens of code + problem statement) + TURN1_Q
        # Turn 2: spill_context (~tps tokens more code) — already has TURN2_Q embedded
        # Turn 3+: short recall question only
        for i in range(args.concurrency):
            if turn == 1:
                content = problems[i].fill_context + f"\n\n{TURN1_Q}"
            elif turn == 2:
                content = problems[i].spill_context   # already ends with TURN2_Q
            else:
                content = RECALL_Q
            cpu_sessions[i].messages.append({"role": "user", "content": content})
            mt_sessions[i].messages.append( {"role": "user", "content": content})

        # Prometheus snapshot before
        cpu_before, mt_before = await asyncio.gather(
            fetch_metrics_snapshot(CPU_URL),
            fetch_metrics_snapshot(MTIER_URL),
        )

        # Live stream for session 0 (hardest problem)
        live = LiveStream(problems[0])
        live.start()

        # Fire all requests concurrently; session 0 gets live callbacks
        async with httpx.AsyncClient() as client:
            tasks = []
            for i in range(args.concurrency):
                lv, lk_cpu, lk_mt = (live, "cpu", "mt") if i == 0 else (None, None, None)
                tasks.append(send_request(client, CPU_URL,   "hybrid-cpu",
                                          problems[i].instance_id, turn,
                                          list(cpu_sessions[i].messages), max_tokens,
                                          live=lv, live_key=lk_cpu))
                tasks.append(send_request(client, MTIER_URL, "hybrid-mtier",
                                          problems[i].instance_id, turn,
                                          list(mt_sessions[i].messages), max_tokens,
                                          live=lv, live_key=lk_mt))
            all_res = await asyncio.gather(*tasks)

        live.finish()

        # Prometheus snapshot after
        cpu_after, mt_after = await asyncio.gather(
            fetch_metrics_snapshot(CPU_URL),
            fetch_metrics_snapshot(MTIER_URL),
        )

        cpu_r = [r for r in all_res if "cpu"   == r.label.split("-")[1]]
        mt_r  = [r for r in all_res if "mtier" in r.label]

        n = args.concurrency
        have_metrics = (cpu_after.queue_count - cpu_before.queue_count) > 0
        cpu_q, cpu_kv, cpu_p = breakdown_ms(cpu_before, cpu_after, n)
        mt_q,  mt_kv,  mt_p  = breakdown_ms(mt_before,  mt_after,  n)

        ref_ms = max((r.ttft_ms for r in cpu_r + mt_r if r.ttft_ms), default=1000.0)

        print_turn_header(turn, args.concurrency, phase)
        print_problem_rows(problems, cpu_r, mt_r, ref_ms,
                           cpu_q, cpu_kv, cpu_p, mt_q, mt_kv, mt_p, have_metrics)
        print_turn_summary(cpu_r, mt_r,
                           cpu_q, cpu_kv, cpu_p, mt_q, mt_kv, mt_p,
                           have_metrics, phase, turn)

        for i in range(args.concurrency):
            # Use one canonical reply for both so histories stay identical →
            # identical prefill context → deterministic outputs across servers.
            canonical = cpu_r[i].reply or mt_r[i].reply or "."
            cpu_sessions[i].messages.append({"role": "assistant", "content": canonical})
            mt_sessions[i].messages.append( {"role": "assistant", "content": canonical})

        all_results.append((cpu_r, mt_r))

        if turn == 1:
            print(f"\n  {DIM}Waiting 3s for KV writes to flush…{RESET}")
            await asyncio.sleep(3)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{BOLD}━━━ Recall Summary (turns 3+) ━━━{RESET}")
    recall = all_results[2:]
    if recall:
        all_cpu = [r for (cr, _) in recall for r in cr]
        all_mt  = [r for (_, mr) in recall for r in mr]

        ct50 = _pct(all_cpu, 50); ct95 = _pct(all_cpu, 95)
        mt50 = _pct(all_mt,  50); mt95 = _pct(all_mt,  95)
        if ct50 and mt50:
            sp_str, sp_col = _speedup_str(ct50, mt50)
            direction = "MTier faster" if ct50 / mt50 > 1.05 else ("CPU faster" if ct50 / mt50 < 0.95 else "similar")
            cpu_bw50 = recall_kv_mb / (ct50 / 1000)
            mt_bw50  = recall_kv_mb / (mt50 / 1000)
            def _p(v): return f"{v:>7.0f}ms" if v else "       ?"
            print(f"  {BLUE}hybrid-cpu   {RESET}  p50 {_p(ct50)}  p95 {_p(ct95)}   eff bw {cpu_bw50:>7.0f} MB/s")
            print(f"  {ORANGE}hybrid-mtier {RESET}  p50 {_p(mt50)}  p95 {_p(mt95)}   eff bw {mt_bw50:>7.0f} MB/s")
            print(f"\n  {BOLD}Overall speedup (p50): {sp_col}{sp_str}  {direction}{RESET}")
            print(f"  KV per session : {recall_kv_mb:.0f} MB  (~{2*tps//1000}K tokens)")


if __name__ == "__main__":
    asyncio.run(main())
