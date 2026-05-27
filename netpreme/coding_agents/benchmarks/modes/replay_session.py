"""Trace-replay engine for captured /v1/messages sessions.

Reads:
  <capture_dir>/capture_meta.json
  <capture_dir>/sessions.jsonl       — one record per session (instance_id,
                                       trace_file, t_session_start)
  <capture_dir>/traces/<instance>.jsonl

Fires each captured request at the recorded absolute time (relative to the
replay run's t0) on a FIXED schedule — turn k+1 fires at
    t0_replay + session_t_start + turn[k+1].t_request
regardless of when the backend responded to turn k. Responses are drained
and discarded — the next request always uses the captured request body.

Intended to be imported by bench_replay (the production driver) — not run
as a script."""
import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path

import aiohttp


# Opt-in length enforcement: when REPLAY_FORCE_OSL=1, rewrite each /v1/messages
# request to pin generation length to the captured output_tokens (requires the
# vllm patch in vllm/entrypoints/anthropic/ that forwards ignore_eos+min_tokens).
_FORCE_OSL = os.environ.get("REPLAY_FORCE_OSL", "0") == "1"
# Optional fixed-OSL override: when >0, every turn's OSL is forced to this value
# instead of the captured one. Only honored when _FORCE_OSL is also set.
_OSL_OVERRIDE = int(os.environ.get("REPLAY_OSL_OVERRIDE", "0") or "0")


@dataclass
class Turn:
    t_request: float
    method: str
    path: str
    request_body: dict | None
    captured_status: int | None
    captured_output_tokens: int | None
    captured_input_tokens:  int | None = None


@dataclass
class Session:
    instance_id: str
    trace_path: Path
    t_session_start: float    # seconds relative to capture-level start
    turns: list[Turn]


def load_session_meta(capture_dir: Path) -> tuple[dict, list[dict]]:
    """Return (capture_meta, [session_record, ...])."""
    meta = json.loads((capture_dir / "capture_meta.json").read_text())
    sessions: list[dict] = []
    with open(capture_dir / "sessions.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sessions.append(json.loads(line))
    sessions.sort(key=lambda r: r["t_session_start"])
    return meta, sessions


def load_session_trace(capture_dir: Path, session_record: dict) -> Session:
    trace_path = capture_dir / "traces" / session_record["trace_file"]
    turns: list[Turn] = []
    with open(trace_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("kind") != "turn":
                continue
            turns.append(Turn(
                t_request=float(e["t_request"]),
                method=e.get("method", "POST"),
                path=e.get("path", "/v1/messages"),
                request_body=e.get("request"),
                captured_status=e.get("status"),
                captured_output_tokens=e.get("output_tokens"),
                captured_input_tokens=e.get("input_tokens"),
            ))
    turns.sort(key=lambda t: t.t_request)
    return Session(
        instance_id=session_record["instance_id"],
        trace_path=trace_path,
        t_session_start=float(session_record["t_session_start"]),
        turns=turns,
    )


async def _sleep_until(monotonic_deadline: float) -> None:
    gap = monotonic_deadline - asyncio.get_event_loop().time()
    if gap > 0:
        await asyncio.sleep(gap)


@dataclass
class SessionStats:
    instance_id: str
    base_url: str
    n_turns:         int = 0
    n_ok:            int = 0
    n_error:         int = 0
    bytes_received:  int = 0
    t_first_request: float | None = None
    t_last_response: float | None = None


async def replay_one_session(
    session: Session,
    base_url: str,
    t_run_loop_start: float,        # asyncio loop time of run t=0
    http_session: aiohttp.ClientSession,
    label: str = "",
) -> SessionStats:
    """Replay one session against `base_url` on fixed schedule."""
    stats = SessionStats(instance_id=session.instance_id, base_url=base_url)

    for turn in session.turns:
        # Absolute loop time at which this request should fire.
        fire_at = t_run_loop_start + session.t_session_start + turn.t_request
        await _sleep_until(fire_at)

        if stats.t_first_request is None:
            stats.t_first_request = asyncio.get_event_loop().time()

        url = base_url.rstrip("/") + turn.path
        body = turn.request_body
        # Pin OSL via max_tokens=min_tokens=N + ignore_eos. By default N is the
        # captured output_tokens; REPLAY_OSL_OVERRIDE forces a constant N.
        if (_FORCE_OSL and isinstance(body, dict)
                and turn.path == "/v1/messages"
                and turn.captured_output_tokens is not None
                and turn.captured_output_tokens > 0):
            n = _OSL_OVERRIDE if _OSL_OVERRIDE > 0 else turn.captured_output_tokens
            body = {
                **body,
                "max_tokens":  n,
                "min_tokens":  n,
                "ignore_eos":  True,
            }
        # Send and drain. Errors are non-fatal — we continue the session schedule.
        try:
            async with http_session.request(
                turn.method, url,
                json=body,
                headers={"Content-Type": "application/json"},
            ) as resp:
                async for chunk in resp.content.iter_any():
                    stats.bytes_received += len(chunk)
                stats.n_turns += 1
                if 200 <= resp.status < 300:
                    stats.n_ok += 1
                else:
                    stats.n_error += 1
        except Exception:
            stats.n_turns += 1
            stats.n_error += 1
        stats.t_last_response = asyncio.get_event_loop().time()

    return stats


