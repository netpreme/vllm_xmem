#!/usr/bin/env python3
"""Build a per-turn CSV from a capture directory.

For each /v1/messages turn across all sessions in a capture, emit one row:
    session_id, turn_idx_in_session, t_session_start, t_request, t_response_end,
    duration_s, isl, osl, isl_new, status

Where:
    isl     = captured input_tokens  (parsed from SSE message_start usage)
    osl     = captured output_tokens (parsed from final SSE message_delta usage)
    isl_new = isl[N] - (isl[N-1] + osl[N-1])   for N > 0 within a session
            = isl                              for N == 0

This approximates the "uncached" portion of the prompt: the new content added
since the previous turn (tool results + user follow-up). Negative values can
occur if claude prunes context — we clip to 0 and flag with isl_new_clipped.

Usage:
    extract_per_turn.py --capture-dir <path> --output per_turn.csv
"""
import argparse
import csv
import json
import re
from pathlib import Path


# Backfill ISL/OSL from SSE for captures that pre-date the
# input_tokens/output_tokens fields in the trace JSONL.
_IN_RE  = re.compile(r'"input_tokens"\s*:\s*(\d+)')
_OUT_RE = re.compile(r'"output_tokens"\s*:\s*(\d+)')


def _backfill(entry: dict, key: str, pattern: re.Pattern) -> int | None:
    v = entry.get(key)
    if v is not None:
        return v
    sse = entry.get("response_sse")
    if not isinstance(sse, str):
        return None
    if key == "input_tokens":
        m = pattern.search(sse)        # first occurrence (message_start)
        return int(m.group(1)) if m else None
    last = None
    for m in pattern.finditer(sse):    # last occurrence (final usage)
        last = m
    return int(last.group(1)) if last else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture-dir", required=True)
    ap.add_argument("--output",      required=True)
    ap.add_argument("--include-non-messages", action="store_true",
        help="Also include HEAD / and count_tokens turns (default: skip)")
    args = ap.parse_args()

    capture_dir = Path(args.capture_dir).expanduser().resolve()
    out_path    = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load session-start records (preserves session order + t_session_start)
    sessions = []
    with open(capture_dir / "sessions.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sessions.append(json.loads(line))
    sessions.sort(key=lambda r: r["t_session_start"])

    fieldnames = [
        "session_id", "turn_idx_in_session",
        "t_session_start", "t_request", "t_response_end", "duration_s",
        "method", "path", "status",
        "isl", "osl", "isl_new", "isl_new_clipped",
    ]

    rows = []
    n_sessions_seen = 0
    n_turns_total   = 0
    n_msg_turns     = 0

    for s in sessions:
        n_sessions_seen += 1
        trace_path = capture_dir / "traces" / s["trace_file"]
        if not trace_path.exists():
            continue

        # Walk turns in order; maintain running ISL_{N-1} + OSL_{N-1} for the
        # session so we can compute the uncached delta.
        turn_idx = 0
        prev_isl: int | None = None
        prev_osl: int | None = None

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
                n_turns_total += 1

                path = e.get("path", "")
                is_msg = (path == "/v1/messages")
                if not args.include_non_messages and not is_msg:
                    continue

                isl = _backfill(e, "input_tokens",  _IN_RE)
                osl = _backfill(e, "output_tokens", _OUT_RE)

                if is_msg:
                    n_msg_turns += 1

                # uncached_isl computation (per session)
                if isl is None:
                    isl_new = None
                    clipped = False
                elif prev_isl is None or prev_osl is None:
                    isl_new = isl   # first turn (no prior context in this session)
                    clipped = False
                else:
                    raw = isl - (prev_isl + prev_osl)
                    if raw < 0:
                        isl_new = 0
                        clipped = True
                    else:
                        isl_new = raw
                        clipped = False

                rows.append({
                    "session_id":          s["instance_id"],
                    "turn_idx_in_session": turn_idx,
                    "t_session_start":     round(s["t_session_start"], 4),
                    "t_request":           round(e.get("t_request", 0.0), 4),
                    "t_response_end":      round(e.get("t_response_end", 0.0), 4),
                    "duration_s":          round((e.get("t_response_end", 0.0) - e.get("t_request", 0.0)), 4),
                    "method":              e.get("method"),
                    "path":                path,
                    "status":              e.get("status"),
                    "isl":                 isl,
                    "osl":                 osl,
                    "isl_new":             isl_new,
                    "isl_new_clipped":     int(clipped),
                })

                # Advance running state ONLY for /v1/messages turns (the
                # health-check and count_tokens turns don't change the chat state)
                if is_msg and isl is not None and osl is not None:
                    prev_isl, prev_osl = isl, osl
                turn_idx += 1

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Print a small summary
    def _stats(vals):
        vs = [v for v in vals if isinstance(v, (int, float))]
        if not vs: return "n/a"
        vs.sort()
        return f"n={len(vs)} min={vs[0]} p50={vs[len(vs)//2]} p95={vs[int(len(vs)*0.95)]} max={vs[-1]}"

    print(f"Wrote {len(rows)} rows → {out_path}")
    print(f"  sessions={n_sessions_seen}  total_turns={n_turns_total}  msg_turns={n_msg_turns}")
    print(f"  ISL    stats: {_stats([r['isl']     for r in rows])}")
    print(f"  OSL    stats: {_stats([r['osl']     for r in rows])}")
    print(f"  ISL_new stats: {_stats([r['isl_new'] for r in rows])}")
    n_clipped = sum(1 for r in rows if r.get('isl_new_clipped'))
    if n_clipped:
        print(f"  Note: {n_clipped} turn(s) had isl_new < 0 (context pruned by claude); clipped to 0.")


if __name__ == "__main__":
    main()
