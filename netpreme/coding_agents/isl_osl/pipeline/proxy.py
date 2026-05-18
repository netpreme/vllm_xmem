"""
Streaming-aware logging proxy for Anthropic /v1/messages.

Sits between `claude -p` and the upstream vLLM Anthropic endpoint, forwarding
bytes verbatim (SSE chunks pass through immediately) while buffering them in
parallel to extract `usage` and content-block types from the response. One
row is emitted per /v1/messages call to both a JSONL file and a sibling CSV,
tagged with whichever instance_id the client passed via the `X-Instance-Id`
header.

Per-turn fields surfaced (in addition to raw usage):
  isl                : total prompt tokens this turn
  osl                : completion tokens this turn
  isl_new            : prompt tokens NOT served from prefix cache
  isl_cached         : prompt tokens served from prefix cache
  cache_hit_rate     : isl_cached / isl  (0.0 when isl == 0)
  category           : text_only | tool_only | mixed | empty

`isl_cached` is sourced from `usage.cache_read_input_tokens`, which the
vllm_xmem fork populates from vLLM's per-request prefix-cache hit count when
the server is started with --enable-prompt-tokens-details.

Why a proxy: claude's --output-format stream-json reports `output_tokens=0`
on every assistant event (final usage only lands in the aggregate `result`),
and it splits one assistant message into one event per content block. So
per-turn ISL/OSL must be sourced from the API response itself.

Other Anthropic routes are passed through unchanged.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse


UPSTREAM: str = ""
USAGE_LOG: Path | None = None
USAGE_CSV: Path | None = None
PER_PROBLEM_CSV_DIR: Path | None = None
MAX_TOKENS_CAP: int = 0  # 0 = no clamp
# Header overrides applied to every upstream request (e.g. Anthropic auth).
INJECT_HEADERS: dict[str, str] = {}
# When True, keep client-supplied auth headers (authorization, x-api-key)
# instead of stripping them. Needed when proxying to api.anthropic.com so
# claude's OAuth bearer reaches Anthropic.
PASSTHROUGH_AUTH: bool = False
LOG_LOCK = asyncio.Lock()


def _safe_filename(s: str) -> str:
    """Filesystem-safe instance_id (replace anything that isn't alnum/_-.)."""
    return "".join(c if c.isalnum() or c in "_-." else "_" for c in s)[:200]

# Stable CSV column order for downstream tooling.
CSV_COLUMNS = [
    "ts", "instance_id", "elapsed_ms",
    "ttft_ms", "decode_ms", "itl_ms",
    "isl", "osl", "isl_new", "isl_cached", "cache_hit_rate",
    "stop_reason", "category", "num_tool_calls",
]


def categorize(content: list[dict[str, Any]] | None) -> str:
    if not isinstance(content, list):
        return "empty"
    has_text = any(
        b.get("type") == "text" and (b.get("text") or "").strip()
        for b in content
    )
    has_tool = any(b.get("type") == "tool_use" for b in content)
    if has_text and has_tool:
        return "mixed"
    if has_tool:
        return "tool_only"
    if has_text:
        return "text_only"
    return "empty"


def _enrich_row(row: dict[str, Any]) -> dict[str, Any]:
    """Compute derived fields once, keep raw fields intact."""
    inp = int(row.get("input_tokens") or 0)
    cache_cr = int(row.get("cache_creation_input_tokens") or 0)
    cache_rd = int(row.get("cache_read_input_tokens") or 0)
    out_tok = int(row.get("output_tokens") or 0)
    isl = inp + cache_cr + cache_rd
    isl_cached = cache_rd
    isl_new = inp + cache_cr
    rate = (isl_cached / isl) if isl > 0 else 0.0
    row["isl"] = isl
    row["osl"] = out_tok
    row["isl_new"] = isl_new
    row["isl_cached"] = isl_cached
    row["cache_hit_rate"] = round(rate, 4)
    return row


async def write_row(row: dict[str, Any]) -> None:
    if USAGE_LOG is None and USAGE_CSV is None and PER_PROBLEM_CSV_DIR is None:
        return
    row = _enrich_row(row)
    json_line = json.dumps(row) + "\n"

    csv_buf = io.StringIO()
    csv.DictWriter(csv_buf, fieldnames=CSV_COLUMNS, extrasaction="ignore").writerow(
        {k: row.get(k) for k in CSV_COLUMNS}
    )
    csv_line = csv_buf.getvalue()

    async with LOG_LOCK:
        if USAGE_LOG is not None:
            with USAGE_LOG.open("a") as f:
                f.write(json_line)
        if USAGE_CSV is not None:
            new_file = not USAGE_CSV.exists() or USAGE_CSV.stat().st_size == 0
            with USAGE_CSV.open("a") as f:
                if new_file:
                    f.write(",".join(CSV_COLUMNS) + "\n")
                f.write(csv_line)
        # Per-problem CSV: one file per instance_id, same schema as global CSV.
        if PER_PROBLEM_CSV_DIR is not None:
            iid = row.get("instance_id")
            if iid:
                pp_path = PER_PROBLEM_CSV_DIR / f"{_safe_filename(str(iid))}.csv"
                new_pp = not pp_path.exists() or pp_path.stat().st_size == 0
                with pp_path.open("a") as f:
                    if new_pp:
                        f.write(",".join(CSV_COLUMNS) + "\n")
                    f.write(csv_line)


def parse_sse_message(buf: bytes) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    """
    Walk the buffered SSE bytes, accumulate content blocks and final usage.

    Returns (final_usage, content_blocks, message_status_count).
    The Anthropic stream emits:
      message_start  (usage.input_tokens populated, output_tokens=1 placeholder)
      content_block_start / delta / stop  (per block; deltas hold text or
        partial_json for tool_use)
      message_delta  (usage.output_tokens final)
      message_stop
    """
    final_usage: dict[str, Any] = {}
    content: list[dict[str, Any]] = []
    n_messages = 0
    # Each SSE event is terminated by a blank line; data: <json>
    for raw in buf.split(b"\n\n"):
        if not raw.strip():
            continue
        data_line = None
        for ln in raw.splitlines():
            if ln.startswith(b"data:"):
                data_line = ln[5:].strip()
                break
        if not data_line:
            continue
        try:
            ev = json.loads(data_line)
        except json.JSONDecodeError:
            continue
        et = ev.get("type")
        if et == "message_start":
            n_messages += 1
            msg = ev.get("message") or {}
            u = msg.get("usage") or {}
            final_usage = {
                "input_tokens": int(u.get("input_tokens") or 0),
                "cache_creation_input_tokens": int(u.get("cache_creation_input_tokens") or 0),
                "cache_read_input_tokens": int(u.get("cache_read_input_tokens") or 0),
                "output_tokens": int(u.get("output_tokens") or 0),
            }
        elif et == "content_block_start":
            block = ev.get("content_block") or {}
            idx = ev.get("index")
            while len(content) <= (idx or 0):
                content.append({})
            content[idx] = {**block, "_text_acc": "", "_json_acc": ""}
        elif et == "content_block_delta":
            idx = ev.get("index") or 0
            d = ev.get("delta") or {}
            if idx >= len(content):
                continue
            if d.get("type") == "text_delta":
                content[idx]["_text_acc"] = content[idx].get("_text_acc", "") + (d.get("text") or "")
            elif d.get("type") == "input_json_delta":
                content[idx]["_json_acc"] = content[idx].get("_json_acc", "") + (d.get("partial_json") or "")
        elif et == "content_block_stop":
            idx = ev.get("index") or 0
            if idx >= len(content):
                continue
            blk = content[idx]
            if blk.get("type") == "text":
                blk["text"] = blk.pop("_text_acc", "")
                blk.pop("_json_acc", None)
            elif blk.get("type") == "tool_use":
                raw_json = blk.pop("_json_acc", "")
                blk.pop("_text_acc", None)
                try:
                    blk["input"] = json.loads(raw_json) if raw_json else {}
                except json.JSONDecodeError:
                    blk["input"] = {"__raw": raw_json}
        elif et == "message_delta":
            d = ev.get("delta") or {}
            u = ev.get("usage") or {}
            # vLLM corrects input_tokens / populates cache_read_input_tokens
            # only on the message_delta. message_start carries the pre-cache
            # placeholder values, so we always prefer message_delta when both
            # are present.
            if u:
                if "input_tokens" in u:
                    final_usage["input_tokens"] = int(u.get("input_tokens") or 0)
                if "output_tokens" in u:
                    final_usage["output_tokens"] = int(u.get("output_tokens") or 0)
                if "cache_creation_input_tokens" in u:
                    final_usage["cache_creation_input_tokens"] = int(
                        u.get("cache_creation_input_tokens") or 0
                    )
                if "cache_read_input_tokens" in u:
                    final_usage["cache_read_input_tokens"] = int(
                        u.get("cache_read_input_tokens") or 0
                    )
            if d.get("stop_reason"):
                final_usage["stop_reason"] = d["stop_reason"]
    return final_usage, content, n_messages


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient(timeout=httpx.Timeout(900.0, connect=10.0))
    try:
        yield
    finally:
        await app.state.client.aclose()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "upstream": UPSTREAM}


@app.post("/v1/messages")
async def messages(request: Request) -> Response:
    body = await request.body()
    upstream_url = f"{UPSTREAM}/v1/messages"
    instance_id = request.headers.get("x-instance-id")
    # Strip accept-encoding so the upstream returns uncompressed bytes — the
    # SSE parser below scans raw bytes for `data:` lines, which silently
    # match nothing if the wire is gzip/br/zstd.
    # Strip hop-by-hop headers + any inbound auth (the proxy injects its own
    # upstream credentials via INJECT_HEADERS below).
    strip = {"host", "content-length", "accept-encoding"}
    if not PASSTHROUGH_AUTH:
        strip |= {"x-api-key", "authorization"}
    fwd_headers = {
        k: v for k, v in request.headers.items() if k.lower() not in strip
    }
    fwd_headers["accept-encoding"] = "identity"
    for k, v in INJECT_HEADERS.items():
        fwd_headers[k] = v

    # Claude requests max_tokens in {20000, 32000} which combined with a
    # 200K+ agent-loop conversation overflows vLLM's strict
    # prompt+max_tokens <= max_model_len check. Clamp to MAX_TOKENS_CAP so
    # we keep ample prompt headroom regardless of context length.
    if MAX_TOKENS_CAP and body:
        try:
            req_obj = json.loads(body)
            mt = int(req_obj.get("max_tokens") or 0)
            msgs = req_obj.get("messages", [])
            sys_obj = req_obj.get("system")
            sys_chars = len(json.dumps(sys_obj)) if sys_obj else 0
            tool_chars = len(json.dumps(req_obj.get("tools") or []))
            msg_chars = len(json.dumps(msgs))
            print(
                f"[proxy] {instance_id or '?'} req: max_tokens={mt} "
                f"messages={len(msgs)} sys_chars={sys_chars} "
                f"tool_chars={tool_chars} msg_chars={msg_chars} "
                f"total_body_chars={len(body)}",
                file=sys.stderr, flush=True,
            )
            # When we see a suspicious giant message, dump it once to disk for inspection.
            if msg_chars > 1_000_000 and not Path("/tmp/proxy_giant_dump.json").exists():
                try:
                    Path("/tmp/proxy_giant_dump.json").write_text(body.decode("utf-8", errors="replace"))
                    print(f"[proxy] DUMPED giant body ({len(body)} chars) to /tmp/proxy_giant_dump.json", file=sys.stderr, flush=True)
                except Exception as e:
                    print(f"[proxy] dump failed: {e!r}", file=sys.stderr, flush=True)
            if mt > MAX_TOKENS_CAP:
                req_obj["max_tokens"] = MAX_TOKENS_CAP
                new_body = json.dumps(req_obj).encode("utf-8")
                body = new_body
                fwd_headers["content-length"] = str(len(new_body))
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            print(f"[proxy] req parse error: {e!r}", file=sys.stderr, flush=True)
    started = time.time()
    client: httpx.AsyncClient = request.app.state.client

    # Sniff streaming intent from the request body (claude defaults to True).
    try:
        is_stream = bool((json.loads(body) or {}).get("stream"))
    except Exception:
        is_stream = False

    if not is_stream:
        upstream_resp = await client.post(upstream_url, content=body, headers=fwd_headers)
        elapsed_ms = int((time.time() - started) * 1000)
        try:
            r = json.loads(upstream_resp.content)
            usage = r.get("usage") or {}
            content = r.get("content") or []
            await write_row({
                "ts": started,
                "instance_id": instance_id,
                "elapsed_ms": elapsed_ms,
                "stream": False,
                "input_tokens": usage.get("input_tokens"),
                "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
                "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "stop_reason": r.get("stop_reason"),
                "category": categorize(content),
                "num_tool_calls": sum(1 for b in content if b.get("type") == "tool_use"),
                "num_text_blocks": sum(1 for b in content if b.get("type") == "text"),
                "model": r.get("model"),
                "message_id": r.get("id"),
            })
        except Exception as e:
            print(f"[proxy] log error: {e!r}", file=sys.stderr)
        resp_headers = {
            k: v for k, v in upstream_resp.headers.items()
            if k.lower() not in ("content-length", "transfer-encoding")
        }
        return Response(
            content=upstream_resp.content,
            status_code=upstream_resp.status_code,
            headers=resp_headers,
            media_type=upstream_resp.headers.get("content-type"),
        )

    # Streaming path: forward bytes immediately while buffering for parse.
    req = client.build_request("POST", upstream_url, content=body, headers=fwd_headers)
    upstream_resp = await client.send(req, stream=True)
    status = upstream_resp.status_code
    media_type = upstream_resp.headers.get("content-type", "text/event-stream")
    fwd_resp_headers = {
        k: v for k, v in upstream_resp.headers.items()
        if k.lower() not in ("content-length", "transfer-encoding")
    }

    async def gen():
        buf = bytearray()
        # TTFT: time from request start → first non-empty upstream byte that
        # carries a streamed content_block_delta (text or tool-arg delta).
        # We approximate by capturing the first chunk that yields a parseable
        # SSE event of those types; if none arrives, we fall back to the
        # first chunk wall-time.
        first_chunk_ts: float | None = None
        first_token_ts: float | None = None
        try:
            async for chunk in upstream_resp.aiter_raw():
                now = time.time()
                if first_chunk_ts is None and chunk:
                    first_chunk_ts = now
                if first_token_ts is None and chunk:
                    # Cheap probe: a content_block_delta event contains either
                    # text_delta or input_json_delta. Don't fully re-parse;
                    # substring search is sufficient and bounded by chunk size.
                    if b"content_block_delta" in chunk:
                        first_token_ts = now
                buf.extend(chunk)
                yield chunk
        finally:
            await upstream_resp.aclose()
            end_ts = time.time()
            elapsed_ms = int((end_ts - started) * 1000)
            ttft_ms = int(((first_token_ts or first_chunk_ts or end_ts) - started) * 1000)
            decode_ms = int((end_ts - (first_token_ts or first_chunk_ts or started)) * 1000)
            try:
                usage, content, _ = parse_sse_message(bytes(buf))
                out_tok = int(usage.get("output_tokens") or 0)
                itl_ms = round(decode_ms / max(1, out_tok - 1), 3) if out_tok > 1 else None
                # Strip internal accumulators before logging.
                clean_content = []
                for b in content:
                    bb = {k: v for k, v in b.items() if not k.startswith("_")}
                    clean_content.append(bb)
                await write_row({
                    "ts": started,
                    "instance_id": instance_id,
                    "elapsed_ms": elapsed_ms,
                    "ttft_ms": ttft_ms,
                    "decode_ms": decode_ms,
                    "itl_ms": itl_ms,
                    "stream": True,
                    "input_tokens": usage.get("input_tokens"),
                    "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
                    "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
                    "output_tokens": usage.get("output_tokens"),
                    "stop_reason": usage.get("stop_reason"),
                    "category": categorize(clean_content),
                    "num_tool_calls": sum(1 for b in clean_content if b.get("type") == "tool_use"),
                    "num_text_blocks": sum(1 for b in clean_content if b.get("type") == "text"),
                })
            except Exception as e:
                print(f"[proxy] stream-log error: {e!r}", file=sys.stderr)

    return StreamingResponse(gen(), status_code=status, headers=fwd_resp_headers, media_type=media_type)


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
)
async def passthrough(path: str, request: Request) -> Response:
    body = await request.body()
    # Strip hop-by-hop headers + any inbound auth (the proxy injects its own
    # upstream credentials via INJECT_HEADERS below).
    strip = {"host", "content-length", "accept-encoding"}
    if not PASSTHROUGH_AUTH:
        strip |= {"x-api-key", "authorization"}
    fwd_headers = {
        k: v for k, v in request.headers.items() if k.lower() not in strip
    }
    fwd_headers["accept-encoding"] = "identity"
    for k, v in INJECT_HEADERS.items():
        fwd_headers[k] = v
    url = f"{UPSTREAM}/{path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"
    client: httpx.AsyncClient = request.app.state.client
    upstream_resp = await client.request(
        request.method, url, content=body, headers=fwd_headers
    )
    resp_headers = {
        k: v for k, v in upstream_resp.headers.items()
        if k.lower() not in ("content-length", "transfer-encoding")
    }
    return Response(
        content=upstream_resp.content,
        status_code=upstream_resp.status_code,
        headers=resp_headers,
        media_type=upstream_resp.headers.get("content-type"),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", required=True, help="e.g. http://localhost:8000")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--usage-log", type=Path, default=None,
                    help="optional JSONL output (one row per /v1/messages call)")
    ap.add_argument("--usage-csv", type=Path, default=None,
                    help="optional CSV mirror of the JSONL output")
    ap.add_argument("--per-problem-csv-dir", type=Path, default=None,
                    help="optional directory; one CSV per instance_id written here")
    ap.add_argument("--max-tokens-cap", type=int, default=4096,
                    help="clamp claude's max_tokens to this; 0 = no clamp")
    ap.add_argument("--inject-header", action="append", default=[],
                    metavar="NAME:VALUE",
                    help="header to inject on every upstream request (repeatable)")
    ap.add_argument("--passthrough-auth", action="store_true",
                    help="forward client's Authorization/x-api-key headers "
                         "unchanged (needed for Anthropic OAuth bearer)")
    args = ap.parse_args()

    global UPSTREAM, USAGE_LOG, USAGE_CSV, PER_PROBLEM_CSV_DIR, MAX_TOKENS_CAP, INJECT_HEADERS, PASSTHROUGH_AUTH
    PASSTHROUGH_AUTH = bool(args.passthrough_auth)
    UPSTREAM = args.upstream.rstrip("/")
    for spec in args.inject_header:
        if ":" not in spec:
            print(f"[proxy] ignoring malformed --inject-header: {spec!r}",
                  file=sys.stderr)
            continue
        name, _, value = spec.partition(":")
        INJECT_HEADERS[name.strip()] = value.strip()
    if args.usage_log is not None:
        USAGE_LOG = args.usage_log
        USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
    if args.usage_csv is not None:
        USAGE_CSV = args.usage_csv
        USAGE_CSV.parent.mkdir(parents=True, exist_ok=True)
    if args.per_problem_csv_dir is not None:
        PER_PROBLEM_CSV_DIR = args.per_problem_csv_dir
        PER_PROBLEM_CSV_DIR.mkdir(parents=True, exist_ok=True)
    MAX_TOKENS_CAP = max(0, args.max_tokens_cap)

    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
