#!/usr/bin/env python3
"""Streaming HTTP proxy that tees every request+response to a JSONL trace file.

Spawned by bench_concurrent_users.py as a per-claude-session subprocess.
Listens on --port, forwards every request to --upstream, streams the
response back chunk-by-chunk while collecting the full body for the trace.

Trace entry per request (one JSON line):
    {
        "t_request":      relative seconds when request body fully received,
        "t_response_end": relative seconds when response stream completed,
        "method":         "POST",
        "path":           "/v1/messages",
        "status":         200,
        "request":        parsed JSON request body (or raw_request_b64 if not JSON),
        "response_sse":   raw response body as utf-8 string (SSE if streamed),
        "output_tokens":  parsed from final usage event in SSE (or None),
    }

The first request received marks t=0; all subsequent timings are relative.
A single `meta` line is written first with session_id and upstream.
"""
import argparse
import asyncio
import base64
import json
import re
import signal
import sys
import time
from pathlib import Path

import aiohttp
from aiohttp import web


# headers that must not be forwarded verbatim (hop-by-hop / framing)
_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "content-length",
    "host",
}


def _strip_hbh(headers) -> dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in _HOP_BY_HOP}


# SSE usage extraction: vLLM emits `event: message_delta` with
# data: {"type":"message_delta","delta":{...},"usage":{"input_tokens":N,"output_tokens":M}}
# Final message_delta carries the cumulative counts. Capture both ISL and OSL.
_OUT_RE = re.compile(rb'"output_tokens"\s*:\s*(\d+)')
_IN_RE  = re.compile(rb'"input_tokens"\s*:\s*(\d+)')


def _extract_output_tokens(body: bytes) -> int | None:
    # Find the LAST occurrence (final usage report).
    last = None
    for m in _OUT_RE.finditer(body):
        last = m
    if last is None:
        return None
    try:
        return int(last.group(1))
    except (ValueError, IndexError):
        return None


def _extract_input_tokens(body: bytes) -> int | None:
    """ISL — present in the FIRST message_start usage block (and unchanged
    through the stream). Use the first occurrence."""
    m = _IN_RE.search(body)
    if m is None:
        return None
    try:
        return int(m.group(1))
    except (ValueError, IndexError):
        return None


class TraceWriter:
    """Append-only JSONL writer. Buffered, fsync on close."""

    def __init__(self, path: Path, session_id: str, upstream: str):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "ab", buffering=0)
        self._lock = asyncio.Lock()
        self._t0: float | None = None
        meta = {
            "kind": "meta",
            "session_id": session_id,
            "upstream": upstream,
            "t_wall_start": time.time(),
        }
        self.f.write((json.dumps(meta) + "\n").encode("utf-8"))

    def t(self) -> float:
        now = time.monotonic()
        if self._t0 is None:
            self._t0 = now
        return now - self._t0

    async def write(self, entry: dict) -> None:
        async with self._lock:
            self.f.write((json.dumps(entry) + "\n").encode("utf-8"))

    def close(self) -> None:
        try:
            self.f.flush()
            self.f.close()
        except Exception:
            pass


async def make_app(args) -> web.Application:
    trace = TraceWriter(Path(args.trace), args.session_id, args.upstream)
    # Re-use one ClientSession; long-lived HTTP/1.1 keepalive to upstream.
    timeout = aiohttp.ClientTimeout(total=None, sock_read=None, sock_connect=10)
    connector = aiohttp.TCPConnector(limit=64, force_close=False)
    upstream_session = aiohttp.ClientSession(timeout=timeout, connector=connector)

    async def on_shutdown(app):
        try:
            await upstream_session.close()
        finally:
            trace.close()

    async def proxy(request: web.Request) -> web.StreamResponse:
        t_req = trace.t()
        body_bytes = await request.read()
        try:
            req_json = json.loads(body_bytes) if body_bytes else None
            req_b64 = None
        except Exception:
            req_json = None
            req_b64 = base64.b64encode(body_bytes).decode("ascii")

        upstream_url = args.upstream.rstrip("/") + request.rel_url.path_qs
        try:
            up_resp = await upstream_session.request(
                request.method,
                upstream_url,
                data=body_bytes,
                headers=_strip_hbh(request.headers),
                allow_redirects=False,
            )
        except Exception as e:
            entry = {
                "kind": "error",
                "t_request": t_req,
                "t_response_end": trace.t(),
                "method": request.method,
                "path": request.path,
                "request": req_json,
                "raw_request_b64": req_b64,
                "error": f"{type(e).__name__}: {e}",
            }
            await trace.write(entry)
            return web.Response(status=502, text=f"upstream error: {e}")

        try:
            client_resp = web.StreamResponse(
                status=up_resp.status,
                headers=_strip_hbh(up_resp.headers),
            )
            await client_resp.prepare(request)

            collected = bytearray()
            async for chunk in up_resp.content.iter_any():
                collected.extend(chunk)
                await client_resp.write(chunk)
            await client_resp.write_eof()
        finally:
            up_resp.release()

        t_end = trace.t()
        response_text = bytes(collected).decode("utf-8", errors="replace")
        output_tokens = _extract_output_tokens(bytes(collected))
        input_tokens  = _extract_input_tokens(bytes(collected))

        entry = {
            "kind": "turn",
            "t_request": round(t_req, 4),
            "t_response_end": round(t_end, 4),
            "method": request.method,
            "path": request.path,
            "status": up_resp.status,
            "request": req_json,
            "response_sse": response_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        if req_b64 is not None:
            entry["raw_request_b64"] = req_b64
        await trace.write(entry)
        return client_resp

    async def health(request: web.Request) -> web.Response:
        return web.Response(text="ok")

    app = web.Application(client_max_size=1024 * 1024 * 256)  # 256 MiB
    app.router.add_get("/_proxy_health", health)
    # Catch-all forward for everything else.
    app.router.add_route("*", "/{path:.*}", proxy)
    app.on_shutdown.append(on_shutdown)
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True,
                    help="Local TCP port to listen on")
    ap.add_argument("--upstream", required=True,
                    help="Upstream vLLM base URL, e.g. http://localhost:8001")
    ap.add_argument("--trace", required=True,
                    help="Path to JSONL trace file (created/truncated)")
    ap.add_argument("--session-id", default="",
                    help="Logical session id (e.g. SWE-bench instance_id)")
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Truncate trace before opening (overwrite stale data).
    Path(args.trace).parent.mkdir(parents=True, exist_ok=True)
    Path(args.trace).write_bytes(b"")

    app = loop.run_until_complete(make_app(args))
    runner = web.AppRunner(app, access_log=None)
    loop.run_until_complete(runner.setup())
    site = web.TCPSite(runner, args.host, args.port, reuse_address=True,
                       reuse_port=True)
    try:
        loop.run_until_complete(site.start())
    except OSError as e:
        print(f"capture_proxy: bind {args.host}:{args.port} failed: {e}",
              file=sys.stderr)
        sys.exit(2)

    # Print a single line on ready so the parent can read & gate on it.
    print(f"capture_proxy ready port={args.port} pid={__import__('os').getpid()} "
          f"trace={args.trace}", flush=True)

    stop_event = asyncio.Event()

    def _stop(*_):
        loop.call_soon_threadsafe(stop_event.set)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _stop)
        except NotImplementedError:
            signal.signal(sig, _stop)

    try:
        loop.run_until_complete(stop_event.wait())
    finally:
        loop.run_until_complete(runner.cleanup())
        loop.close()


if __name__ == "__main__":
    main()
