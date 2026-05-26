"""Thin reverse-proxy that sits between claude-cli and vLLM.

Purpose: capture the one thing vLLM cannot tell us from its Prometheus
metrics — whether each /v1/messages call is claude's outer agent loop or
a Task-tool sub-agent. Everything else (timings, tokens, KV stats) stays
the watcher's job.

The proxy is deliberately minimal. It does three things on each request:

    1. Classify it (main vs sub) by inspecting the system-prompt size.
    2. Append a label record to `<run_dir>/.agent_labels`.
    3. Forward the request to vLLM unmodified and stream the response back.

It owns no timing or token data; the watcher reads those from vLLM's
/metrics directly. Steps 1-2 are pure / side-effect-only and live in the
"Classification" section. Step 3 is pure HTTP transport and lives in the
"Reverse proxy" section. The two are wired together only inside the
Starlette route handler at the bottom.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route

from agent_labels import LabelRecord, LabelWriter


# ===========================================================================
# Classification — pure functions, no I/O.
# ===========================================================================
#
# Only /v1/messages requests are classified. Two pieces of information come
# out of a request body: a parsed view (numbers we want to log) and a final
# agent label (main vs sub). These are split because the parsed view is
# also useful for the label record's payload, while the decision logic is
# tiny enough to keep in one obvious place.


@dataclass(frozen=True)
class ParsedRequest:
    """Just the fields we extract from a /v1/messages body. Keeping this
    as a dataclass makes the labeler's downstream code self-documenting:
    you can see at a glance what we look at and what we don't."""

    system_prompt_chars: int
    num_tool_defs:       int
    num_messages:        int


def _system_prompt_text(body: dict) -> str:
    """Anthropic's API accepts `system` as either a string or a list of
    content blocks. Collapse both into a single plaintext string so we
    can measure its length uniformly."""
    system = body.get("system")
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts: list[str] = []
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text") or "")
        return "".join(parts)
    return ""


def parse_request(body: dict) -> ParsedRequest:
    """Extract the three signals we care about. Pure: no I/O, no
    classification, no side effects."""
    return ParsedRequest(
        system_prompt_chars=len(_system_prompt_text(body)),
        num_tool_defs=len(body.get("tools")  or []),
        num_messages=len(body.get("messages") or []),
    )


def classify_agent(parsed: ParsedRequest, sub_threshold_chars: int) -> str:
    """Return "main" or "sub". The decision is a single threshold on
    system-prompt size: claude-cli's outer loop ships ~27 k chars of
    system prompt (tool schemas + agent instructions), while a Task-tool
    sub-agent gets a stripped-down prompt around 3 k. A 10 k cutoff
    lands cleanly between the two clusters."""
    return "sub" if parsed.system_prompt_chars < sub_threshold_chars else "main"


# ===========================================================================
# Reverse proxy — pure HTTP transport.
# ===========================================================================
#
# These helpers know nothing about agents or labels. They just forward
# bytes between claude-cli and vLLM with streaming preserved.


# Hop-by-hop headers that must not be relayed (per RFC 7230 §6.1).
# Without filtering these, httpx will refuse to set Content-Length and
# upstream will hang on long streams.
_HOP_BY_HOP_HEADERS: frozenset[str] = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
    "host", "content-length",
})


def _strip_hop_by_hop(headers: Iterable[tuple[str, str]]) -> dict[str, str]:
    return {k: v for k, v in headers if k.lower() not in _HOP_BY_HOP_HEADERS}


async def forward(
    client:    httpx.AsyncClient,
    upstream:  str,
    request:   Request,
    raw_body:  bytes,
) -> Response:
    """Forward `request` to `upstream`, streaming the response back to the
    caller. The body is already buffered (we needed to inspect it for
    classification); the response side stays streaming so claude-cli sees
    SSE chunks as they arrive."""
    method  = request.method
    path    = request.url.path
    query   = request.url.query
    headers = _strip_hop_by_hop(request.headers.items())

    target = f"{upstream.rstrip('/')}{path}"
    if query:
        target = f"{target}?{query}"

    upstream_req  = client.build_request(method, target,
                                         content=raw_body, headers=headers)
    upstream_resp = await client.send(upstream_req, stream=True)

    # Pass through headers but drop hop-by-hop and any Content-Length that
    # would lie about the streamed size.
    out_headers = {
        k: v for k, v in upstream_resp.headers.items()
        if k.lower() not in _HOP_BY_HOP_HEADERS
    }

    return StreamingResponse(
        upstream_resp.aiter_raw(),
        status_code=upstream_resp.status_code,
        headers=out_headers,
        background=BackgroundTask(upstream_resp.aclose),
    )


# ===========================================================================
# Application wiring — composes classification + forwarding.
# ===========================================================================


def build_app(
    upstream:             str,
    labels_path:          Path,
    sub_threshold_chars:  int,
) -> Starlette:
    """Construct the Starlette app. The httpx client is created in the
    lifespan handler so we share one connection pool across the entire
    run; configuration is stashed on `app.state` for the route handler."""
    writer = LabelWriter(labels_path)

    @asynccontextmanager
    async def lifespan(app: Starlette):
        async with httpx.AsyncClient(timeout=None) as client:
            app.state.client = client
            yield

    async def handler(request: Request) -> Response:
        # Buffer the request body once. We need it for both classification
        # and forwarding; the underlying stream is single-use.
        raw_body = await request.body()

        if request.method == "POST" and request.url.path == "/v1/messages":
            try:
                parsed = parse_request(json.loads(raw_body))
                writer.write(LabelRecord(
                    agent=classify_agent(parsed, sub_threshold_chars),
                    num_tool_defs=parsed.num_tool_defs,
                    num_messages=parsed.num_messages,
                    system_prompt_chars=parsed.system_prompt_chars,
                ))
            except (json.JSONDecodeError, ValueError) as exc:
                # Don't ever block the forward path on a parser bug —
                # log and continue. The row gets a "main" fallback.
                print(f"[agent-labeler] parse error: {exc!r}",
                      file=sys.stderr, flush=True)

        return await forward(request.app.state.client, upstream,
                             request, raw_body)

    return Starlette(
        routes=[Route("/{path:path}", endpoint=handler,
                      methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD"])],
        lifespan=lifespan,
    )


# ===========================================================================
# CLI.
# ===========================================================================


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream",       required=True,
                        help="vLLM base URL, e.g. http://localhost:8000")
    parser.add_argument("--labels-file",    required=True, type=Path,
                        help="append target for label records")
    parser.add_argument("--listen-host",    default="127.0.0.1")
    parser.add_argument("--listen-port",    type=int, default=8001)
    parser.add_argument("--sub-threshold-chars", type=int, default=10_000,
                        help="system-prompt size below which a request is "
                             "classified as a Task-tool sub-agent (default "
                             "10k — main is ~27k, sub is ~3k)")
    args = parser.parse_args()

    app = build_app(args.upstream, args.labels_file, args.sub_threshold_chars)
    config = uvicorn.Config(
        app,
        host=args.listen_host,
        port=args.listen_port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    try:
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
