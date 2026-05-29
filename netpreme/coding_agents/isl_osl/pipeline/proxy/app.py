"""The reverse-proxy ASGI app: forward claude-cli ↔ vLLM, tee per-turn rows.

``ProxyApp(upstream, out_dir, instance_id).build()`` returns a Starlette app
that, for each `POST /v1/messages`, parses the JSON request + SSE response and
appends one RAW row to `<out_dir>/<instance_id>/proxy.jsonl`; other paths
stream through unchanged (e.g. /v1/models).
"""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Iterable

import httpx
from loguru import logger
from pipeline.jsonl import JsonlWriter
from pipeline.proxy.parse import (
    ParsedRequest,
    parse_request,
    parse_sse_response,
    rerole_system_messages,
)
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route

_HOP_BY_HOP_HEADERS: frozenset[str] = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
    }
)


class ProxyApp:
    """Reverse-proxy ASGI app for one problem (claude-cli ↔ vLLM)."""

    def __init__(self, upstream: str, out_dir: Path, instance_id: str) -> None:
        self.upstream = upstream.rstrip("/")
        self.instance_id = instance_id
        self._writer = JsonlWriter(out_dir, "proxy.jsonl")

    def build(self) -> Starlette:
        return Starlette(
            routes=[
                Route(
                    "/{path:path}",
                    endpoint=self._handle,
                    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD"],
                )
            ],
            lifespan=self._lifespan,
        )

    @asynccontextmanager
    async def _lifespan(self, app: Starlette):
        async with httpx.AsyncClient(timeout=None) as client:
            app.state.client = client
            yield

    def _target(self, request: Request) -> str:
        target = f"{self.upstream}{request.url.path}"
        if request.url.query:
            target = f"{target}?{request.url.query}"
        return target

    async def _handle(self, request: Request) -> Response:
        raw_body = await request.body()
        client = request.app.state.client
        if request.method == "POST" and request.url.path == "/v1/messages":
            return await self._messages(client, request, raw_body)
        return await self._passthrough(client, request, raw_body)

    async def _messages(
        self, client: httpx.AsyncClient, request: Request, raw_body: bytes
    ) -> Response:
        """Forward a /v1/messages POST; buffer the response so we can parse
        it; write the combined request+response row; return the buffered
        response to claude-cli.

        Buffering breaks "live" streaming to claude-cli, but at concurrency=1
        that's invisible — claude still parses the SSE chunks the same way."""
        ts = time.time()
        forward_body = raw_body
        try:
            body = json.loads(raw_body)
            # claude-cli injects role:"system" messages that vLLM 400s on; the
            # top-level `system` (cached prefix) is left untouched.
            rerole_system_messages(body)
            forward_body = json.dumps(body).encode()
            parsed_req = parse_request(body)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("request parse error: {!r}", exc)
            parsed_req = ParsedRequest(
                system_prompt_chars=0,
                tools_chars=0,
                messages_chars=0,
                num_tool_defs=0,
                num_messages=0,
            )

        upstream_resp = await client.request(
            request.method,
            self._target(request),
            content=forward_body,
            headers=_strip_hop_by_hop(request.headers.items()),
        )

        parsed_resp = parse_sse_response(upstream_resp.content)
        self._writer.write(
            self.instance_id,
            {
                "ts": round(ts, 3),
                "system_prompt_chars": parsed_req.system_prompt_chars,
                "tools_chars": parsed_req.tools_chars,
                "messages_chars": parsed_req.messages_chars,
                "num_tool_defs": parsed_req.num_tool_defs,
                "num_messages": parsed_req.num_messages,
                "num_tool_calls": parsed_resp.num_tool_calls,
                "tool_names": parsed_resp.tool_names,
                "has_thinking": parsed_resp.has_thinking,
                "response_text_chars": parsed_resp.response_text_chars,
                "claude_stop_reason": parsed_resp.claude_stop_reason,
            },
        )

        return Response(
            content=upstream_resp.content,
            status_code=upstream_resp.status_code,
            headers=_strip_hop_by_hop(upstream_resp.headers.items()),
        )

    async def _passthrough(
        self, client: httpx.AsyncClient, request: Request, raw_body: bytes
    ) -> Response:
        """Stream non-/v1/messages requests through unchanged (e.g. /v1/models)."""
        upstream_req = client.build_request(
            request.method,
            self._target(request),
            content=raw_body,
            headers=_strip_hop_by_hop(request.headers.items()),
        )
        upstream_resp = await client.send(upstream_req, stream=True)
        return StreamingResponse(
            upstream_resp.aiter_raw(),
            status_code=upstream_resp.status_code,
            headers=_strip_hop_by_hop(upstream_resp.headers.items()),
            background=BackgroundTask(upstream_resp.aclose),
        )


def _strip_hop_by_hop(headers: Iterable[tuple[str, str]]) -> dict[str, str]:
    return {
        name: value
        for name, value in headers
        if name.lower() not in _HOP_BY_HOP_HEADERS
    }
