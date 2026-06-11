"""The reverse-proxy ASGI app: forward claude-cli ↔ vLLM, tee per-turn rows.

``ProxyApp(upstream, out_dir, instance_id, raw=True).build()`` returns a
Starlette app that, for each `POST /v1/messages`, sanitizes the request and
(with raw capture) tees the per-turn text trace to
`<out_dir>/<instance_id>/vllm_traces.jsonl`; other paths stream through
unchanged (e.g. /v1/models).
"""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import httpx
from loguru import logger
from pipeline.proxy.parse import (
    common_prefix_len,
    parse_response_text,
    request_units,
    rerole_system_messages,
)
from pipeline.utils.jsonl import JsonlWriter
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


@dataclass(frozen=True)
class MessageRequest:
    """Sanitized request state for one /v1/messages call."""

    body: dict | None
    forward_body: bytes


class ProxyApp:
    """Reverse-proxy ASGI app for one problem (claude-cli ↔ vLLM)."""

    def __init__(
        self,
        upstream: str,
        out_dir: Path,
        instance_id: str,
        *,
        raw: bool = False,
    ) -> None:
        self.upstream = upstream.rstrip("/")
        self.instance_id = instance_id
        # With raw capture: tee the raw text traces (isl_new + osl) to
        # vllm_traces.jsonl. `_prev_units` is the previous turn's request units, so each
        # turn's new suffix (isl_new as text) is a pure cross-turn string diff.
        self._raw_writer = JsonlWriter(out_dir, "vllm_traces.jsonl") if raw else None
        self._prev_units: list[str] = []

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
            return await self._messages(
                client=client, request=request, raw_body=raw_body
            )
        return await self._passthrough(
            client=client, request=request, raw_body=raw_body
        )

    async def _messages(
        self, client: httpx.AsyncClient, request: Request, raw_body: bytes
    ) -> Response:
        """Forward a /v1/messages POST; buffer the response so we can parse
        it; write the combined request+response row; return the buffered
        response to claude-cli.

        Buffering breaks "live" streaming to claude-cli, but at concurrency=1
        that's invisible — claude still parses the SSE chunks the same way."""
        timestamp = time.time()
        message_request = self._prepare_message_request(raw_body)

        upstream_response = await self._forward_buffered(
            client=client,
            request=request,
            body=message_request.forward_body,
        )

        if (
            self._raw_writer is not None
            and message_request.body is not None
            and not _should_skip_telemetry(message_request.body)
        ):
            self._write_raw(
                timestamp=timestamp,
                body=message_request.body,
                osl_text=parse_response_text(upstream_response.content),
            )

        return _buffered_response(upstream_response)

    def _prepare_message_request(self, raw_body: bytes) -> MessageRequest:
        try:
            body = json.loads(raw_body)
            if not isinstance(body, dict):
                raise ValueError("/v1/messages body must be a JSON object")
            # claude-cli injects role:"system" messages that vLLM 400s on; the
            # top-level `system` (cached prefix) is left untouched.
            rerole_system_messages(body)
            return MessageRequest(
                body=body,
                forward_body=json.dumps(body).encode(),
            )
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("request parse error: {!r}", exc)
            return MessageRequest(body=None, forward_body=raw_body)

    async def _forward_buffered(
        self,
        client: httpx.AsyncClient,
        request: Request,
        body: bytes,
    ) -> httpx.Response:
        return await client.request(
            request.method,
            self._target(request),
            content=body,
            headers=_strip_hop_by_hop(request.headers.items()),
        )

    def _write_raw(self, timestamp: float, body: dict, osl_text: str) -> None:
        """Tee the raw text for this turn:
        isl_text     — the full input (system + tools + messages),
        isl_new_text — the input appended since the previous turn (cached
                       prefix stripped off; == isl_text on the first turn),
        osl_text     — the generated assistant text (incl. tool calls).
        """
        units = request_units(body)
        prefix_length = common_prefix_len(
            previous_units=self._prev_units,
            current_units=units,
        )
        self._prev_units = units

        assert self._raw_writer is not None
        self._raw_writer.write(
            instance_id=self.instance_id,
            row={
                "ts": round(timestamp, 3),
                "isl_text": "\n".join(units),
                "isl_new_text": "\n".join(units[prefix_length:]),
                "osl_text": osl_text,
            },
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


def _should_skip_telemetry(body: dict | None) -> bool:
    """Skip requests that vLLM handles outside the normal engine path."""
    if body is None:
        return False
    return _is_title_request(body)


def _is_title_request(body: dict) -> bool:
    """True for Claude Code's session-title request.

    Mirrors the signal vLLM's anthropic entrypoint uses to short-circuit it
    (vllm/entrypoints/anthropic/serving.py): a "generate a … sentence-case
    title" instruction in the top-level system prompt. `system` may be a
    string or a list of `{type, text}` blocks.
    """
    system = body.get("system")
    if isinstance(system, str):
        return "sentence-case title" in system
    if isinstance(system, list):
        return any(
            isinstance(block, dict)
            and "sentence-case title" in (block.get("text") or "")
            for block in system
        )
    return False


def _buffered_response(
    upstream_response: httpx.Response,
    content: bytes | None = None,
) -> Response:
    if content is None:
        content = upstream_response.content
    return Response(
        content=content,
        status_code=upstream_response.status_code,
        headers=_strip_hop_by_hop(upstream_response.headers.items()),
    )


def _strip_hop_by_hop(headers: Iterable[tuple[str, str]]) -> dict[str, str]:
    return {
        name: value
        for name, value in headers
        if name.lower() not in _HOP_BY_HOP_HEADERS
    }
