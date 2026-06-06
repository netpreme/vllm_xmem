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
from pipeline.utils.jsonl import JsonlWriter
from pipeline.proxy.parse import (
    ParsedRequest,
    common_prefix_len,
    parse_request,
    parse_sse_response,
    request_units,
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

    def __init__(
        self,
        upstream: str,
        out_dir: Path,
        instance_id: str,
        *,
        raw: bool = False,
        write_metrics: bool = False,
        inject_token_ids: bool = True,
    ) -> None:
        self.upstream = upstream.rstrip("/")
        self.instance_id = instance_id
        self._writer = JsonlWriter(out_dir, "proxy.jsonl")
        # With raw capture: also tee the raw text traces (isl_new + osl) to
        # raw.jsonl. `_prev_units` is the previous turn's request units, so each
        # turn's new suffix (isl_new as text) is a pure cross-turn string diff.
        self._raw = raw
        self._raw_writer = JsonlWriter(out_dir, "raw.jsonl") if raw else None
        self._prev_units: list[str] = []
        # Remote (Anthropic) backend: there's no vLLM /metrics, so derive
        # isl/osl/isl_new from the response's `usage` and write a vllm.jsonl
        # row ourselves. inject_token_ids is vLLM-only (Anthropic rejects it).
        self._inject_token_ids = inject_token_ids
        self._vllm_writer = JsonlWriter(out_dir, "vllm.jsonl") if write_metrics else None

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
            return await self._messages(client=client, request=request, raw_body=raw_body)
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
        ts = time.time()
        forward_body = raw_body
        body: dict | None = None
        try:
            body = json.loads(raw_body)
            # claude-cli injects role:"system" messages that vLLM 400s on; the
            # top-level `system` (cached prefix) is left untouched.
            rerole_system_messages(body)
            # With raw capture on the vLLM backend, ask for exact token ids
            # (returned in a trailing `vllm_token_ids` event we tee then strip).
            # Skipped for the Anthropic backend, which rejects the unknown field.
            if self._raw and self._inject_token_ids:
                body["return_token_ids"] = True
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

        # Skip Claude Code's session-title request: vLLM short-circuits it
        # before the engine, so recording it would leave a phantom leading row
        # that shifts the positional vllm<->proxy join. Still forwarded.
        if body is not None and _is_title_request(body):
            return Response(
                content=upstream_resp.content,
                status_code=upstream_resp.status_code,
                headers=_strip_hop_by_hop(upstream_resp.headers.items()),
            )

        self._writer.write(
            instance_id=self.instance_id,
            row={
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

        # Remote backend: no vLLM /metrics, so derive isl/osl/isl_new from the
        # Anthropic `usage` and write the vllm.jsonl row ourselves. isl is the
        # full prompt; isl_new is the non-cache-read part (what got processed).
        if self._vllm_writer is not None:
            isl = (
                parsed_resp.input_tokens
                + parsed_resp.cache_read_input_tokens
                + parsed_resp.cache_creation_input_tokens
            )
            self._vllm_writer.write(
                instance_id=self.instance_id,
                row={
                    "ts": round(ts, 3),
                    "isl": isl,
                    "osl": parsed_resp.output_tokens,
                    "isl_new": isl - parsed_resp.cache_read_input_tokens,
                    "prefix_cache_hits": parsed_resp.cache_read_input_tokens,
                    "stop_reason": parsed_resp.claude_stop_reason,
                },
            )

        if self._raw_writer is not None and body is not None:
            self._write_raw(ts=ts, body=body, parsed_resp=parsed_resp)

        # Strip our non-standard token-ids event before returning, so claude-cli
        # only ever sees a standard Anthropic stream.
        content = upstream_resp.content
        if self._raw:
            content = _strip_token_ids_event(content)

        return Response(
            content=content,
            status_code=upstream_resp.status_code,
            headers=_strip_hop_by_hop(upstream_resp.headers.items()),
        )

    def _write_raw(self, ts: float, body: dict, parsed_resp) -> None:
        """Tee the raw text + token-id trace for this turn:
          isl_text     — the full input (system + tools + messages),
          isl_new_text — the input appended since the previous turn (cached
                         prefix stripped off; == isl_text on the first turn),
          osl_text     — the generated assistant text (incl. tool calls),
          isl_ids      — exact full prompt token ids for this turn (input_ids),
          osl_ids      — exact generated output token ids.
        The *_ids fields are present only when vLLM returned them (the trailing
        `vllm_token_ids` event); they are exact, not a re-tokenization. There is
        no isl_new_ids: it is just the tail of isl_ids, derived at analysis time
        as isl_ids[-isl_new:] using the isl_new count from vllm.jsonl.
        """
        units = request_units(body)
        k = common_prefix_len(a=self._prev_units, b=units)
        self._prev_units = units

        assert self._raw_writer is not None
        self._raw_writer.write(
            instance_id=self.instance_id,
            row={
                "ts": round(ts, 3),
                "isl_text": "\n".join(units),
                "isl_new_text": "\n".join(units[k:]),
                "osl_text": parsed_resp.response_text,
                "isl_ids": parsed_resp.prompt_token_ids,
                "osl_ids": parsed_resp.output_token_ids,
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


def _strip_token_ids_event(content: bytes) -> bytes:
    """Remove vLLM's non-standard `vllm_token_ids` SSE record from a buffered
    response so claude-cli only sees a standard Anthropic stream. SSE records
    are separated by blank lines; we drop the one carrying that event."""
    if b"vllm_token_ids" not in content:
        return content
    text = content.decode("utf-8", errors="replace")
    kept = [r for r in text.split("\n\n") if "event: vllm_token_ids" not in r]
    return "\n\n".join(kept).encode()


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
            isinstance(b, dict) and "sentence-case title" in (b.get("text") or "")
            for b in system
        )
    return False


def _strip_hop_by_hop(headers: Iterable[tuple[str, str]]) -> dict[str, str]:
    return {
        name: value
        for name, value in headers
        if name.lower() not in _HOP_BY_HOP_HEADERS
    }
