"""Pure parsing of claude-cli's Anthropic request + SSE response (no I/O).

Derivations (`agent` main/sub, category, …) are NOT done here — they live in
the analysis layer. This just extracts RAW per-turn fields.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Request — pure functions over the request JSON.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParsedRequest:
    # Where the request's input bulk lives. system is text-only; tools and
    # messages are serialized-JSON length (what actually gets tokenized).
    # Together these explain `isl`: input ≈ system + tools + messages.
    system_prompt_chars: int  # text of the top-level `system` field
    tools_chars: int  # serialized `tools` array (the tool definitions)
    messages_chars: int  # serialized `messages` array (problem + history)
    num_tool_defs: int
    num_messages: int


def parse_request(body: dict) -> ParsedRequest:
    tools = body.get("tools") or []
    messages = body.get("messages") or []
    return ParsedRequest(
        system_prompt_chars=len(_system_prompt_text(body)),
        tools_chars=len(json.dumps(tools)) if tools else 0,
        messages_chars=len(json.dumps(messages)) if messages else 0,
        num_tool_defs=len(tools),
        num_messages=len(messages),
    )


def rerole_system_messages(body: dict) -> int:
    """Re-role any ``role:"system"`` entry in ``messages[]`` to ``"user"``.

    claude-cli (2.1.15x, Skills feature) injects its "available skills" notice
    as a ``role:"system"`` message *inside* ``messages[]``. The Anthropic
    Messages spec only allows ``user``/``assistant`` there — system text belongs
    in the top-level ``system`` field — and vLLM's /v1/messages enforces this
    with a 400, which aborts the whole claude session. Re-roling to ``user``
    (content untouched, in place) makes it validate while leaving the top-level
    ``system`` — the cached prefix we measure — alone. Returns the count moved.
    """
    moved = 0
    for message in body.get("messages") or []:
        if isinstance(message, dict) and message.get("role") == "system":
            message["role"] = "user"
            moved += 1
    return moved


# ---------------------------------------------------------------------------
# SSE response — walk the stream.
#
# Anthropic's streaming format:
#     event: message_start         data: {... usage, model, ...}
#     event: content_block_start   data: {type, content_block: {type, ...}}
#     event: content_block_delta   data: {type, delta: {type, text | partial_json}}
#     event: content_block_stop    data: {...}
#     event: message_delta         data: {delta: {stop_reason, ...}, usage}
#     event: message_stop          data: {...}
# ---------------------------------------------------------------------------


@dataclass
class ParsedResponse:
    num_tool_calls: int = 0
    tool_names: list[str] = field(default_factory=list)
    has_thinking: bool = False
    response_text_chars: int = 0
    claude_stop_reason: str = ""


def parse_sse_response(body: bytes) -> ParsedResponse:
    parsed = ParsedResponse()
    text = body.decode("utf-8", errors="replace")
    for record in text.split("\n\n"):
        data = ""
        for line in record.splitlines():
            if line.startswith("data:"):
                data = line[5:].strip()
                break
        if not data or data == "[DONE]":
            continue
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue
        event_type = event.get("type")
        if event_type == "content_block_start":
            block = event.get("content_block") or {}
            block_type = block.get("type")
            if block_type == "tool_use":
                parsed.num_tool_calls += 1
                if block.get("name"):
                    parsed.tool_names.append(block["name"])
            elif block_type == "thinking":
                parsed.has_thinking = True
        elif event_type == "content_block_delta":
            delta = event.get("delta") or {}
            if delta.get("type") == "text_delta":
                parsed.response_text_chars += len(delta.get("text") or "")
        elif event_type == "message_delta":
            stop_reason = (event.get("delta") or {}).get("stop_reason")
            if stop_reason:
                parsed.claude_stop_reason = stop_reason
    return parsed


def _system_prompt_text(body: dict) -> str:
    """Anthropic accepts `system` as either str or a list of content blocks."""
    system = body.get("system")
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        return "".join(
            block.get("text") or ""
            for block in system
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""
