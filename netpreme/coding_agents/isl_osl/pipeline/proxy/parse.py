"""Pure parsing of claude-cli's Anthropic request + SSE response (no I/O).

Derivations (`agent` main/sub, category, …) are NOT done here — they live in
the analysis layer. This just extracts RAW per-turn fields.
"""

from __future__ import annotations

import json
import re


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


# Anthropic SSE stream:
#   content_block_start  {content_block: {type, ...}}
#   content_block_delta  {delta: {type, text | partial_json | thinking}}


# claude-cli prepends a per-request billing header to the system prompt, e.g.
# `x-anthropic-billing-header: cc_version=…; cc_entrypoint=sdk-cli; cch=04bbe;`.
# The `cch` nonce (and cc_version) change every turn; left in, they make the
# whole system block — and thus every later unit — look "new", defeating the
# cross-turn diff. vLLM's own prefix cache still hits ~98% despite it, so it's
# pure noise for our purposes. Strip the preamble through the `cch=…;` token.
_BILLING_RE = re.compile(r"^x-anthropic-billing-header:.*?cch=[^;]*;", re.S)


def _strip_volatile_system(text: str) -> str:
    return _BILLING_RE.sub("", text, count=1)


def request_units(body: dict) -> list[str]:
    """Ordered serialized 'units' of the request input, for prefix-diffing
    across turns. The conversation is append-only, so turn N's units share a
    common prefix with turn N-1's; the new suffix is ``isl_new`` as raw text.

    Units: the (static) system text, then the (static) tools blob, then one
    per message — so on later turns only freshly-appended messages differ."""
    units: list[str] = []
    system = _strip_volatile_system(_system_prompt_text(body))
    if system:
        units.append("SYSTEM:" + system)
    tools = body.get("tools") or []
    if tools:
        units.append("TOOLS:" + json.dumps(tools))
    for message in body.get("messages") or []:
        units.append("MSG:" + json.dumps(message))
    return units


def common_prefix_len(previous_units: list[str], current_units: list[str]) -> int:
    """Length of the shared leading run of two unit lists."""
    prefix_length = 0
    for previous_unit, current_unit in zip(previous_units, current_units):
        if previous_unit != current_unit:
            break
        prefix_length += 1
    return prefix_length


def parse_response_text(body: bytes) -> str:
    """The generated assistant text for one turn (osl as text): visible text,
    streamed tool-call args, reasoning, and a marker per tool call — tool calls
    ARE output tokens, so they're included."""
    parts: list[str] = []
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
            if block.get("type") == "tool_use":
                parts.append(f"\n[tool_use:{block.get('name') or ''}] ")
        elif event_type == "content_block_delta":
            delta = event.get("delta") or {}
            delta_type = delta.get("type")
            if delta_type == "text_delta":
                parts.append(delta.get("text") or "")
            elif delta_type == "input_json_delta":
                parts.append(delta.get("partial_json") or "")
            elif delta_type == "thinking_delta":
                parts.append(delta.get("thinking") or "")
    return "".join(parts)


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
