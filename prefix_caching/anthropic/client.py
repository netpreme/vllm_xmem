"""Anthropic SDK client pointed at a local vLLM server."""

import os
import time
import anthropic

_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "http://localhost:8000")
_API_KEY = os.getenv("ANTHROPIC_API_KEY", "dummy")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-Coder-32B-Instruct")

_MAX_RETRIES = 4
_RETRY_BASE_DELAY = 1.0  # seconds


def _make_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(base_url=_BASE_URL, api_key=_API_KEY)


def _with_backoff(fn, *args, **kwargs):
    """Call fn(*args, **kwargs) with exponential backoff on connection errors."""
    delay = _RETRY_BASE_DELAY
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except anthropic.APIConnectionError as exc:
            if attempt == _MAX_RETRIES:
                raise
            print(f"  [retry {attempt}/{_MAX_RETRIES}] connection error: {exc}; "
                  f"retrying in {delay:.1f}s ...")
            time.sleep(delay)
            delay *= 2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def send_message(
    prompt: str,
    *,
    system: str | None = None,
    max_tokens: int = 1024,
    model: str = MODEL_NAME,
) -> anthropic.types.Message:
    """Send a single user message and return the full Message response."""
    client = _make_client()
    kwargs: dict = dict(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    if system:
        kwargs["system"] = system
    return _with_backoff(client.messages.create, **kwargs)


def stream_message(
    prompt: str,
    *,
    system: str | None = None,
    max_tokens: int = 1024,
    model: str = MODEL_NAME,
) -> str:
    """Stream a response and return the accumulated text."""
    client = _make_client()
    kwargs: dict = dict(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    if system:
        kwargs["system"] = system

    def _stream():
        accumulated = []
        with client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
                accumulated.append(text)
        print()  # newline after stream ends
        return "".join(accumulated)

    return _with_backoff(_stream)


def call_with_tools(
    prompt: str,
    tools: list[dict],
    *,
    max_tokens: int = 1024,
    model: str = MODEL_NAME,
) -> anthropic.types.Message:
    """Send a message with tool definitions; return the raw Message."""
    client = _make_client()
    return _with_backoff(
        client.messages.create,
        model=model,
        max_tokens=max_tokens,
        tools=tools,
        messages=[{"role": "user", "content": prompt}],
    )
