"""Demo: calls send_message(), stream_message(), and call_with_tools()."""

import json
from client import send_message, stream_message, call_with_tools

# ---------------------------------------------------------------------------
# 1. send_message — blocking, returns full Message object
# ---------------------------------------------------------------------------
print("=" * 60)
print("1. send_message()")
print("=" * 60)

response = send_message(
    "Give me 3 concise bullet points on why quantization helps LLM inference.",
    system="You are a helpful AI assistant. Be concise.",
)
for block in response.content:
    if block.type == "text":
        print(block.text)
print(f"\n[stop_reason={response.stop_reason}, "
      f"input_tokens={response.usage.input_tokens}, "
      f"output_tokens={response.usage.output_tokens}]")


# ---------------------------------------------------------------------------
# 2. stream_message — prints tokens as they arrive
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("2. stream_message()")
print("=" * 60)

full_text = stream_message(
    "Write a Python one-liner that reverses a string.",
    system="Reply with only code and a one-sentence explanation.",
)
print(f"[streamed {len(full_text)} chars]")


# ---------------------------------------------------------------------------
# 3. call_with_tools — structured tool use
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("3. call_with_tools()")
print("=" * 60)

TOOLS = [
    {
        "name": "get_weather",
        "description": "Return the current weather for a given city.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g. 'San Francisco'",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit",
                },
            },
            "required": ["city"],
        },
    }
]

tool_response = call_with_tools(
    "What is the weather like in Tokyo right now?",
    tools=TOOLS,
)
for block in tool_response.content:
    if block.type == "tool_use":
        print(f"Tool called : {block.name}")
        print(f"Tool input  : {json.dumps(block.input, indent=2)}")
    elif block.type == "text":
        print(block.text)
print(f"[stop_reason={tool_response.stop_reason}]")
