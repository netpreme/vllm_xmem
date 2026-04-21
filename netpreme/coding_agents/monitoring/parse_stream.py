#!/usr/bin/env python3
"""
Parse claude --output-format stream-json from stdin.

Writes per-turn tool call data to /tmp/vllm_tool_calls.json (read by watch_vllm.py).
Prints a flat table — one row per vLLM turn and one row per tool call/result.

Usage (via run_claude.sh --auto — not meant to be called directly):
    claude ... --output-format stream-json -p "task" | python3 parse_stream.py

Columns:
    Type  — "turn" (vLLM request) or "tool" (tool call + result)
    ISL   — turn: input tokens sent to vLLM  |  tool: input arg length (chars)
    OSL   — turn: output tokens from vLLM    |  tool: result length (chars)
    Name  — turn: "-"                        |  tool: tool name
    Detail — turn: first text snippet        |  tool: key input argument
"""

import json
import sys
from pathlib import Path

TOOL_CALLS_FILE = Path("/tmp/vllm_tool_calls.json")

COL    = "{:<5}  {:>9}  {:>9}  {:<14}  {}"
HDR    = COL.format("Type", "ISL(in)", "OSL(out)", "Name", "Detail")
SEP    = "─" * len(HDR)
TSEP   = "┄" * len(HDR)         # lighter separator between turns


def fmt_n(n) -> str:
    return f"{int(n):,}" if n is not None else "—"


def first_text(content: list) -> str:
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            t = block.get("text", "").replace("\n", " ").strip()
            if t:
                return t[:80]
    return ""


def fmt_input(name: str, inp: dict) -> str:
    key_map = {
        "Read":      "file_path",
        "Write":     "file_path",
        "Edit":      "file_path",
        "Bash":      "command",
        "Grep":      "pattern",
        "Glob":      "pattern",
        "Agent":     "description",
        "WebFetch":  "url",
        "WebSearch": "query",
    }
    key = key_map.get(name)
    if key and key in inp:
        val = str(inp[key]).replace("\n", " ").strip()
        return f"{key}={val[:70]}"
    if inp:
        k, v = next(iter(inp.items()))
        return f"{k}={str(v).replace(chr(10), ' ').strip()[:70]}"
    return ""


def result_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict) and c.get("type") == "text":
                parts.append(c.get("text", ""))
            elif isinstance(c, str):
                parts.append(c)
        return " ".join(parts)
    return str(content)


def main():
    TOOL_CALLS_FILE.write_text("[]")

    sidecar: list[dict] = []
    turn_num       = 0
    pending_tools: dict[str, dict] = {}   # tool_use_id → {name, input}

    print(SEP)
    print(HDR)
    print(f"  (ISL/OSL = tokens for 'turn' rows, chars for 'tool' rows)")
    print(SEP)

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        etype = event.get("type")

        # ── vLLM request completed ────────────────────────────────────────────
        if etype == "assistant":
            msg     = event.get("message", {})
            usage   = msg.get("usage", {})
            content = msg.get("content", [])

            isl        = usage.get("input_tokens")
            osl        = usage.get("output_tokens")
            tool_uses  = [b for b in content
                          if isinstance(b, dict) and b.get("type") == "tool_use"]
            tool_names = [b.get("name", "") for b in tool_uses]
            snippet    = first_text(content)

            turn_num += 1

            for b in tool_uses:
                pending_tools[b.get("id", "")] = {
                    "name":  b.get("name", "?"),
                    "input": b.get("input", {}),
                }

            sidecar.append({"turn": turn_num, "tool_calls": len(tool_uses), "tool_names": tool_names})
            try:
                TOOL_CALLS_FILE.write_text(json.dumps(sidecar))
            except Exception:
                pass

            print(COL.format("turn", fmt_n(isl), fmt_n(osl), "-", snippet))
            sys.stdout.flush()

        # ── tool results (appended to context before next vLLM request) ──────
        elif etype == "user":
            content = event.get("message", {}).get("content", [])
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                tool_id   = block.get("tool_use_id", "")
                tool_info = pending_tools.pop(tool_id, {})
                name      = tool_info.get("name", "?")
                inp       = tool_info.get("input", {})
                res_text  = result_text(block.get("content", ""))
                is_error  = block.get("is_error", False)

                isl_chars = len(json.dumps(inp))   # input arg size
                osl_chars = len(res_text)           # result size
                detail    = fmt_input(name, inp)
                if is_error:
                    detail += "  [ERROR]"

                print(COL.format("tool", fmt_n(isl_chars), fmt_n(osl_chars), name, detail))
            if content:
                print(TSEP)
            sys.stdout.flush()

        # ── final summary ─────────────────────────────────────────────────────
        elif etype == "result":
            final_usage = event.get("usage", {})
            subtype     = event.get("subtype", "")
            result_body = event.get("result", "")
            print(SEP)
            status = "OK" if subtype == "success" else subtype.upper()
            print(f"  Result       : {status}")
            if result_body:
                snippet = result_body.replace("\n", " ").strip()
                print(f"  Final output : {snippet[:100]}{'…' if len(result_body) > 100 else ''}")
            if final_usage:
                print(f"  Total input  : {fmt_n(final_usage.get('input_tokens'))} tokens")
                print(f"  Total output : {fmt_n(final_usage.get('output_tokens'))} tokens")
                cr = final_usage.get("cache_read_input_tokens")
                if cr:
                    print(f"  Cache read   : {fmt_n(cr)} tokens")
            print(f"  Turns        : {turn_num}")
            total_tools = sum(e["tool_calls"] for e in sidecar)
            print(f"  Tool calls   : {total_tools}")
            print(SEP)


if __name__ == "__main__":
    main()
