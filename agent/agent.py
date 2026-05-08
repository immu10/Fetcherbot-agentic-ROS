"""
LLM agent: tool-calling loop over Groq.

Reads a natural-language command, lets the model pick tools from `tools.TOOLS`,
runs them, feeds results back, and stops when the model returns a final text
response.

Run (from the project root):
    python -m agent.agent             # type a command at the prompt
    python -m agent.agent "fetch the pen"

Requires GROQ_API_KEY in the environment or in a .env file at the project root.
"""

from __future__ import annotations

import json
import logging
import os
import sys

from dotenv import load_dotenv
from groq import Groq

try:
    from . import tools as agent_tools          # `python -m agent.agent`
except ImportError:
    import tools as agent_tools                 # `python agent/agent.py`

# Load .env from the project root (one dir above this file).
load_dotenv(os.path.join(os.path.dirname(__file__), os.pardir, ".env"))

# Free-tier friendly. Alternatives:
#   "llama-3.1-8b-instant"      — much faster, less reliable with tools
#   "openai/gpt-oss-20b"        — decent middle ground
MODEL = "llama-3.3-70b-versatile"
MAX_TOKENS = 1024
MAX_TURNS = 12  # safety cap on the tool-use loop

# Shared file logger configured by agent_node._setup_file_logging(). When run
# standalone (`python -m agent.agent`), it has no handlers and our calls are
# silent — that's fine, this module isn't the place to set up logging policy.
_log = logging.getLogger("air")

# AIR_LLM_DEBUG=1 dumps the full request body (every message, every tool
# schema) and the full response body to the log file. Default OFF — verbose
# but invaluable when an answer surprises you.
_LLM_DEBUG = os.environ.get("AIR_LLM_DEBUG") == "1"

SYSTEM_PROMPT = """You are the brain of a mobile manipulator robot.

You receive a voice command from a user and must complete the task by calling
the tools available to you. You can scan the scene, navigate, pick up objects,
and ask the user clarifying questions.

Reasoning guidelines:
- Always scan the scene before navigating or picking up.
- If multiple objects match, pick the closest or ask the user.
- If navigation fails, try a different approach angle before giving up.
- If pick up fails, retry once, then ask the user.
- When the task is complete, respond with a short confirmation to the user.
"""


def _to_openai_tools(anthropic_tools: list[dict]) -> list[dict]:
    """Convert Anthropic-style tool schema to OpenAI/Groq function-call schema."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in anthropic_tools
    ]


def _run_tool(name: str, args: dict) -> str:
    fn = agent_tools.DISPATCH.get(name)
    if fn is None:
        return json.dumps({"error": f"unknown tool: {name}"})
    try:
        result = fn(**(args or {}))
    except Exception as e:
        result = {"error": f"{type(e).__name__}: {e}"}
    return json.dumps(result)


def run(command: str) -> str:
    """Run the agent loop on a single user command. Returns the final reply."""
    client = Groq()  # reads GROQ_API_KEY from env
    tools = _to_openai_tools(agent_tools.TOOLS)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": command},
    ]

    _log.info(f"[groq] === run start === model={MODEL} command={command!r}")

    for turn in range(1, MAX_TURNS + 1):
        # ---- log request ----
        _log.info(f"[groq] req turn={turn} messages={len(messages)} tools={len(tools)}")
        if _LLM_DEBUG:
            _log.info(f"[groq] req body: {json.dumps(messages, default=str)}")

        resp = client.chat.completions.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            tools=tools,
            tool_choice="auto",
            messages=messages,
        )

        # ---- log response ----
        choice = resp.choices[0]
        msg = choice.message
        usage = getattr(resp, "usage", None)
        if usage is not None:
            _log.info(
                f"[groq] resp turn={turn} finish={choice.finish_reason} "
                f"tokens prompt={usage.prompt_tokens} completion={usage.completion_tokens} "
                f"total={usage.total_tokens}"
            )
        else:
            _log.info(f"[groq] resp turn={turn} finish={choice.finish_reason} (no usage)")
        if msg.content:
            preview = msg.content if len(msg.content) <= 300 else msg.content[:300] + "..."
            _log.info(f"[groq] content: {preview!r}")
        if msg.tool_calls:
            names = [tc.function.name for tc in msg.tool_calls]
            _log.info(f"[groq] tool_calls: {names}")
        if _LLM_DEBUG:
            try:
                _log.info(f"[groq] resp body: {resp.model_dump_json()}")
            except Exception:  # SDK shape changes shouldn't kill the loop.
                _log.info(f"[groq] resp body: {resp!r}")

        # Append the assistant turn (Groq SDK objects serialize cleanly via .model_dump).
        assistant_entry: dict = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_entry)

        if not msg.tool_calls:
            # Final answer.
            final = (msg.content or "").strip()
            _log.info(f"[groq] === run end (final, turn={turn}) ===")
            return final

        # Run every tool call in this turn and feed results back.
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            call_line = f"{tc.function.name}({json.dumps(args)})"
            print(f"[tool] {call_line}")
            _log.info(f"[tool] {call_line}")
            output = _run_tool(tc.function.name, args)
            print(f"  → {output}")
            _log.info(f"[tool]   → {output}")
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": output,
            })

    _log.warning(f"[groq] === run end (MAX_TURNS={MAX_TURNS} hit) ===")
    return "(agent hit max turns without finishing)"


def main():
    if not os.environ.get("GROQ_API_KEY"):
        print("Set GROQ_API_KEY in your .env or environment first.", file=sys.stderr)
        sys.exit(1)

    command = " ".join(sys.argv[1:]).strip() or input("command: ").strip()
    if not command:
        return

    try:
        reply = run(command)
        print(f"\n[agent] {reply}")
    finally:
        agent_tools.release()


if __name__ == "__main__":
    main()
