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

    for _ in range(MAX_TURNS):
        resp = client.chat.completions.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            tools=tools,
            tool_choice="auto",
            messages=messages,
        )

        msg = resp.choices[0].message

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
            return (msg.content or "").strip()

        # Run every tool call in this turn and feed results back.
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            print(f"[tool] {tc.function.name}({json.dumps(args)})")
            output = _run_tool(tc.function.name, args)
            print(f"  → {output}")
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": output,
            })

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
