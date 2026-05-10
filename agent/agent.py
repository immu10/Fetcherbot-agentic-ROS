"""
LangGraph-based agent: think ⇄ tool / ask_user, with the user-question state
made explicit in the graph (instead of buried inside a tool dispatch).

Graph shape:

                       START
                         │
                         ▼
                       think  ◄─────────────────┐
                         │                       │
            ┌────────────┼────────────┐          │
            ▼            ▼            ▼          │
        exec_tool    ask_user       finish       │
            │            │            │          │
            └────┬───────┘            │          │
                 │ (loops back)       │          │
                 └────────────────────┼──────────┘
                                      ▼
                                     END

Public API:
    run(command: str) -> str

Same signature as the previous Groq-only loop, so agent_node imports unchanged.

Run (standalone):
    python -m agent.agent             # type a command at the prompt
    python -m agent.agent "fetch the cup"

Requires GROQ_API_KEY in the environment or in a .env at the project root.
"""

from __future__ import annotations

import json
import logging
import operator
import os
import sys
from typing import Annotated, List, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage,
)
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

try:
    from . import tools as agent_tools          # `python -m agent.agent`
    from .prompts import SYSTEM_PROMPT
except ImportError:
    import tools as agent_tools                 # `python agent/agent.py`
    from prompts import SYSTEM_PROMPT

# Load .env from the project root (one dir above this file).
load_dotenv(os.path.join(os.path.dirname(__file__), os.pardir, ".env"))

# ---------- knobs ----------
MODEL = "llama-3.3-70b-versatile"
MAX_TOKENS = 1024
RECURSION_LIMIT = 50  # caps total node visits per run; LangGraph's MAX_TURNS analog.

# Shared file logger configured by agent_node._setup_file_logging(). Silent
# when run standalone (no handlers attached).
_log = logging.getLogger("air")

# AIR_LLM_DEBUG=1 dumps full request + response bodies to the log file.
_LLM_DEBUG = os.environ.get("AIR_LLM_DEBUG") == "1"

# SYSTEM_PROMPT is imported from agent.prompts.


# ---------- LangChain tool wrappers ----------
# Thin shims over agent_tools.DISPATCH so the underlying ROS-side logic
# (and the IS_TEST_RUN gate) is unchanged. LangChain reads the docstring
# and type hints to build the schema sent to Groq.

@tool
def scan_scene() -> str:
    """Run object detection on the current camera frame.

    Returns a JSON string with detected objects: labels, confidence scores,
    bbox in pixel coords, and 3D position (when depth is available).
    """
    return json.dumps(agent_tools.scan_scene())


@tool
def navigate_to(x: float, y: float) -> str:
    """Drive the robot base to the given (x, y) target position in map frame.

    Returns a JSON status; status:'active' means the goal was accepted —
    poll check_nav_status for completion.
    """
    return json.dumps(agent_tools.navigate_to(x=x, y=y))


@tool
def check_nav_status() -> str:
    """Poll the navigation status: idle | active | succeeded | failed | canceled."""
    return json.dumps(agent_tools.check_nav_status())


@tool
def pick_up(object_label: str) -> str:
    """Plan and execute an arm trajectory to pick up the named object."""
    return json.dumps(agent_tools.pick_up(object_label=object_label))


@tool
def ask_user(question: str) -> str:
    """Ask the user a clarifying question. Blocks until the user replies on /agent/answer."""
    return json.dumps(agent_tools.ask_user(question=question))


TOOLS = [scan_scene, navigate_to, check_nav_status, pick_up, ask_user]


# ---------- LLM ----------
# Lazy: don't construct on import so AIR_LLM_ENABLED=0 (which avoids importing
# this module entirely) doesn't pay the price, and a missing GROQ_API_KEY
# error surfaces at run time, not import time.
_llm = None

def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(model=MODEL, max_tokens=MAX_TOKENS).bind_tools(TOOLS)
    return _llm


# ---------- graph state ----------

class AgentState(TypedDict):
    """Threaded through every node. `operator.add` makes lists concatenate."""
    messages: Annotated[List[AnyMessage], operator.add]
    final_reply: str


# ---------- nodes ----------

def think_node(state: AgentState) -> dict:
    """Send the running message history to Groq; append its reply."""
    msgs = state["messages"]
    _log.info(f"[graph] think: messages={len(msgs)}")
    if _LLM_DEBUG:
        try:
            _log.info(f"[graph] req body: {[m.dict() for m in msgs]}")
        except Exception:
            pass

    response: AIMessage = _get_llm().invoke(msgs)

    # Token usage lives in response_metadata.token_usage on the langchain-groq
    # response shape; guard against schema drift across versions.
    usage = (response.response_metadata or {}).get("token_usage", {}) or {}
    if usage:
        _log.info(
            f"[graph] resp: tokens prompt={usage.get('prompt_tokens')} "
            f"completion={usage.get('completion_tokens')} total={usage.get('total_tokens')}"
        )
    if response.content:
        text = response.content if isinstance(response.content, str) else str(response.content)
        preview = text if len(text) <= 300 else text[:300] + "..."
        _log.info(f"[graph] content: {preview!r}")
    if response.tool_calls:
        _log.info(f"[graph] tool_calls: {[tc['name'] for tc in response.tool_calls]}")
    if _LLM_DEBUG:
        try:
            _log.info(f"[graph] resp body: {response.dict()}")
        except Exception:
            pass

    return {"messages": [response]}


def _run_tool(tc: dict) -> ToolMessage:
    """Dispatch one tool call through agent_tools.DISPATCH and wrap the result.

    Used by both exec_tool_node and ask_user_node — the only difference between
    them is which calls they receive (route_after_think handles that). Keeps
    the actual dispatch in one place so behaviour stays identical.
    """
    name = tc["name"]
    args = tc.get("args") or {}
    fn = agent_tools.DISPATCH.get(name)
    if fn is None:
        result = json.dumps({"error": f"unknown tool: {name}"})
    else:
        try:
            output = fn(**args)
            result = json.dumps(output)
        except Exception as e:
            result = json.dumps({"error": f"{type(e).__name__}: {e}"})
    print(f"[tool] {name}({json.dumps(args)})")
    print(f"  → {result}")
    _log.info(f"[tool] {name}({json.dumps(args)}) → {result}")
    return ToolMessage(content=result, tool_call_id=tc["id"], name=name)


def exec_tool_node(state: AgentState) -> dict:
    """Run *every* tool call from the latest assistant turn.

    Reached when the LLM made any non-pure-ask_user batch. We run the whole
    batch (including any ask_user mixed in) because OpenAI/Groq tool-calling
    requires every assistant tool_call.id to be answered before the next
    think iteration — splitting the batch across nodes would break that
    invariant.
    """
    last = state["messages"][-1]
    return {"messages": [_run_tool(tc) for tc in (last.tool_calls or [])]}


def ask_user_node(state: AgentState) -> dict:
    """Run tool calls when the assistant turn is *only* ask_user calls.

    This node exists for graph clarity — it makes the user-question state a
    first-class transition rather than a hidden side-effect inside a tool
    dispatch. Behaviour is identical to exec_tool_node; the split is semantic.
    """
    last = state["messages"][-1]
    return {"messages": [_run_tool(tc) for tc in (last.tool_calls or [])]}


def finish_node(state: AgentState) -> dict:
    """Capture the LLM's final natural-language reply into final_reply."""
    last = state["messages"][-1]
    text = last.content if isinstance(last.content, str) else str(last.content)
    reply = (text or "").strip()
    _log.info(f"[graph] finish: reply={reply!r}")
    return {"final_reply": reply}


# ---------- edges ----------

def route_after_think(state: AgentState) -> str:
    last = state["messages"][-1]
    tcs = last.tool_calls or []
    if not tcs:
        return "finish"
    # Pure user-question turn → ask_user node (visual-only routing).
    if all(tc["name"] == "ask_user" for tc in tcs):
        return "ask_user"
    return "exec_tool"


# ---------- compile ----------

def _build_graph():
    g = StateGraph(AgentState)
    g.add_node("think",     think_node)
    g.add_node("exec_tool", exec_tool_node)
    g.add_node("ask_user",  ask_user_node)
    g.add_node("finish",    finish_node)

    g.set_entry_point("think")
    g.add_conditional_edges("think", route_after_think, {
        "exec_tool": "exec_tool",
        "ask_user":  "ask_user",
        "finish":    "finish",
    })
    g.add_edge("exec_tool", "think")
    g.add_edge("ask_user",  "think")
    g.add_edge("finish",    END)
    return g.compile()


# Compile once at import; each invoke() runs through fresh state.
_graph = _build_graph()


# ---------- public API ----------

def run(command: str) -> str:
    """Run the agent loop on a single user command. Returns the final reply.

    Drop-in replacement for the previous Groq-only run(); agent_node calls
    this exactly as before.
    """
    _log.info(f"[graph] === run start === model={MODEL} command={command!r}")
    initial: AgentState = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=command),
        ],
        "final_reply": "",
    }
    final = _graph.invoke(initial, config={"recursion_limit": RECURSION_LIMIT})
    reply = final.get("final_reply") or "(no reply)"
    _log.info(f"[graph] === run end === reply={reply!r}")
    return reply


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
