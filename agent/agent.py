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
    """Run object detection on the current camera frame (single snapshot).

    Returns a JSON string with detected objects: labels, confidence scores,
    bbox in pixel coords, and 3D position in map frame. Use this for a quick
    check of what's directly in front of the robot. If the result is empty
    or you suspect things are out of view, prefer look_around() instead.
    """
    return json.dumps(agent_tools.scan_scene())


# DISABLED — broke things in testing. Re-enable by uncommenting and adding
# `look_around` back into TOOLS below.
# @tool
# def look_around() -> str:
#     """Spin the robot in place 360°, scanning periodically; return merged
#     detections from all viewpoints.
#
#     Use this when asked open-ended questions like "what do you see?" or when
#     a single scan_scene() returned nothing — the object you're looking for
#     might be off to the side. Takes ~13 seconds. Detections are deduplicated
#     by label + position, so the same object seen from multiple angles
#     appears once.
#     """
#     return json.dumps(agent_tools.look_around())


@tool
def navigate_to(points: list) -> str:
    """Drive the robot base in map frame. `points` is a list of [x, y] pairs.

    For a single destination, pass [[x, y]]. Currently only the primary target
    (points[0]) is used; the rest are reserved for future multi-waypoint
    routing — pass them anyway if you have a plan in mind.

    Returns a JSON status; status:'active' means the goal was accepted — your
    turn ends, you'll be re-invoked when nav completes (do NOT poll).
    """
    return json.dumps(agent_tools.navigate_to(points=points))


# DISABLED — re-enable by uncommenting and adding `approach` back into
# ALL_TOOLS + TOOLS_BY_PHASE below.
# @tool
# def approach(x: float, y: float, stop_distance: float = 0.30) -> str:
#     """Drive toward (x, y) but stop `stop_distance` metres short.
#
#     Use this when you have a low-confidence detection and want to get close
#     enough to scan_scene() again confidently. After calling, poll
#     check_nav_status until 'succeeded', then re-scan to confirm.
#     Default stop_distance: 30 cm — close enough for high-confidence YOLO,
#     far enough to not collide.
#     """
#     return json.dumps(agent_tools.approach(x=x, y=y, stop_distance=stop_distance))


@tool
def check_nav_status(wait_seconds: float = 60.0) -> str:
    """Wait up to wait_seconds for the active navigation to finish, then return
    its status: idle | active | succeeded | failed | canceled.

    This BLOCKS server-side on a Nav2 result-callback event — it does NOT poll.
    Call it once after navigate_to / approach / go_to_checkpoint and you'll
    wake up the moment the drive ends. No need to call repeatedly.

    wait_seconds:
      - 60 (default) — fine for any normal drive in this room.
      - small (2-5) — when you want to interleave (e.g. ask the user something
        mid-drive). Returns 'active' on timeout; you can call again to resume
        waiting.
      - 0 — no wait, immediate snapshot.
    """
    return json.dumps(agent_tools.check_nav_status(wait_seconds=wait_seconds))


@tool
def pick_up(object_label: str) -> str:
    """Plan and execute an arm trajectory to pick up the named object."""
    return json.dumps(agent_tools.pick_up(object_label=object_label))


@tool
def list_checkpoints() -> str:
    """List all named navigation checkpoints (e.g. 'kitchen', 'desk') the bot
    knows about. Returns JSON: {"checkpoints": [{"name", "x", "y"}, ...]}.

    Use this when the user mentions a place by name and you want to confirm
    it exists before driving — or to discover what destinations are available
    when answering a "where can you go?" question.
    """
    return json.dumps(agent_tools.list_checkpoints())


@tool
def go_to_checkpoint(name: str) -> str:
    """Navigate to a named checkpoint (e.g. go_to_checkpoint('kitchen')).

    Returns JSON status; status:'active' means accepted — poll check_nav_status
    for completion. Returns status:'failed' immediately if the name is unknown
    (no Nav2 round-trip wasted) — call list_checkpoints() first if unsure, or
    ask_user() to clarify which place they meant.
    """
    return json.dumps(agent_tools.go_to_checkpoint(name=name))


@tool
def ask_user(question: str) -> str:
    """Ask the user a clarifying question. Blocks until the user replies on /agent/answer."""
    return json.dumps(agent_tools.ask_user(question=question))


ALL_TOOLS = [scan_scene, navigate_to, check_nav_status, pick_up,
             list_checkpoints, go_to_checkpoint, ask_user]
# look_around + approach removed — re-add to this list (and uncomment their
# @tool defs above) to enable. release will join when Stage 3 wires it.

# ---------- phase → allowed tools ----------
# Restricting the LLM's tool palette per phase kills whole classes of invalid
# action ("pick_up while driving", "release empty-handed") before the model
# can even consider them. Also trims ~50-100 prompt tokens per turn.
#
# Permissive on read-only/safe tools (scan_scene, list_checkpoints, ask_user)
# in every phase. Restrictive on stateful/destructive ones.
TOOLS_BY_PHASE: dict[str, set[str]] = {
    "idle": {
        "scan_scene", "navigate_to", "go_to_checkpoint",
        "list_checkpoints", "pick_up", "ask_user",
    },
    "navigating": {
        # No new navs (already one in flight). check_nav_status lets the LLM
        # answer "how much longer?" mid-drive without burning a real poll —
        # it'll usually just return immediately with cached state.
        "scan_scene", "list_checkpoints", "ask_user", "check_nav_status",
    },
    "holding": {
        # Holding an object: can navigate to drop-off, can't pick up another
        # thing. release (Stage 3) will land here once wired.
        "scan_scene", "navigate_to", "go_to_checkpoint",
        "list_checkpoints", "ask_user",
    },
}

# Tools that fire Nav2 — exec_tool flags the turn as "yielded" after running
# any of these so the graph ends without another think round. The outer ring
# then waits on the event queue for nav_done before re-invoking.
NAV_FIRING_TOOLS = {"navigate_to", "go_to_checkpoint"}


# ---------- LLM ----------
# Lazy: don't construct on import so AIR_LLM_ENABLED=0 (which avoids importing
# this module entirely) doesn't pay the price, and a missing GROQ_API_KEY
# error surfaces at run time, not import time. We bind tools per think-call
# (cheap) so the toolset can vary by phase.
_base_llm = None

def _get_base_llm():
    global _base_llm
    if _base_llm is None:
        _base_llm = ChatGroq(model=MODEL, max_tokens=MAX_TOKENS)
    return _base_llm


def _llm_for_phase(phase: str):
    """Return an LLM bound to the tool subset allowed in `phase`."""
    allowed_names = TOOLS_BY_PHASE.get(phase, TOOLS_BY_PHASE["idle"])
    tools = [t for t in ALL_TOOLS if t.name in allowed_names]
    return _get_base_llm().bind_tools(tools)


# ---------- graph state ----------

class AgentState(TypedDict):
    """Threaded through every node. `operator.add` concatenates message lists;
    other fields use the default replace semantics."""
    messages: Annotated[List[AnyMessage], operator.add]
    final_reply: str
    yielded: bool  # set True after a nav-firing tool; routes graph to finish


# ---------- nodes ----------

def think_node(state: AgentState) -> dict:
    """Send the running message history to Groq; append its reply.

    Tool palette is selected by the current phase (read via agent_tools.get_phase()
    so the source of truth stays in agent_node). Fewer tools = fewer prompt
    tokens + fewer ways for the LLM to do something the phase doesn't allow.
    """
    msgs = state["messages"]
    phase = agent_tools.get_phase()
    _log.info(f"[graph] think: messages={len(msgs)} phase={phase}")
    if _LLM_DEBUG:
        try:
            body = json.dumps([m.model_dump() for m in msgs], indent=2, default=str)
            _log.info(f"[graph] req body:\n{body}")
        except Exception:
            pass

    response: AIMessage = _llm_for_phase(phase).invoke(msgs)

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
            body = json.dumps(response.model_dump(), indent=2, default=str)
            _log.info(f"[graph] resp body:\n{body}")
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
    """Run every tool call from the latest assistant turn.

    OpenAI/Groq tool-calling requires every assistant tool_call.id to be
    answered before the next think iteration, so the whole batch runs here
    (including any ask_user mixed in).

    If any call was a nav-firing tool (navigate_to / approach / go_to_checkpoint),
    flag the turn as yielded. The graph routes yielded turns directly to finish
    so the outer ring can park on the event queue instead of polling.
    """
    last = state["messages"][-1]
    tcs = last.tool_calls or []
    tool_msgs = [_run_tool(tc) for tc in tcs]
    update: dict = {"messages": tool_msgs}
    if any(tc["name"] in NAV_FIRING_TOOLS for tc in tcs):
        update["yielded"] = True
        _log.info("[graph] exec_tool: yielded (nav fired)")
    return update


def ask_user_node(state: AgentState) -> dict:
    """Run tool calls when the assistant turn is *only* ask_user calls.

    Kept separate from exec_tool for graph clarity — makes the user-question
    state a first-class transition. Behaviour identical to exec_tool_node;
    ask_user is not nav-firing, so no yield handling needed here.
    """
    last = state["messages"][-1]
    return {"messages": [_run_tool(tc) for tc in (last.tool_calls or [])]}


def finish_node(state: AgentState) -> dict:
    """Capture the LLM's final natural-language reply into final_reply.

    Yielded turns end on a ToolMessage (the nav-firing call's result) — there
    is no LLM reply to capture, so final_reply stays empty.
    """
    if state.get("yielded"):
        _log.info("[graph] finish: yielded (no reply)")
        return {"final_reply": ""}
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


def route_after_exec(state: AgentState) -> str:
    """After tools ran, either yield (graph ends) or back to think.

    Yielded turns happen when the LLM just fired Nav2 — we want the graph to
    end so the outer ring can wait for nav_done. Non-yielded turns continue
    reasoning normally.
    """
    return "finish" if state.get("yielded") else "think"


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
    # exec_tool fans out — back to think for more reasoning, or straight to
    # finish if a nav-firing tool yielded.
    g.add_conditional_edges("exec_tool", route_after_exec, {
        "think":  "think",
        "finish": "finish",
    })
    g.add_edge("ask_user",  "think")
    g.add_edge("finish",    END)
    return g.compile()


# Compile once at import; each invoke() runs through fresh state.
_graph = _build_graph()


# ---------- public API ----------

def run_step(history: List[AnyMessage]) -> tuple[List[AnyMessage], str, bool]:
    """Run the graph once against an existing message history.

    Returns:
        (updated_history, reply, yielded)
        - updated_history: messages list with everything the graph appended
          (LLM turns, tool results) — caller (agent_node) keeps this across
          step calls within a session.
        - reply: final natural-language reply, or '' if yielded.
        - yielded: True when a nav-firing tool was the last action and the
          graph chose to end early. Caller should keep history and wait for
          the next event (nav_done / user message).

    Invoked once per "wakeup event" from the outer ring's event queue.
    """
    _log.info(f"[graph] === step start === model={MODEL} history_len={len(history)}")
    initial: AgentState = {
        "messages": history,
        "final_reply": "",
        "yielded": False,
    }
    final = _graph.invoke(initial, config={"recursion_limit": RECURSION_LIMIT})
    new_history = final["messages"]
    reply = final.get("final_reply", "") or ""
    yielded = bool(final.get("yielded"))
    _log.info(
        f"[graph] === step end === yielded={yielded} reply={reply!r} "
        f"history_len={len(new_history)}"
    )
    return new_history, reply, yielded


def run(command: str) -> str:
    """One-shot back-compat helper for the standalone CLI (`python -m agent.agent`).

    Wraps run_step in a single session: build history, step until not yielded.
    Not used by agent_node anymore (that goes straight to run_step).
    """
    _log.info(f"[graph] === run start === model={MODEL} command={command!r}")
    history: List[AnyMessage] = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=command),
    ]
    reply = ""
    # Loop in case of yield — the CLI doesn't have a Nav2 backend to wake us,
    # so in practice yield only happens under IS_TEST_RUN where check_nav_status
    # returns 'succeeded' synthetically. Inject a synthetic nav_done so the LLM
    # can continue reasoning.
    for _ in range(8):
        history, reply, yielded = run_step(history)
        if not yielded:
            break
        # Real mode flips this in agent_node._on_nav_done; CLI/test mode has
        # no Nav2 callback, so simulate the transition here.
        agent_tools.set_phase("idle")
        history.append(HumanMessage(content="[system] navigation finished: succeeded"))
    _log.info(f"[graph] === run end === reply={reply!r}")
    return reply or "(no reply)"


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
