# Agent Flow

The agent runs in two layers:

- **Outer ring** lives in [ros/air/air/agent_node.py](../ros/air/air/agent_node.py) — owns the
  ROS topics (`/agent/question`, `/agent/answer`, `/agent/response`) and the
  read-command → run-agent → publish-reply loop.
- **Inner ring** lives in [agent/agent.py](agent.py) — a LangGraph state machine that
  takes one user command, calls Groq + tools until a final reply is produced,
  returns the reply.

---

## The whole picture

```
                    ┌──────────────┐
START ─────────────►│ wait_for_cmd │ ◄────────────────────────────┐
                    │  (publish    │                               │
                    │   prompt,    │                               │
                    │   await ans  │                               │
                    │   on /agent/ │                               │
                    │   answer)    │                               │
                    └──────┬───────┘                               │
                           │ command                                │
                           ▼                                        │
                    ┌──────────────┐                                │
                    │    think     │ ◄──────────────────┐           │
                    │  (groq call  │                     │           │
                    │   via LangGr │                     │           │
                    │   aph node)  │                     │           │
                    └──────┬───────┘                     │           │
                           │                              │           │
            ┌──────────────┼──────────────┐               │           │
            │ tool_calls:  │ tool_calls:  │ no tool_calls│           │
            │ any non-     │ only         │ (final text) │           │
            │ ask_user     │ ask_user     │              │           │
            ▼              ▼              ▼              │           │
     ┌─────────────┐ ┌─────────────┐ ┌─────────────┐    │           │
     │  exec_tool  │ │  ask_user   │ │   finish    │    │           │
     │ (scan_scene,│ │ (publish Q  │ │ (capture    │    │           │
     │  navigate_  │ │  on /agent/ │ │  reply into │    │           │
     │  to, check_ │ │  question,  │ │  state.     │    │           │
     │  nav_status,│ │  await ans  │ │  final_     │    │           │
     │  pick_up,   │ │  on /agent/ │ │  reply)     │    │           │
     │  ask_user   │ │  answer)    │ └──────┬──────┘    │           │
     │  if mixed)  │ │             │        │           │           │
     └──────┬──────┘ └──────┬──────┘        │           │           │
            │ ToolMessage   │ ToolMessage   │ END       │           │
            │ appended      │ appended      │ of run    │           │
            └───────┬───────┘               │           │           │
                    │                        │           │           │
                    └────────────────────────┼───────────┘           │
                          (back to think)    │                       │
                                              ▼                       │
                                    ┌──────────────────┐              │
                                    │  publish_reply   │              │
                                    │ (push final text │              │
                                    │  on /agent/      │              │
                                    │  response)       │              │
                                    └────────┬─────────┘              │
                                             │                        │
                                             └────────────────────────┘
                                                  (loop forever
                                                   until "quit"
                                                   or Ctrl-C)
```

The dotted region inside the second column (`think` / `exec_tool` / `ask_user` /
`finish`) is the **LangGraph** in [agent/agent.py](agent.py).

The outer ring (`wait_for_cmd` / `publish_reply`) lives in
[agent_node.run_interactive_loop](../ros/air/air/agent_node.py).

---

## Nodes (inner graph)

| Node | What it does | Implemented in |
|---|---|---|
| `think` | Send running message history to Groq, append the response (`AIMessage`). Logs token counts, finish reason, and tool-call names. | `think_node` |
| `exec_tool` | Run every tool call from the latest assistant turn through `agent_tools.DISPATCH`. Used when the LLM mixed any non-`ask_user` calls (or only non-`ask_user`). | `exec_tool_node` |
| `ask_user` | Run when the assistant turn is **only** `ask_user` calls. Same dispatch as `exec_tool`; the split exists to make user-question pauses a visible graph state. | `ask_user_node` |
| `finish` | Capture the LLM's final natural-language reply into `state.final_reply`. | `finish_node` |

All four log to `logs/agent_*.log` via the `air` logger.

---

## Routing rule (`route_after_think`)

```python
tcs = last.tool_calls or []
if not tcs:                                          → "finish"
elif all(tc["name"] == "ask_user" for tc in tcs):    → "ask_user"
else:                                                → "exec_tool"
```

The mixed case (`ask_user` + other tools in the same turn) goes through
`exec_tool` — that node runs **all** tool calls (including any `ask_user`
mixed in), because Groq's tool-calling API requires every assistant
`tool_call.id` to be answered before the next `think` iteration. Splitting the
batch across nodes would break that invariant.

---

## Tools available to the LLM

Each tool is defined in [agent/agent.py](agent.py) as a `@tool`-decorated thin
shim, forwarding to [agent/tools.py](tools.py)'s `DISPATCH` (which respects the
`IS_TEST_RUN` gate and the ROS singleton).

| Tool | What it does | Backed by |
|---|---|---|
| `scan_scene()` | YOLO on latest RGB frame; ground-plane projection to `map` xyz | real ROS via `agent_node.scan_scene` |
| `navigate_to(x, y)` | Send a `NavigateToPose` goal to Nav2 in `map` frame | real ROS via Nav2 action client |
| `check_nav_status()` | Poll the stored Nav2 goal handle | real ROS via Nav2 action client |
| `pick_up(object_label)` | Plan + execute arm grasp | **stub — `NotImplementedError`** |
| `ask_user(question)` | Publish on `/agent/question`, block until `/agent/answer` | real ROS via the outer ring's pub/sub |

`navigate_to` returns immediately with `status:"active"`; the LLM polls
completion via `check_nav_status()`.

---

## Where ROS shows up

The graph itself is pure Python — every ROS interaction is hidden inside the
tool dispatch. So:

- The graph can run in `IS_TEST_RUN=1` (canned data, no ROS) for laptop dev.
- The graph can run inside `agent_node` (full ROS via the singleton).
- Switching simulators (Gazebo → Unity) only touches the agent_node side and
  the launch files; `agent.py` is untouched.

---

## Knobs

| Env var | Effect | Default |
|---|---|---|
| `GROQ_API_KEY` | Groq auth | required |
| `AIR_LLM_ENABLED` | Outer ring runs the graph (`1`) or the scan-only loop (`0`) | `1` |
| `AIR_LLM_DEBUG` | Inside `think_node`, log full request + response bodies | `0` |
| `IS_TEST_RUN` | `agent_tools.DISPATCH` returns canned data instead of hitting ROS | `0` |

See [../md/check.md](../md/check.md) for the full reference.

---

## What changed from the previous design

Earlier sketch (in this file's git history) had a *router* upfront that classified
each user turn as `communication` vs `navigation` vs `manipulation` and
dispatched to one of two loops. We collapsed that to a single LangGraph because:

- Groq's tool-calling already handles the "should I call a tool?" decision
  (it routes to `finish` when no tool call is needed — that *is* the
  conversational case).
- A separate router would be one extra LLM call per turn, no behavioural win.
- The "communication ⇄ action" jump is now just an `ask_user` tool call — no
  classifier, no two-loop bookkeeping.

If we ever want a true planning/decomposition step (the *"plan → think →
verify"* pattern), it'd slot in as a new node before `think` — easy add to the
graph, no architectural surgery.
