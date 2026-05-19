# Agent Flow

The agent runs in two layers:

- **Outer ring** lives in [ros/air/air/agent_node.py](../ros/air/air/agent_node.py)
  — owns the ROS topics (`/agent/user`, `/agent/question`, `/agent/response`)
  and the **event-driven** loop. Blocks on an internal event queue; only
  invokes the LLM when something happens (user message, Nav2 done, etc.).
- **Inner ring** lives in [agent/agent.py](agent.py) — a LangGraph state
  machine that runs one "step" per outer-ring event, calling Groq + tools
  until it either produces a final reply or yields (after firing nav).

---

## The whole picture

```
                        ┌─────────────────────────┐
                        │      OUTER RING         │
                        │    (run_interactive     │
                        │       _loop)            │
                        │                         │
                        │   blocks on event queue │
                        │   ─ user msg            │
                        │   ─ nav_done            │
                        │   ─ (future) pick_done  │
                        │   ─ (future) interrupt  │
                        └────────────┬────────────┘
                                     │ event → append to history
                                     ▼
                        ┌─────────────────────────┐
                        │      run_step(history)  │
                        └────────────┬────────────┘
                                     │
                        ┌────────────▼────────────┐
                        │         think           │ ◄────────┐
                        │   (bind tools by phase, │          │
                        │    invoke Groq)         │          │
                        └────────────┬────────────┘          │
                                     │                       │
              ┌──────────────────────┼──────────────────────┐│
              │ tool_calls (any)     │ only ask_user        ││ no tool calls
              ▼                      ▼                      ▼│
       ┌─────────────┐        ┌─────────────┐       ┌─────────────┐
       │  exec_tool  │        │  ask_user   │       │   finish    │
       │ (dispatch   │        │ (publish Q, │       │ (capture    │
       │  every call,│        │  block on   │       │  final reply│
       │  flag yield │        │  /agent/    │       │  or empty   │
       │  if nav was │        │  user)      │       │  if yielded)│
       │  fired)     │        └──────┬──────┘       └──────┬──────┘
       └──────┬──────┘               │                     │
              │                      │                     │
       ┌──────┴───────┐               │                     │
       │ yielded?     │               │                     │
       │   no  → think│ ──────────────┴─────────────────────┤
       │   yes → fin. │ ────────────────────────────────────┤
       └──────────────┘                                     │
                                                            ▼
                                              ┌─────────────────────────┐
                                              │  back to outer ring     │
                                              │  ─ reply → /agent/      │
                                              │     response            │
                                              │  ─ if yielded → keep    │
                                              │     history, wait again │
                                              │  ─ else → reset session │
                                              └─────────────────────────┘
```

The boxed inner region (`think` / `exec_tool` / `ask_user` / `finish`) is the
**LangGraph** in [agent/agent.py](agent.py). The outer ring is in
[agent_node.run_interactive_loop](../ros/air/air/agent_node.py).

---

## Why event-driven

Previously the LLM polled `check_nav_status` every 1–2 seconds during a drive
— ~20 Groq calls per nav. Now Nav2's result callback wakes the outer ring,
which re-invokes the LLM exactly once. ~85–90% token reduction per fetch task.

A nav-firing tool (`navigate_to` / `approach` / `go_to_checkpoint`) is special:
after `exec_tool` runs one of these, the graph **yields** — ends the step
without another `think` round. The outer ring keeps the message history and
waits on the event queue. When `nav_done` arrives, it appends
`"[system] navigation finished: <status>"` to history and re-invokes — the
LLM resumes reasoning with full context.

---

## Phases & tool restriction

Coarse task phase lives in `agent_node` (read by `agent_tools.get_phase()`).
The graph rebinds the LLM's tool palette each `think` based on phase — kills
whole classes of invalid action before the model can consider them.

| Phase        | Allowed tools                                                                  | Entered by                  | Exited by                   |
|--------------|-------------------------------------------------------------------------------|-----------------------------|-----------------------------|
| `idle`       | scan, navigate, approach, go_to_checkpoint, list_checkpoints, pick_up, ask    | startup, after release      | navigate fires / pick_up    |
| `navigating` | scan, list_checkpoints, ask, check_nav_status                                 | navigate_to / approach / go_to_checkpoint accepted | `_on_nav_done`               |
| `holding`    | scan, navigate, approach, go_to_checkpoint, list_checkpoints, ask  *(release: Stage 3)* | pick_up success (Stage 3)   | release success (Stage 3)   |

Permissive on read-only tools (`scan_scene`, `list_checkpoints`, `ask_user`)
in every phase. Restrictive on stateful/destructive ones.

---

## Nodes (inner graph)

| Node       | What it does                                                                                                               | Implemented in   |
|------------|---------------------------------------------------------------------------------------------------------------------------|------------------|
| `think`    | Read current phase, bind tools allowed in that phase, send history to Groq, append response.                              | `think_node`     |
| `exec_tool`| Run every tool call from the latest assistant turn. Flag `yielded=True` if any call was a nav-firing tool.                | `exec_tool_node` |
| `ask_user` | Run when the turn is **only** `ask_user`. Same dispatch as `exec_tool`; split for graph clarity.                          | `ask_user_node`  |
| `finish`   | Capture final reply (empty if yielded).                                                                                    | `finish_node`    |

---

## Routing

```python
# after think:
if no tool_calls:                         → finish
elif all calls are ask_user:              → ask_user
else:                                     → exec_tool

# after exec_tool:
if state.yielded (nav-firing tool ran):   → finish
else:                                     → think
```

---

## ROS topics

| Topic              | Dir  | Purpose                                                              |
|--------------------|------|----------------------------------------------------------------------|
| `/agent/user`      | sub  | All user input. Subscriber routes to ask_user wait or event queue.  |
| `/agent/question`  | pub  | Questions from the LLM (via `ask_user`).                            |
| `/agent/response`  | pub  | Final replies from the LLM at end of a task.                        |

The graph itself is pure Python — every ROS interaction is hidden in the tool
dispatch. So:

- `IS_TEST_RUN=1` runs the graph + fake-tool layer with no ROS (laptop dev).
- Real mode runs inside `agent_node` (full ROS via the singleton).
- Switching simulators only touches `agent_node` + launch files; `agent.py`
  is untouched.

---

## Knobs

| Env var            | Effect                                                                | Default  |
|--------------------|----------------------------------------------------------------------|----------|
| `GROQ_API_KEY`     | Groq auth                                                            | required |
| `AIR_LLM_ENABLED`  | Outer ring runs the graph (`1`) or the scan-only loop (`0`)          | `1`      |
| `AIR_LLM_DEBUG`    | Log full request + response bodies inside `think_node`               | `0`      |
| `IS_TEST_RUN`      | `agent_tools.DISPATCH` returns canned data instead of hitting ROS    | `0`      |
| `AIR_CHECKPOINTS`  | JSON `{name: [x, y]}` of named places (set by `gazebo.launch.py`)   | unset    |

See [../md/check.md](../md/check.md) for the full reference.
