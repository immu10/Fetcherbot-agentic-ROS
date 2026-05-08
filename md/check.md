# Environment Variables — `.env` Reference

> All `.env` keys are **optional unless marked Required**. Defaults are baked into
> the code such that a completely empty `.env` runs the system in **production
> mode**: real ROS, real LLM loop, real camera, no debug output.
>
> If you're adding a new env var to the codebase, **its default in code must
> match the production-mode value**, so users only ever set a `.env` key when
> they want to *deviate* from production.

`.env` lives at the repo root: `~/air/roooomba/.env`. It's already in
[.gitignore](.gitignore) — never commit secrets.

---

## Production-mode `.env` (recommended baseline)

Just one required line:

```dotenv
GROQ_API_KEY=gsk_your_real_key_here
```

That's it. Everything else uses defaults.

---

## All keys

### `GROQ_API_KEY` — **Required**

Your Groq API key. Get one free at <https://console.groq.com/keys>.

| Value | Behaviour |
|---|---|
| (unset) | LLM loop refuses to start; `agent.run()` raises before sending any request. |
| `gsk_…` (real key) | Production. Real LLM calls. Tokens are spent per call. |
| `no` / any junk string | Groq accepts the request and returns `401 Invalid API Key`. Useful for end-to-end wiring tests with **zero token spend** — auth fails before inference. |

---

### `AIR_LLM_ENABLED`

Master switch for the LLM agent loop.

| Value | Behaviour |
|---|---|
| `1` (default — production) | `agent_node` runs `run_interactive_loop()`: ask user → run agent → publish reply → repeat. |
| `0` | `agent_node` runs `run_scan_only_loop()`: every 5 s, run YOLO and log detections. **No LLM calls, no `/agent/question`/`/agent/answer` traffic, no tokens.** Useful while iterating on Gazebo / camera / models without burning budget. |

---

### `AIR_LLM_DEBUG`

Verbose request/response logging for every Groq call. Always-on logging
(token counts, finish reason, tool calls, final content) happens regardless.

| Value | Behaviour |
|---|---|
| `0` (default — production) | Compact logs only. Roughly 4–6 lines per LLM round-trip in `logs/agent_*.log`. |
| `1` | Adds full `req body` (every message, every tool schema) and full `resp body` (the entire LangGraph `AIMessage` dump) to the log file. Files get large; great for "why did the LLM do that?" debugging. |

---

### `IS_TEST_RUN`

Switches `agent/tools.py` between real ROS calls and offline stubs. Originally
designed for `python -m agent.agent` development on a laptop without ROS.

| Value | Behaviour |
|---|---|
| `0` / unset (default — production) | Tools call into the live ROS singleton (`air.agent_node.get_node()`). Real camera, real Nav2, real MoveIt2. |
| `1` | Tools return canned data (a `_FAKE_SCENE`, fake nav success, etc.). **Note:** `IS_TEST_RUN=1` + `ros2 launch` deadlocks on `ask_user` (it tries to `input()` from a non-interactive stdin). Use `IS_TEST_RUN=1` only with `python -m agent.agent` from a real terminal. |

---

### `AIR_RGB_TOPIC` / `AIR_DEPTH_TOPIC` / `AIR_CAMERA_INFO_TOPIC`

Camera topic overrides. Read **at module import time** in `agent_node.py`,
which means they must be set in the **shell environment before launch** — a
`.env` value is loaded too late to affect these. Use shell `export` if you
need to override.

| Var | Default (TB3 manipulation Gazebo) |
|---|---|
| `AIR_RGB_TOPIC` | `/pi_camera/image_raw` |
| `AIR_DEPTH_TOPIC` | `/pi_camera/depth/image_raw` |
| `AIR_CAMERA_INFO_TOPIC` | `/pi_camera/camera_info` |

If `ros2 topic list | grep camera` shows different names on your sim, override
like:

```bash
export AIR_RGB_TOPIC=/camera/rgb/image_raw
ros2 launch air gazebo.launch.py
```

---

## Quick `.env` recipes

### Production (real LLM, real ROS, quiet logs)

```dotenv
GROQ_API_KEY=gsk_your_real_key_here
```

### Vision-only iteration (no LLM, no token spend)

```dotenv
GROQ_API_KEY=no
AIR_LLM_ENABLED=0
```

### LLM wiring test (verify pipeline without spending tokens)

```dotenv
GROQ_API_KEY=no
AIR_LLM_ENABLED=1
```

LLM call goes through, gets `401 Invalid API Key`, error is caught + logged,
loop continues. Confirms every wire is connected.

### Deep debug a weird LLM answer

```dotenv
GROQ_API_KEY=gsk_your_real_key_here
AIR_LLM_DEBUG=1
```

Full message bodies dumped to `logs/agent_*.log`.

### Offline laptop dev (no ROS, no Gazebo)

```dotenv
GROQ_API_KEY=gsk_your_real_key_here
IS_TEST_RUN=1
```

Run with `python -m agent.agent "fetch the cup"` — uses canned tool stubs.

---

## Where each var is read

| Var | Read in | Loaded by |
|---|---|---|
| `GROQ_API_KEY` | `agent/agent.py` (Groq client init) | `.env` via `load_dotenv` |
| `AIR_LLM_ENABLED` | `agent_node.main()` | `.env` via `load_dotenv` |
| `AIR_LLM_DEBUG` | `agent/agent.py` module top | `.env` via `load_dotenv` |
| `IS_TEST_RUN` | `agent/tools.py` module top | `.env` via `load_dotenv` (or shell) |
| `AIR_*_TOPIC` | `agent_node.py` module top | **shell only** (loaded too early for `.env`) |

---

## Policy for adding new env vars

When you add a new env var to the code:

1. **Pick a default that matches production behaviour.** If the answer to "what's the production value?" is non-obvious, the env var probably shouldn't exist — pick a sensible hard-coded value instead.
2. **Read with a default**: `os.environ.get("AIR_NEW_FLAG", "default") == "1"` (boolean) or `os.environ.get("AIR_NEW_PATH", "/sane/default")` (string).
3. **Document it here** under "All keys" with a value table and behaviour description.
4. **Add to `.env.example` if we ever introduce one** (currently we don't — `check.md` is the source of truth).
5. **Mention in the recipe section** if it enables a common workflow.
