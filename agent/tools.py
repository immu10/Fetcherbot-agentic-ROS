"""
Agent tools — isolated test stubs.

These are pure-Python stubs so the LLM tool-use loop can be exercised
without a webcam, ROS, Nav2, or MoveIt2. Each stub returns plausible JSON
that matches the shape the real ROS wrapper will eventually produce.

When this gets wrapped as a ROS2 node:
  - `scan_scene` will subscribe to /camera/image_raw, run YOLO (OD.py's
    `detect_frame`) on the latest frame, and pass detections through the
    localizer for real 3D positions.
  - `navigate_to` / `check_nav_status` will call Nav2 actions.
  - `pick_up` will call MoveIt2.
  - `ask_user` will publish to a UI topic and await a reply.

`TOOLS` is the schema list passed to the Claude API.
`DISPATCH` maps tool name → callable for the agent loop.
"""

from __future__ import annotations

# ---------- fake world state for testing ----------

_FAKE_SCENE = {
    "detections": [
        {"label": "pen",    "confidence": 0.91, "position": {"x": 1.2, "y": 0.4, "z": 0.01}},
        {"label": "eraser", "confidence": 0.87, "position": {"x": 0.8, "y": 0.6, "z": 0.01}},
    ]
}


# ---------- tool implementations (stubs) ----------

def scan_scene() -> dict:
    """Return a canned detection list. Real impl runs YOLO on /camera/image_raw."""
    return _FAKE_SCENE


def navigate_to(x: float, y: float) -> dict:
    # Pretend the path at (1.1, 0.3) is blocked so the agent has to replan.
    if abs(x - 1.2) < 0.05 and abs(y - 0.4) < 0.05:
        return {"status": "failed", "reason": "path blocked at (1.1, 0.3)"}
    return {"status": "success", "reason": f"arrived at ({x}, {y})"}


def check_nav_status() -> dict:
    return {"status": "idle"}


def pick_up(object_label: str) -> dict:
    return {"status": "success", "object": object_label}


def ask_user(question: str) -> dict:
    print(f"\n[agent asks] {question}")
    answer = input("you: ").strip()
    return {"answer": answer}


def release():
    """No-op for the stub build; kept so agent.py can call it unconditionally."""
    return None


# ---------- Tools schema ----------

TOOLS = [
    {
        "name": "scan_scene",
        "description": (
            "Run object detection on the current camera frame. Returns a list "
            "of detected objects with labels, confidence scores, and 3D positions."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "navigate_to",
        "description": "Drive the robot base to the given (x, y) target position.",
        "input_schema": {
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "X coordinate of target."},
                "y": {"type": "number", "description": "Y coordinate of target."},
            },
            "required": ["x", "y"],
        },
    },
    {
        "name": "check_nav_status",
        "description": "Poll the current navigation status (idle/active/succeeded/failed).",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "pick_up",
        "description": "Plan and execute an arm trajectory to pick up the named object.",
        "input_schema": {
            "type": "object",
            "properties": {
                "object_label": {
                    "type": "string",
                    "description": "Label of the object to pick up (e.g. 'pen').",
                },
            },
            "required": ["object_label"],
        },
    },
    {
        "name": "ask_user",
        "description": "Ask the user a clarifying question when stuck or uncertain.",
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "Question to ask the user."},
            },
            "required": ["question"],
        },
    },
]


DISPATCH = {
    "scan_scene": lambda **_: scan_scene(),
    "navigate_to": lambda **kw: navigate_to(**kw),
    "check_nav_status": lambda **_: check_nav_status(),
    "pick_up": lambda **kw: pick_up(**kw),
    "ask_user": lambda **kw: ask_user(**kw),
}
