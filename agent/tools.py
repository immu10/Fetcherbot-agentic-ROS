"""
Agent tools — thin façade the LLM tool-use loop calls into.

All robot I/O lives in agent/ros/agent_node.py. These functions just forward
to the singleton node. tools.py itself stays pure Python (no rclpy at import
time) so the agent loop can be unit-tested with a mocked DISPATCH.

`TOOLS` is the schema list passed to the LLM.
`DISPATCH` maps tool name → callable for the agent loop.
"""

from __future__ import annotations

import os


# ---------- mode switch ----------
# Default: real ROS calls (camera, Nav2, MoveIt2). Set IS_TEST_RUN=1 to fall
# back to offline stubs — useful on Windows / before `colcon build`.
IS_TEST_RUN = os.environ.get("IS_TEST_RUN") == "1"


# ---------- tool implementations ----------

def _node():
    # Available after `colcon build` + `source install/setup.bash` on Linux.
    from air.agent_node import get_node
    return get_node()


_FAKE_SCENE = {
    "detections": [
        {"label": "pen",    "confidence": 0.91, "position": {"x": 1.2, "y": 0.4, "z": 0.01}},
        {"label": "eraser", "confidence": 0.87, "position": {"x": 0.8, "y": 0.6, "z": 0.01}},
    ]
}


def scan_scene() -> dict:
    if IS_TEST_RUN:
        return _FAKE_SCENE
    return _node().scan_scene()


# DISABLED — see agent/agent.py for re-enable steps.
# def look_around() -> dict:
#     if IS_TEST_RUN:
#         return _FAKE_SCENE  # same canned scene; test path doesn't move the bot
#     return _node().look_around()


def navigate_to(points: list) -> dict:
    if IS_TEST_RUN:
        if not points:
            return {"status": "failed", "reason": "navigate_to: empty points list"}
        x, y = float(points[0][0]), float(points[0][1])
        # Pretend (1.2, 0.4) is blocked so the agent has to replan.
        if abs(x - 1.2) < 0.05 and abs(y - 0.4) < 0.05:
            return {"status": "failed", "reason": "path blocked at (1.1, 0.3)"}
        set_phase("navigating")  # mirror real-mode transition (agent_node does this)
        return {"status": "active", "target": {"x": x, "y": y}}
    return _node().navigate_to(points)


def approach(x: float, y: float, stop_distance: float = 0.30) -> dict:
    if IS_TEST_RUN:
        set_phase("navigating")
        return {"status": "active", "reason": f"approaching ({x}, {y}) within {stop_distance}m"}
    return _node().approach(x, y, stop_distance)


def check_nav_status(wait_seconds: float = 60.0) -> dict:
    if IS_TEST_RUN:
        # Pretend the drive finished while we "waited". Lets the agent loop
        # be exercised end-to-end offline without hanging.
        return {"status": "succeeded"}
    return _node().check_nav_status(wait_seconds=wait_seconds)


def pick_up(object_label: str) -> dict:
    if IS_TEST_RUN:
        return {"status": "success", "object": object_label}
    return _node().pick_up(object_label)


_FAKE_CHECKPOINTS = {
    "kitchen": (2.0, 1.0),
    "desk":    (-1.5, 0.5),
    "couch":   (1.0, -1.5),
}


def list_checkpoints() -> dict:
    if IS_TEST_RUN:
        return {"checkpoints": [{"name": n, "x": x, "y": y}
                                for n, (x, y) in _FAKE_CHECKPOINTS.items()]}
    return _node().list_checkpoints()


def go_to_checkpoint(name: str) -> dict:
    if IS_TEST_RUN:
        if name not in _FAKE_CHECKPOINTS:
            return {"status": "failed",
                    "reason": f"unknown checkpoint {name!r}; known: {list(_FAKE_CHECKPOINTS)}"}
        x, y = _FAKE_CHECKPOINTS[name]
        set_phase("navigating")
        return {"status": "active", "target": {"name": name, "x": x, "y": y}}
    return _node().go_to_checkpoint(name)


def ask_user(question: str) -> dict:
    if IS_TEST_RUN:
        print(f"\n[agent asks] {question}")
        return {"answer": input("you: ").strip()}
    return _node().ask_user(question)


def release():
    """Shut down the ROS node / executor. Called once at agent exit."""
    if IS_TEST_RUN:
        return
    from air.agent_node import shutdown_node
    shutdown_node()


# ---------- phase forwarder ----------
# The graph (agent.py) reads phase to pick which tools to bind. Source of
# truth lives in agent_node (mutated by navigate_to / _on_nav_done / etc).
# In test mode we keep a module-level fake so offline runs don't need a node.
_test_phase = "idle"


def get_phase() -> str:
    if IS_TEST_RUN:
        return _test_phase
    return _node().get_phase()


def set_phase(phase: str) -> None:
    """Test-mode helper for advancing the fake phase. No-op in real mode —
    agent_node mutates its own phase directly from navigate_to / pick_up / etc.
    """
    global _test_phase
    if IS_TEST_RUN:
        _test_phase = phase


# ---------- LLM tools schema ----------

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
    # "look_around": lambda **_: look_around(),  # DISABLED
    "navigate_to": lambda **kw: navigate_to(**kw),
    "approach": lambda **kw: approach(**kw),
    "check_nav_status": lambda **kw: check_nav_status(**kw),
    "pick_up": lambda **kw: pick_up(**kw),
    "ask_user": lambda **kw: ask_user(**kw),
    "list_checkpoints": lambda **_: list_checkpoints(),
    "go_to_checkpoint": lambda **kw: go_to_checkpoint(**kw),
}
