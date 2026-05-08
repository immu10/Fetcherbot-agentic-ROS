"""
Agent tools — thin façade the LLM tool-use loop calls into.

All robot I/O lives in agent/ros/agent_node.py. These functions just forward
to the singleton node. tools.py itself stays pure Python (no rclpy at import
time) so the agent loop can be unit-tested with a mocked DISPATCH.

`TOOLS` is the schema list passed to the LLM.
`DISPATCH` maps tool name → callable for the agent loop.
"""

from __future__ import annotations


# ---------- tool implementations (forward to ROS node) ----------

def _node():
    # Available after `colcon build` + `source install/setup.bash` on Linux.
    from air.agent_node import get_node
    return get_node()


def scan_scene() -> dict:
    return _node().scan_scene()


def navigate_to(x: float, y: float) -> dict:
    return _node().navigate_to(x, y)


def check_nav_status() -> dict:
    return _node().check_nav_status()


def pick_up(object_label: str) -> dict:
    return _node().pick_up(object_label)


def ask_user(question: str) -> dict:
    return _node().ask_user(question)


def release():
    """Shut down the ROS node / executor. Called once at agent exit."""
    from air.agent_node import shutdown_node
    shutdown_node()


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
    "navigate_to": lambda **kw: navigate_to(**kw),
    "check_nav_status": lambda **_: check_nav_status(),
    "pick_up": lambda **kw: pick_up(**kw),
    "ask_user": lambda **kw: ask_user(**kw),
}
