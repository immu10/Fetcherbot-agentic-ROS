"""
ROS2 wrapper for the agent tools.

One rclpy node owns the YOLO model, action clients, subscribers, and TF buffer.
tools.py calls into the singleton via `get_node()`. rclpy spins on a background
thread so the synchronous agent loop can block on `future.result()`.
"""

from __future__ import annotations

import threading

# import rclpy
# from rclpy.node import Node
# from rclpy.executors import MultiThreadedExecutor


_node_singleton = None
_lock = threading.Lock()


class AgentNode:  # (Node):
    """Owns all robot I/O. One instance per process."""

    def __init__(self):
        # rclpy.init() if not already
        # super().__init__("agent_node")
        # action clients: NavigateToPose, MoveGroup, GripperCommand
        # subscribers: /camera/image_raw, /camera/depth/image_raw, /camera/camera_info, /agent/answer
        # publishers:  /agent/question
        # TF buffer + listener
        # YOLO model load (OD.load_model)
        # MultiThreadedExecutor on a daemon thread
        raise NotImplementedError

    # ---- tool methods (called from tools.py) ----

    def scan_scene(self) -> dict:
        raise NotImplementedError

    def navigate_to(self, x: float, y: float) -> dict:
        raise NotImplementedError

    def check_nav_status(self) -> dict:
        raise NotImplementedError

    def pick_up(self, object_label: str) -> dict:
        raise NotImplementedError

    def ask_user(self, question: str) -> dict:
        raise NotImplementedError

    def shutdown(self):
        # cancel goals, executor.shutdown(), rclpy.shutdown()
        raise NotImplementedError


def get_node() -> AgentNode:
    """Lazy singleton. First call constructs the node + starts the executor."""
    global _node_singleton
    with _lock:
        if _node_singleton is None:
            _node_singleton = AgentNode()
        return _node_singleton


def shutdown_node():
    global _node_singleton
    with _lock:
        if _node_singleton is not None:
            _node_singleton.shutdown()
            _node_singleton = None


def main():
    """Entry point for `ros2 run air agent_node` and the launch file."""
    get_node()
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        shutdown_node()
