"""ros2 launch air agent.launch.py — brings up agent_node."""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="air",
            executable="agent_node",
            name="agent_node",
            output="screen",
            parameters=[{
                "nav_timeout_s": 60.0,
                "pick_timeout_s": 30.0,
                "ask_timeout_s": 60.0,
                "scan_cache_ttl_s": 2.0,
                "yolo_model_path": "yolo11s.pt",
                "yolo_conf": 0.35,
            }],
        ),
    ])
