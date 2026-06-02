"""ros2 launch air agent.launch.py — brings up agent_node."""

from launch import LaunchDescription
from launch_ros.actions import Node


# ============================================================================
# PICKUP TUNING (mirrored from arm_test.launch.py). gazebo.launch.py uses
# this file too, so a single edit here keeps both launches in sync. Joint
# angles in radians for joint1..4 (OpenManipulator-X).
# ============================================================================
POSE_PRE_GRASP = [0.0,  0.7, -0.4,  0.1]
POSE_GRASP     = [0.0,  1.1, -0.7,  0.2]
POSE_LIFT      = [0.0,  0.4,  0.0,  0.0]

GRIPPER_OPEN   = 0.019
GRIPPER_CLOSED = 0.010

DUR_PRE_GRASP  = 5.0
DUR_GRASP      = 4.0
DUR_LIFT       = 6.0
# ============================================================================


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
                # Pickup tuning — needed so fake-attach + bot-pin lifecycle
                # in pick_up have the same trajectories that worked in
                # arm_test. Without these the agent falls back to the
                # built-in defaults in agent_node, which may differ.
                "pose_pre_grasp":     POSE_PRE_GRASP,
                "pose_grasp":         POSE_GRASP,
                "pose_lift":          POSE_LIFT,
                "gripper_open":       GRIPPER_OPEN,
                "gripper_closed":     GRIPPER_CLOSED,
                "arm_dur_pre_grasp":  DUR_PRE_GRASP,
                "arm_dur_grasp":      DUR_GRASP,
                "arm_dur_lift":       DUR_LIFT,
                # Label → Gazebo entity-name map for fake-attach. The LLM
                # (or scripted code) calls pick_up("sports ball") /
                # pick_up("test_object"); both labels resolve to the
                # "test_obj" entity we spawn in gazebo.launch.py. Extend
                # this map when more spawnable objects are added.
                "fake_attach_labels":   ["test_object", "sports ball", "ball"],
                "fake_attach_entities": ["test_obj",    "test_obj",    "test_obj"],
            }],
        ),
    ])
