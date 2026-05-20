"""ros2 launch air arm_test.launch.py — Gazebo + bot + ball, NO Nav2/SLAM.

Stripped-down launch for tuning the scripted pick_up routine. Differences from
gazebo.launch.py:
  - No Nav2, no SLAM, no RViz, no named checkpoints.
  - Ball spawned ~30cm in front of the bot, on the floor — right at the
    arm's reach so you don't need to drive anywhere.
  - agent_node still comes up; run with AIR_LLM_ENABLED=0 so it fires
    pick_up() once at startup automatically (see run_scan_only_loop).

Iterate on POSE_PRE_GRASP / POSE_GRASP / POSE_LIFT in agent_node.py, then
relaunch this. Faster turnaround than the full sim.
"""

import os
import tempfile

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


TB3_SIM_PACKAGE = "turtlebot3_manipulation_gazebo"
TB3_SIM_LAUNCH  = "gazebo.launch.py"


def _spawn_db(name: str, db_model: str, x: float, y: float, z: float):
    return Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        name=f"spawn_{name}",
        output="screen",
        arguments=[
            "-entity", name,
            "-x", str(x), "-y", str(y), "-z", str(z),
            "-database", db_model,
        ],
    )


def generate_launch_description():
    # Gazebo + bot + ros2_control (same as the full launch).
    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare(TB3_SIM_PACKAGE), "launch", TB3_SIM_LAUNCH,
            ])
        ),
    )

    # Test object ~30 cm in front of the bot. Bot spawns at (-2, -0.5) facing
    # +X, so (-1.7, -0.5) is directly ahead at arm's reach.
    #
    # Use a coke_can (cylinder) instead of a sphere — Gazebo's contact solver
    # generates huge impulse forces against spheres because the contact patch
    # is tiny, which sends the bot flying when the gripper touches it. Flat
    # bases (cans, cubes) give stable contacts.
    obj = _spawn_db("test_obj", "coke_can", x=-1.7, y=-0.5, z=0.05)
    delayed_obj = TimerAction(period=8.0, actions=[obj])

    # agent_node — same one. With AIR_LLM_ENABLED=0 it fires pick_up() once
    # at startup, perfect for tuning arm poses without LLM round-trips.
    agent_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("air"), "launch", "agent.launch.py",
            ])
        )
    )
    delayed_agent = TimerAction(period=10.0, actions=[agent_launch])

    return LaunchDescription([
        sim_launch,
        delayed_obj,
        delayed_agent,
    ])
