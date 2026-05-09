"""ros2 launch air gazebo.launch.py — Gazebo + TB3 Manipulation + SLAM + Nav2 + agent.

Flow:
  1. Bring up Gazebo with the TurtleBot3 Manipulation robot already in it
     (delegated to the upstream `turtlebot3_manipulation_bringup` launch).
  2. Wait a few seconds for Gazebo + ros2_control + the robot to settle.
  3. Spawn a few graspable test objects in front of the robot.
  4. Start slam_toolbox (online_async, mapping mode) — publishes /map and tf
     chain map→odom→base_footprint that ground-plane projection + Nav2 need.
  5. Start Nav2 (turtlebot3_manipulation_navigation2) — exposes the
     /navigate_to_pose action server agent_node will hit.
  6. Include agent.launch.py to start agent_node.

Switch sims by writing a sibling file (e.g. unity.launch.py) — agent.launch.py
stays untouched.

Launch arguments:
  spawn_objects : 'false' to skip the test-object spawns (default: true).
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


# -----------------------------------------------------------------------------
# ROS 2 Humble apt build of TB3 Manipulation: the canonical sim launch lives in
# the moveit_config package, not bringup. moveit_gazebo.launch.py spins up
# Gazebo, spawns the Waffle-Pi base + OpenManipulator-X arm with ros2_control,
# AND starts MoveIt2 in one shot — so we don't need a separate MoveIt include.
# -----------------------------------------------------------------------------
TB3_SIM_PACKAGE = "turtlebot3_manipulation_gazebo"
TB3_SIM_LAUNCH  = "gazebo.launch.py"


# ---------- spawning real Gazebo models ----------
# Switched away from inline-SDF primitive shapes to Gazebo's bundled model
# database — the YOLO11/COCO labels are far more useful when the visuals are
# actual textured objects ("bottle", "cup", "sports ball") instead of bare
# coloured cubes. spawn_entity.py's -database flag pulls the named model from
# Gazebo's model db (cached under ~/.gazebo/models/ after first fetch); a
# working internet connection is needed once, after that it's offline-friendly.

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
    do_spawn  = LaunchConfiguration("spawn_objects")

    # 1) Gazebo + the robot + MoveIt2 (all from the upstream launch).
    #    No launch_arguments forwarded: moveit_gazebo.launch.py doesn't declare
    #    `world` or `headless`, and forwarding undeclared args is a hard error.
    #    Add args here only after confirming the upstream launch declares them.
    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare(TB3_SIM_PACKAGE), "launch", TB3_SIM_LAUNCH,
            ])
        ),
    )

    # 2) Test objects in front of the robot (origin = robot center, +X = forward).
    #    The model names below come from Gazebo's bundled database; the third
    #    column is the COCO class YOLO11 will most likely return.
    #
    #      coke_can       → "bottle"      (sometimes "cup" depending on angle)
    #      plastic_cup    → "cup"
    #      cricket_ball   → "sports ball"
    #
    #    All three cache under ~/.gazebo/models/ after first fetch. If Gazebo
    #    can't reach its database, swap to local SDFs (see git history) or use
    #    Ignition Fuel `<include><uri>https://fuel...</uri></include>`.
    spawns = [
        _spawn_db("coke",  "coke_can",     x=0.45, y=0.00,  z=0.05),
        _spawn_db("cup",   "plastic_cup",  x=0.45, y=0.15,  z=0.05),
        _spawn_db("ball",  "cricket_ball", x=0.45, y=-0.15, z=0.05),
    ]

    # Gazebo + ros2_control + URDF parsing takes a few seconds. Spawning objects
    # too early races the world-loading and they fall through the floor or get
    # rejected. 8s is conservative-but-cheap; tune down if your machine is fast.
    delayed_spawns = TimerAction(
        period=8.0,
        actions=spawns,
        condition=IfCondition(do_spawn),
    )

    # 3) SLAM — slam_toolbox in mapping mode (online async). Publishes /map and
    #    the map→odom→base_footprint tf chain that Nav2 + ground-plane
    #    projection both consume. To switch to a saved-map workflow later,
    #    swap online_async_launch.py for localization_launch.py and pass
    #    `slam_params_file:=...` pointing at a saved posegraph.
    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("slam_toolbox"), "launch", "online_async_launch.py",
            ])
        ),
        launch_arguments={"use_sim_time": "true"}.items(),
    )
    delayed_slam = TimerAction(period=6.0, actions=[slam_launch])

    # 4) Nav2 — `nav2_bringup/navigation_launch.py` brings up only the planner +
    #    controller + bt_navigator + costmaps; it skips map_server and AMCL,
    #    which means it consumes /map from whoever publishes it (here:
    #    slam_toolbox in mapping mode, live).
    #
    #    We deliberately do NOT use turtlebot3_manipulation_navigation2's
    #    navigation2.launch.py — that one expects a saved map file path and
    #    blows up with "No such file or directory: ''" if you don't pass one.
    #    For live-SLAM workflows the bringup variant is the right pick.
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("nav2_bringup"),
                "launch", "navigation_launch.py",
            ])
        ),
        launch_arguments={"use_sim_time": "true"}.items(),
    )
    delayed_nav2 = TimerAction(period=8.0, actions=[nav2_launch])

    # 5) Our agent_node. Pulled from the existing launch file so we don't
    #    duplicate its parameters here — single source of truth.
    agent_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("air"), "launch", "agent.launch.py",
            ])
        )
    )
    # Same race concern as the object spawns: the YOLO model load is slow and
    # camera topics aren't published until Gazebo is ready. Delay enough that
    # SLAM and Nav2 are also up by the time agent_node first scans / navigates.
    delayed_agent = TimerAction(period=12.0, actions=[agent_launch])

    return LaunchDescription([
        DeclareLaunchArgument("spawn_objects", default_value="true",
                              description="Set 'false' to skip the test-object spawns."),

        sim_launch,
        delayed_spawns,
        delayed_slam,
        delayed_nav2,
        delayed_agent,
    ])
