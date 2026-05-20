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

import json
import os
import tempfile

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


# ---------- named checkpoints ----------
# Single source of truth for "places the bot knows by name". Drives BOTH:
#   1. Visual markers spawned in Gazebo (so you can see where each name is).
#   2. The AIR_CHECKPOINTS env var that agent_node parses at startup, exposing
#      list_checkpoints() and go_to_checkpoint(name) tools to the LLM.
#
# Real-world analog: a marker would be a sign / doorway / decal NEXT TO the
# actual destination, not blocking it. We use a no-collision SDF below so Nav2's
# costmap doesn't inflate around the marker and the bot can drive right up to
# the named coordinate.
#
# FUTURE:
#   - A `save_checkpoint(name)` tool could let the user teach new spots at
#     runtime (capture current bot pose via tf, append here / persist to YAML).
#   - "Fetch to me" — instead of a fixed checkpoint, run YOLO for the 'person'
#     class, project to map frame, navigate to that point. Falls back to
#     ask_user if no person is detected. Keeps the checkpoint dict relevant
#     for non-person destinations ("put it on the desk").
# Out of scope for now — edit this dict and re-launch to add places.
CHECKPOINTS = {
    # name:    (x,    y)   in map frame
    "kitchen": (2.0,  1.0),
    "desk":    (-1.5, 0.5),
    "couch":   (1.0, -1.5),
}


# Purely visual marker — no <collision> tag means Nav2 ignores it entirely.
# A blue post with an orange ball on top, ~0.7m tall (visible above clutter
# but not in the way of the camera at table height).
_MARKER_SDF = """<?xml version="1.0"?>
<sdf version="1.6">
  <model name="MODEL_NAME">
    <static>true</static>
    <link name="link">
      <visual name="post">
        <pose>0 0 0.30 0 0 0</pose>
        <geometry><cylinder><radius>0.04</radius><length>0.60</length></cylinder></geometry>
        <material>
          <ambient>0.1 0.5 0.9 1</ambient>
          <diffuse>0.1 0.5 0.9 1</diffuse>
        </material>
      </visual>
      <visual name="cap">
        <pose>0 0 0.65 0 0 0</pose>
        <geometry><sphere><radius>0.08</radius></sphere></geometry>
        <material>
          <ambient>0.9 0.4 0.1 1</ambient>
          <diffuse>0.9 0.4 0.1 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""


def _patched_nav2_params() -> str:
    """Copy nav2_bringup's nav2_params.yaml to /tmp with `allow_unknown: true`
    flipped on for the global planner. Returns the patched file's path.

    Why patch instead of vendoring: the upstream YAML is ~500 lines and tracks
    nav2 versions; vendoring means we drift. Patching keeps us aligned with
    whatever the installed nav2_bringup version ships, and only the bits we
    actually care about (a single string substitution).

    `allow_unknown: true` lets NavfnPlanner plan paths through cells SLAM
    hasn't seen yet — bot drives into unexplored space, SLAM fills in as it
    goes. Without this, "go to the kitchen" fails until the kitchen has been
    mapped by manual teleop.
    """
    from ament_index_python.packages import get_package_share_directory
    src = os.path.join(
        get_package_share_directory("nav2_bringup"),
        "params", "nav2_params.yaml",
    )
    with open(src, "r") as f:
        text = f.read()

    # The planner_server -> GridBased plugin block sets `allow_unknown: false`
    # on Humble. Flip every occurrence to be safe (the controller also has a
    # similar key but it's harmless to flip there too).
    patched = text.replace("allow_unknown: false", "allow_unknown: true")
    if "allow_unknown" not in patched:
        # Upstream changed the key name or removed the default — bail out and
        # use the file as-is rather than silently doing nothing.
        return src

    # Tighten the goal tolerance from upstream's 0.25 m → 0.10 m. Default is
    # too loose for our use case — bot can stop 25 cm from a checkpoint marker
    # and Nav2 still calls it "arrived", which looks visually short. 10 cm is
    # close enough that the bot is visibly at the marker without being so
    # tight that Nav2 spins forever trying to nudge into position.
    patched = patched.replace("xy_goal_tolerance: 0.25", "xy_goal_tolerance: 0.10")

    out = os.path.join(tempfile.gettempdir(), "air_nav2_params.yaml")
    with open(out, "w") as f:
        f.write(patched)
    return out


def _spawn_checkpoint_marker(name: str, x: float, y: float):
    """Write the no-collision SDF to /tmp and spawn it via gazebo_ros.
    Using -file (not -string) because the apt-installed spawn_entity.py on
    Humble doesn't accept -string in all builds.
    """
    sdf_path = os.path.join(tempfile.gettempdir(), f"air_cp_{name}.sdf")
    with open(sdf_path, "w") as f:
        f.write(_MARKER_SDF.replace("MODEL_NAME", f"cp_{name}"))
    return Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        name=f"spawn_cp_{name}",
        output="screen",
        arguments=[
            "-entity", f"cp_{name}",
            "-file", sdf_path,
            "-x", str(x), "-y", str(y), "-z", "0.0",
        ],
    )


# -----------------------------------------------------------------------------
# ROS 2 Humble apt build of TB3 Manipulation: the canonical sim launch lives in
# the moveit_config package, not bringup. moveit_gazebo.launch.py spins up
# Gazebo, spawns the Waffle-Pi base + OpenManipulator-X arm with ros2_control,
# AND starts MoveIt2 in one shot — so we don't need a separate MoveIt include.
# -----------------------------------------------------------------------------
TB3_SIM_PACKAGE = "turtlebot3_manipulation_gazebo"
TB3_SIM_LAUNCH  = "gazebo.launch.py"

# Flip to False to skip RViz on launch (e.g. WSL without X server).
RVIZ_ENABLED = False


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
        # Cup + coke disabled while debugging ball-only nav. Their costmap
        # inflation halos overlap the ball's approach zone, making stop_distance
        # navigation fail. Re-enable once spawn coords are spread out.
        # _spawn_db("coke",  "coke_can",     x=0.45, y=-0.15,  z=0.05),
        # _spawn_db("cup",   "plastic_cup",  x=0.45, y=-0.30,  z=0.05),
        _spawn_db("ball",  "cricket_ball", x=-0.45, y=-0.45, z=0.05),
    ]

    # Gazebo + ros2_control + URDF parsing takes a few seconds. Spawning objects
    # too early races the world-loading and they fall through the floor or get
    # rejected. 8s is conservative-but-cheap; tune down if your machine is fast.
    delayed_spawns = TimerAction(
        period=8.0,
        actions=spawns,
        condition=IfCondition(do_spawn),
    )

    # Checkpoint markers — same delay as object spawns. Always spawned (not
    # gated by spawn_objects); they're cheap and the agent_node will only see
    # them if AIR_CHECKPOINTS is set, which we do unconditionally below.
    checkpoint_spawns = [
        _spawn_checkpoint_marker(n, x, y) for n, (x, y) in CHECKPOINTS.items()
    ]
    delayed_checkpoints = TimerAction(period=8.0, actions=checkpoint_spawns)

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
    #
    #    `params_file` MUST be passed explicitly: nav2_bringup's launch
    #    declares it with a sensible default, but DeclareLaunchArgument
    #    defaults don't propagate through IncludeLaunchDescription on Humble.
    #    Instead of pointing at the upstream file as-is, we patch it at launch
    #    time to enable `allow_unknown` — lets the planner route through cells
    #    SLAM hasn't mapped yet, so "go to the kitchen" succeeds even when
    #    half the room is still unexplored.
    nav2_params_file = _patched_nav2_params()
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("nav2_bringup"),
                "launch", "navigation_launch.py",
            ])
        ),
        launch_arguments={
            "use_sim_time": "true",
            "params_file": nav2_params_file,
            "autostart":   "true",
        }.items(),
    )
    delayed_nav2 = TimerAction(period=8.0, actions=[nav2_launch])

    # 4b) RViz — Nav2's default view shows /map, both costmaps, the planned
    #     path, the local plan, and the bot's footprint. Gated by the
    #     RVIZ_ENABLED constant at the top of this file — flip it to False
    #     if WSL X server is missing or RViz crashes on your machine.
    if RVIZ_ENABLED:
        rviz_config = PathJoinSubstitution([
            FindPackageShare("nav2_bringup"), "rviz", "nav2_default_view.rviz",
        ])
        rviz_node = Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
            arguments=["-d", rviz_config],
            parameters=[{"use_sim_time": True}],
        )
        delayed_rviz = TimerAction(period=10.0, actions=[rviz_node])

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

        # Export checkpoints to env so agent_node (started further down inside
        # agent.launch.py) reads the same dict that drives the markers above.
        # Must precede the agent include — env vars are inherited by child
        # processes only if set before they're spawned.
        SetEnvironmentVariable("AIR_CHECKPOINTS", json.dumps(CHECKPOINTS)),

        sim_launch,
        delayed_spawns,
        delayed_checkpoints,
        delayed_slam,
        delayed_nav2,
        delayed_agent,
        *([delayed_rviz] if RVIZ_ENABLED else []),
    ])
