"""ros2 launch air arm_test.launch.py — Gazebo + bot + test object, NO Nav2/SLAM.

Stripped-down launch for tuning the scripted pick_up routine. Differences from
gazebo.launch.py:
  - No Nav2, no SLAM, no RViz, no named checkpoints.
  - Test object spawned ~30 cm in front of the bot — right at arm's reach.
  - Bot is spawned from OUR wrapper xacro (urdf/air_robot.urdf.xacro) which
    pulls in the upstream TB3 manipulation URDF and bolts a Gazebo vacuum-
    gripper plugin onto end_effector_link. This sidesteps Gazebo Classic's
    broken finger contact — objects stick to the tip when the suction is on.
  - agent_node still comes up, BUT with tuning overrides for pick_up poses
    and durations declared right here in this file. Edit those values and
    relaunch to iterate without touching agent_node.py.
  - LLM is forced off here too (env var) so pick_up fires once at startup via
    run_scan_only_loop.

Iterate flow:
  1. Edit POSE_* / DUR_* below.
  2. ./run_arm_test.sh --no-build  (launch files are installed, but Python
     source for agent_node also is — re-build if you touched any *.py).
  3. Watch the arm in Gazebo.
"""

from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    RegisterEventHandler,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


# ============================================================================
# TUNING KNOBS — edit these to tune the scripted pick_up. agent_node reads
# all of these as ROS params (see declare_parameter calls in its __init__).
# Joint angles in radians for joint1..4 (OpenManipulator-X).
# ============================================================================
POSE_PRE_GRASP = [0.0,  0.7, -0.4,  0.1]   # extended forward, gripper level
POSE_GRASP     = [0.0,  1.1, -0.7,  0.2]   # gripper at object height, parallel
POSE_LIFT      = [0.0,  0.4,  0.0,  0.0]   # raised, holding

GRIPPER_OPEN   =  0.019    # max open
GRIPPER_CLOSED =  0.0      # close on object (less aggressive than -0.01)

# Durations (seconds). Long because the arm is heavy relative to the mobile
# base; fast moves apply reaction forces that flip the bot.
DUR_PRE_GRASP  = 5.0
DUR_GRASP      = 4.0
DUR_LIFT       = 6.0
# ============================================================================


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
    # ---- patched URDF (upstream xacro + vacuum plugin) -------------------
    # We can't include turtlebot3_manipulation_gazebo's gazebo.launch.py
    # wholesale because it spawns the stock URDF — no suction. Instead
    # replicate what base.launch.py + that gazebo launch do, but feed
    # robot_state_publisher OUR wrapper xacro. Net effect: the spawned model
    # has the vacuum plugin on end_effector_link, exposing
    # /vacuum_gripper/switch (std_srvs/SetBool) which pick_up() toggles.
    urdf_file = Command([
        PathJoinSubstitution([FindExecutable(name="xacro")]),
        " ",
        PathJoinSubstitution([FindPackageShare("air"), "urdf", "air_robot.urdf.xacro"]),
        " ",
        "prefix:=",               '""',
        " ",
        "use_sim:=",              "true",
        " ",
        "use_fake_hardware:=",    "false",
        " ",
        "fake_sensor_commands:=", "false",
    ])

    robot_state_pub = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{
            "robot_description": ParameterValue(urdf_file, value_type=str),
            "use_sim_time": True,
        }],
    )

    # Gazebo server + client. Pass the same world the upstream TB3 wrapper
    # uses — bare gazebo_ros default is pitch black (no sun, no ground plane).
    world_file = PathJoinSubstitution([
        FindPackageShare("turtlebot3_gazebo"), "worlds", "turtlebot3_world.world",
    ])
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("gazebo_ros"), "launch", "gazebo.launch.py",
            ])
        ),
        launch_arguments={"verbose": "false", "world": world_file}.items(),
    )

    # Spawn the bot from the URDF we just published on /robot_description.
    spawn_bot = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        output="screen",
        arguments=[
            "-topic", "robot_description",
            "-entity", "turtlebot3_manipulation_system",
            "-x", "-2.00", "-y", "-0.50", "-z", "0.01",
        ],
    )

    # ---- controllers (chained off joint_state_broadcaster's success) -----
    # Mirrors upstream base.launch.py exactly. In sim mode the ros2_control_node
    # is NOT standalone — gazebo_ros2_control loads inside Gazebo from the
    # URDF's <plugin> block, so we only need the spawners here.
    jsb_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
        output="screen",
    )
    arm_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["arm_controller"],
        output="screen",
    )
    gripper_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["gripper_controller"],
        output="screen",
    )
    imu_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["imu_broadcaster"],
        output="screen",
    )

    def after_jsb(spawner):
        return RegisterEventHandler(
            event_handler=OnProcessExit(target_action=jsb_spawner, on_exit=[spawner])
        )

    # ---- MoveIt2 move_group (kept; harmless if pick_up doesn't use it yet) -
    move_group_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("turtlebot3_manipulation_moveit_config"),
                "launch", "move_group.launch.py",
            ])
        ),
        launch_arguments={"use_sim_time": "true"}.items(),
    )
    delayed_move_group = TimerAction(period=9.0, actions=[move_group_launch])

    # ---- Test object ~30 cm in front of the bot --------------------------
    # coke_can (cylinder) instead of cricket_ball (sphere) — Gazebo's contact
    # solver doesn't blow up against flat-based objects.
    obj = _spawn_db("test_obj", "coke_can", x=-1.7, y=-0.5, z=0.05)
    delayed_obj = TimerAction(period=8.0, actions=[obj])

    # ---- agent_node ------------------------------------------------------
    # Launched directly (not via agent.launch.py) so we can pass tuning
    # overrides as parameters. With AIR_LLM_ENABLED=0 (set below) the scan-
    # only loop fires pick_up() once at startup.
    agent_node = Node(
        package="air",
        executable="agent_node",
        name="agent_node",
        output="screen",
        parameters=[{
            "pose_pre_grasp":     POSE_PRE_GRASP,
            "pose_grasp":         POSE_GRASP,
            "pose_lift":          POSE_LIFT,
            "gripper_open":       GRIPPER_OPEN,
            "gripper_closed":     GRIPPER_CLOSED,
            "arm_dur_pre_grasp":  DUR_PRE_GRASP,
            "arm_dur_grasp":      DUR_GRASP,
            "arm_dur_lift":       DUR_LIFT,
        }],
    )
    delayed_agent = TimerAction(period=10.0, actions=[agent_node])

    return LaunchDescription([
        # Force LLM off — pick_up runs automatically on startup via the
        # scan-only loop. Matches what run_arm_test.sh exports.
        SetEnvironmentVariable("AIR_LLM_ENABLED", "0"),
        robot_state_pub,
        gazebo_launch,
        spawn_bot,
        jsb_spawner,
        after_jsb(arm_spawner),
        after_jsb(gripper_spawner),
        after_jsb(imu_spawner),
        delayed_obj,
        delayed_move_group,
        delayed_agent,
    ])
