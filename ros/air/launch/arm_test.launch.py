"""ros2 launch air arm_test.launch.py — Gazebo + bot + test object, NO Nav2/SLAM.

Stripped-down launch for tuning the scripted pick_up routine. Differences from
gazebo.launch.py:
  - No Nav2, no SLAM, no RViz, no named checkpoints.
  - Test object spawned ~30 cm in front of the bot — right at arm's reach.
  - agent_node still comes up, BUT with tuning overrides for pick_up poses
    and durations declared right here in this file. Edit those values and
    relaunch to iterate without touching agent_node.py.
  - LLM is forced off here too (env var) so pick_up fires once at startup via
    run_scan_only_loop.
  - Robot is spawned from OUR wrapper xacro (urdf/air_robot.urdf.xacro) which
    injects the gazebo_ros_vacuum_gripper plugin onto end_effector_link, so
    pick_up can call /vacuum_gripper/switch to actually hold the can.

Iterate flow:
  1. Edit POSE_* / DUR_* below.
  2. ./run_arm_test.sh --no-build  (launch files are installed, but Python
     source for agent_node also is — re-build if you touched any *.py).
  3. Watch the arm in Gazebo.
"""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, RegisterEventHandler, SetEnvironmentVariable, TimerAction
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.actions import Node
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
    # ----- URDF from OUR wrapper xacro (injects the vacuum plugin) -----
    # Same xacro args base.launch.py passes upstream — kept identical so the
    # controllers, hardware interfaces, and Gazebo bits all match.
    robot_description = {
        "robot_description": Command([
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([
                FindPackageShare("air"), "urdf", "air_robot.urdf.xacro",
            ]),
            " ", "prefix:=",               '""',
            " ", "use_sim:=",              "true",
            " ", "use_fake_hardware:=",    "false",
            " ", "fake_sensor_commands:=", "false",
        ])
    }

    # ----- robot_state_publisher with our patched URDF -----
    rsp_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[robot_description, {"use_sim_time": True}],
        output="screen",
    )

    # ----- Gazebo (gzserver + gzclient) via gazebo_ros -----
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("gazebo_ros"), "launch", "gazebo.launch.py",
            ])
        ),
        launch_arguments={"verbose": "false"}.items(),
    )

    # ----- Spawn the bot from /robot_description -----
    spawn_robot = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-topic", "robot_description",
            "-entity", "turtlebot3_manipulation_system",
            "-x", "-2.00", "-y", "-0.50", "-z", "0.01",
        ],
        output="screen",
    )

    # ----- Controller spawners (mirrors base.launch.py upstream) -----
    # joint_state_broadcaster first; the others chain off its exit so they
    # only start once the controller_manager has it loaded.
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

    chain_arm     = RegisterEventHandler(OnProcessExit(target_action=jsb_spawner, on_exit=[arm_spawner]))
    chain_gripper = RegisterEventHandler(OnProcessExit(target_action=jsb_spawner, on_exit=[gripper_spawner]))
    chain_imu     = RegisterEventHandler(OnProcessExit(target_action=jsb_spawner, on_exit=[imu_spawner]))

    # ----- MoveIt2 move_group -----
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

    # ----- Test object ~30 cm in front of the bot -----
    obj = _spawn_db("test_obj", "coke_can", x=-1.7, y=-0.5, z=0.05)
    delayed_obj = TimerAction(period=8.0, actions=[obj])

    # ----- agent_node with tuning overrides -----
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
        rsp_node,
        gazebo_launch,
        spawn_robot,
        jsb_spawner,
        chain_arm,
        chain_gripper,
        chain_imu,
        delayed_obj,
        delayed_move_group,
        delayed_agent,
    ])
