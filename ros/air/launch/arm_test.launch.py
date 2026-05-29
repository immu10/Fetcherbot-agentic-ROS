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

Iterate flow:
  1. Edit POSE_* / DUR_* below.
  2. ./run_arm_test.sh --no-build  (launch files are installed, but Python
     source for agent_node also is — re-build if you touched any *.py).
  3. Watch the arm in Gazebo.
"""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


TB3_SIM_PACKAGE = "turtlebot3_manipulation_gazebo"
TB3_SIM_LAUNCH  = "gazebo.launch.py"


# ============================================================================
# TUNING KNOBS — edit these to tune the scripted pick_up. agent_node reads
# all of these as ROS params (see declare_parameter calls in its __init__).
# Joint angles in radians for joint1..4 (OpenManipulator-X).
# ============================================================================
POSE_PRE_GRASP = [0.0,  0.7, -0.4,  0.1]   # extended forward, gripper level
POSE_GRASP     = [0.0,  1.1, -0.7,  0.2]   # gripper at object height, parallel
POSE_LIFT      = [0.0,  0.4,  0.0,  0.0]   # raised, holding

GRIPPER_OPEN   =  0.019    # max open
GRIPPER_CLOSED =  0.010    # barely close — just touch, don't pinch (was 0.0)

# Durations (seconds). Long because the arm is heavy relative to the mobile
# base; fast moves apply reaction forces that flip the bot.
DUR_PRE_GRASP  = 5.0
DUR_GRASP      = 4.0
DUR_LIFT       = 6.0
# ============================================================================


def _spawn_file(name: str, sdf_path, x: float, y: float, z: float):
    return Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        name=f"spawn_{name}",
        output="screen",
        arguments=[
            "-entity", name,
            "-x", str(x), "-y", str(y), "-z", str(z),
            "-file", sdf_path,
        ],
    )


def _spawn_db(name: str, db_model: str, x: float, y: float, z: float):
    """Spawn a Gazebo db model (coke_can, plastic_cup, cricket_ball, ...).
    Brings full native collision + gravity — NOT compatible with fake-attach.
    Used here only for YOLO-detection smoke tests before we wrap each model
    into a ghost-collision SDF for actual pickup."""
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
    # Gazebo + bot + ros2_control (same as the full launch), BUT with our own
    # world file. Upstream's default (turtlebot3_world.world) doesn't load the
    # gazebo_ros_state plugin, so /gazebo/get_entity_state and
    # /gazebo/set_entity_state never exist and fake-attach silently no-ops.
    # Our world is empty ground + sun + that plugin — nothing else needed for
    # the scripted pick_up tuning loop.
    arm_test_world = PathJoinSubstitution([
        FindPackageShare("air"), "worlds", "arm_test.world",
    ])
    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare(TB3_SIM_PACKAGE), "launch", TB3_SIM_LAUNCH,
            ])
        ),
        launch_arguments={"world": arm_test_world}.items(),
    )

    # MoveIt2 move_group — adds IK + motion planning on top of the controllers
    # we already have running. Includes from the upstream config package. We
    # rely on the sim already having published /robot_description and bringing
    # up /joint_states via ros2_control, so we only need the brain, not the
    # full Gazebo-tied launch.
    move_group_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("turtlebot3_manipulation_moveit_config"),
                "launch", "move_group.launch.py",
            ])
        ),
        launch_arguments={"use_sim_time": "true"}.items(),
    )
    # Delay enough that sim + controllers are up first; MoveIt needs them.
    delayed_move_group = TimerAction(period=9.0, actions=[move_group_launch])

    # Test object ~25 cm in front of the bot. Custom SDF (not -database
    # coke_can) because fake-attach pose-snaps the object every tick — with
    # collisions enabled, the contact solver fights the snap and launches
    # the bot. Our test_can.sdf is a coke-can-sized cylinder with collision
    # bitmask 0x00 so it never registers contacts with anything.
    # ====== KNOWN-GOOD PICKUP STATE (commented for YOLO smoke test) ======
    # Restore by uncommenting this block and commenting out the db-spawn
    # block below. This is the working fake-attach pickup setup — one
    # ghost-collision test object spawned at the canned-pose grab point.
    # test_obj_sdf = PathJoinSubstitution([
    #     FindPackageShare("air"), "models", "test_ball.sdf",
    # ])
    # obj = _spawn_file("test_obj", test_obj_sdf, x=-1.75, y=-0.5, z=0.05)
    # delayed_obj = TimerAction(period=8.0, actions=[obj])
    # =====================================================================

    # YOLO smoke test: spawn the three db objects from gazebo.launch.py
    # in front of the bot. NATIVE collision + gravity — pickup will NOT
    # work cleanly on these (fake-attach fights physics); only here to
    # verify YOLO classifies them. Coords approximate the arm-test grab
    # zone (~(-1.75, -0.5)) with the three objects spread sideways so the
    # camera sees all three at once.
    yolo_spawns = [
        # Google Research photogrammetry-scanned teddy bear from Gazebo Fuel.
        # Real-product texture → YOLO/COCO "teddy bear" is a reliable hit.
        # Requires GAZEBO_MODEL_PATH to include ~/gazebo_models.
        _spawn_db("teddy", "Lovable_Huggable_Cuddly_Boutique_Teddy_Bear_Beige",
                  x=-1.60, y=-0.50, z=0.25),
    ]
    delayed_obj = TimerAction(period=8.0, actions=yolo_spawns)

    # agent_node — launched directly (not via agent.launch.py) so we can pass
    # the tuning overrides as parameters. With AIR_LLM_ENABLED=0 (forced
    # below) the scan-only loop fires pick_up() once at startup.
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
            # Label→entity map for fake-attach. scan_only fires
            # pick_up("test_object") and we spawn the SDF as "test_obj".
            # The full launch will extend these arrays.
            "fake_attach_labels":   ["test_object"],
            "fake_attach_entities": ["test_obj"],
        }],
    )
    delayed_agent = TimerAction(period=10.0, actions=[agent_node])

    return LaunchDescription([
        # Force LLM off — pick_up runs automatically on startup via the
        # scan-only loop. Matches what run_arm_test.sh exports.
        SetEnvironmentVariable("AIR_LLM_ENABLED", "0"),
        sim_launch,
        delayed_obj,
        delayed_move_group,
        delayed_agent,
    ])
