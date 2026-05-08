"""ros2 launch air gazebo.launch.py — Gazebo + TB3 Manipulation + test objects + agent.

Flow:
  1. Bring up Gazebo with the TurtleBot3 Manipulation robot already in it
     (delegated to the upstream `turtlebot3_manipulation_bringup` launch).
  2. Wait a few seconds for Gazebo + ros2_control + the robot to settle.
  3. Spawn a few graspable test objects in front of the robot.
  4. Include agent.launch.py to start agent_node.

Switch sims by writing a sibling file (e.g. unity.launch.py) — agent.launch.py
stays untouched.

Launch arguments:
  world     : .world file path passed through to the TB3 bringup (default: empty).
  headless  : 'true' to run Gazebo without the GUI client (default: false).
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
# ASSUMPTION: ROS 2 Humble + the upstream `turtlebot3_manipulation_bringup`
# package, which ships a `gazebo.launch.py` that spawns the Waffle-Pi base +
# OpenManipulator-X arm and starts ros2_control. If your distro/fork uses a
# different package (e.g. `turtlebot3_manipulation_simulations` on Foxy, or
# `turtlebot3_manipulation_gazebo` in some forks), change TB3_SIM_PACKAGE below.
# -----------------------------------------------------------------------------
TB3_SIM_PACKAGE = "turtlebot3_manipulation_bringup"
TB3_SIM_LAUNCH  = "gazebo.launch.py"


# ---------- inline SDF for the test objects ----------
# Inline strings beat shipping model files: no extra install rules, no model
# path env-var dance, deterministic across machines. Everything below is a
# plain primitive shape with reasonable mass/inertia so MoveIt's gripper can
# actually pick it up without explosions.

def _box_sdf(name: str, size: float, rgba: str) -> str:
    return f"""<?xml version='1.0'?>
<sdf version='1.7'>
  <model name='{name}'>
    <static>false</static>
    <link name='link'>
      <inertial>
        <mass>0.05</mass>
        <inertia><ixx>1e-5</ixx><iyy>1e-5</iyy><izz>1e-5</izz>
                 <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
      </inertial>
      <collision name='c'><geometry><box><size>{size} {size} {size}</size></box></geometry></collision>
      <visual name='v'>
        <geometry><box><size>{size} {size} {size}</size></box></geometry>
        <material><ambient>{rgba}</ambient><diffuse>{rgba}</diffuse></material>
      </visual>
    </link>
  </model>
</sdf>"""


def _cylinder_sdf(name: str, radius: float, length: float, rgba: str) -> str:
    return f"""<?xml version='1.0'?>
<sdf version='1.7'>
  <model name='{name}'>
    <static>false</static>
    <link name='link'>
      <inertial>
        <mass>0.05</mass>
        <inertia><ixx>1e-5</ixx><iyy>1e-5</iyy><izz>1e-5</izz>
                 <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
      </inertial>
      <collision name='c'><geometry><cylinder><radius>{radius}</radius><length>{length}</length></cylinder></geometry></collision>
      <visual name='v'>
        <geometry><cylinder><radius>{radius}</radius><length>{length}</length></cylinder></geometry>
        <material><ambient>{rgba}</ambient><diffuse>{rgba}</diffuse></material>
      </visual>
    </link>
  </model>
</sdf>"""


def _sphere_sdf(name: str, radius: float, rgba: str) -> str:
    return f"""<?xml version='1.0'?>
<sdf version='1.7'>
  <model name='{name}'>
    <static>false</static>
    <link name='link'>
      <inertial>
        <mass>0.05</mass>
        <inertia><ixx>1e-5</ixx><iyy>1e-5</iyy><izz>1e-5</izz>
                 <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
      </inertial>
      <collision name='c'><geometry><sphere><radius>{radius}</radius></sphere></geometry></collision>
      <visual name='v'>
        <geometry><sphere><radius>{radius}</radius></sphere></geometry>
        <material><ambient>{rgba}</ambient><diffuse>{rgba}</diffuse></material>
      </visual>
    </link>
  </model>
</sdf>"""


def _spawn(name: str, sdf_xml: str, x: float, y: float, z: float):
    """gazebo_ros' spawn_entity.py reads SDF as a string and pokes it into the
    running gzserver via service call. Fully ROS-side, no Gazebo plugins needed.
    """
    return Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        name=f"spawn_{name}",
        output="screen",
        arguments=[
            "-entity", name,
            "-x", str(x), "-y", str(y), "-z", str(z),
            "-string", sdf_xml,
        ],
    )


def generate_launch_description():
    world     = LaunchConfiguration("world")
    headless  = LaunchConfiguration("headless")
    do_spawn  = LaunchConfiguration("spawn_objects")

    # 1) Gazebo + the robot. Forward our launch args through to the upstream
    #    bringup. Argument names below match the upstream TB3 manipulation
    #    convention; if your fork uses different ones, tweak here.
    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare(TB3_SIM_PACKAGE), "launch", TB3_SIM_LAUNCH,
            ])
        ),
        launch_arguments={
            "world":    world,
            "headless": headless,
        }.items(),
    )

    # 2) Test objects, sized for the OpenManipulator-X gripper (~2cm jaw width).
    #    Coordinates are in the `world` (== `map`) frame. The robot spawns at
    #    the origin facing +X, so these land in front of the robot, well within
    #    camera FOV and arm reach (~30cm radius).
    cube_x, cube_y       = 0.45,  0.00
    cylinder_x, cylinder_y = 0.45,  0.15
    sphere_x, sphere_y   = 0.45, -0.15

    spawns = [
        _spawn("red_cube",    _box_sdf("red_cube", 0.03, "0.8 0.1 0.1 1"),
               cube_x, cube_y, 0.05),
        _spawn("blue_can",    _cylinder_sdf("blue_can", 0.025, 0.08, "0.1 0.2 0.9 1"),
               cylinder_x, cylinder_y, 0.05),
        _spawn("green_ball",  _sphere_sdf("green_ball", 0.025, "0.1 0.7 0.2 1"),
               sphere_x, sphere_y, 0.05),
    ]

    # Gazebo + ros2_control + URDF parsing takes a few seconds. Spawning objects
    # too early races the world-loading and they fall through the floor or get
    # rejected. 8s is conservative-but-cheap; tune down if your machine is fast.
    delayed_spawns = TimerAction(
        period=8.0,
        actions=spawns,
        condition=IfCondition(do_spawn),
    )

    # 3) Our agent_node. Pulled from the existing launch file so we don't
    #    duplicate its parameters here — single source of truth.
    agent_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("air"), "launch", "agent.launch.py",
            ])
        )
    )
    # Same race concern as the object spawns: the YOLO model load is slow and
    # camera topics aren't published until Gazebo is ready. Delay slightly.
    delayed_agent = TimerAction(period=10.0, actions=[agent_launch])

    return LaunchDescription([
        DeclareLaunchArgument("world",         default_value="",
                              description="Optional .world file passed to the TB3 sim bringup."),
        DeclareLaunchArgument("headless",      default_value="false",
                              description="Set 'true' to run Gazebo without the GUI client."),
        DeclareLaunchArgument("spawn_objects", default_value="true",
                              description="Set 'false' to skip the test-object spawns."),

        sim_launch,
        delayed_spawns,
        delayed_agent,
    ])
