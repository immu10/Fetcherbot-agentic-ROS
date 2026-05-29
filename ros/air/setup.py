from setuptools import setup

package_name = "air"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", [
            "launch/agent.launch.py",
            "launch/gazebo.launch.py",
            "launch/arm_test.launch.py",
        ]),
        (f"share/{package_name}/urdf", [
            "urdf/air_robot.urdf.xacro",
        ]),
        (f"share/{package_name}/worlds", [
            "worlds/arm_test.world",
        ]),
        (f"share/{package_name}/models", [
            "models/test_can.sdf",
        ]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="immu10",
    maintainer_email="s.s.immanuel149@gmail.com",
    description="LLM agent ROS2 wrapper.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "agent_node = air.agent_node:main",
        ],
    },
)
