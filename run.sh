#!/bin/bash
# One-command run: build + source + launch + filter terminal noise.
#
# Full unfiltered logs still go to:
#   ~/.ros/log/<run>/                   — every ROS node's stdout/stderr
#   ~/air/roooomba/logs/agent_<ts>.log  — agent_node + LangGraph + LLM trace
# The grep below only filters what shows up in your terminal.
#
# Usage:
#   ./run.sh             # build + run
#   ./run.sh --no-build  # skip build, just run

set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
WS_ROOT="$REPO_ROOT/ros"

# Always source ROS itself.
source /opt/ros/humble/setup.bash

# 1) Build (skippable). symlink-install can break colcon when setuptools is
#    too new on Humble — we use the plain copy install for safety.
if [ "$1" != "--no-build" ]; then
    echo "[run.sh] building air..."
    cd "$WS_ROOT"
    colcon build
fi

# 2) Source the workspace.
source "$WS_ROOT/install/setup.bash"

# 3) Launch with terminal-side noise filtered. Add patterns to the regex below
#    when new spam appears. Logs on disk are unaffected.
echo "[run.sh] launching air gazebo.launch.py (filtered terminal output)"
ros2 launch air gazebo.launch.py 2>&1 | grep --line-buffered -Ev \
  'worldToMap failed|out of map bounds|out of bounds of the costmap|controller_manager.*list_controllers|waiting for service /controller_manager'
