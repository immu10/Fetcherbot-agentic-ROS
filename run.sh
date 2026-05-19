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

# Load .env so launch-time env vars (AIR_RVIZ, AIR_SAVE_SCANS, ...) reach the
# launch script. agent_node reads .env on its own via python-dotenv, but the
# launch file's os.environ.get only sees real shell env, so we have to export.
if [ -f "$REPO_ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1090
    . "$REPO_ROOT/.env"
    set +a
fi

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

# 3) Launch. Two output sinks:
#    a) Full unfiltered stream → logs/launch_<ts>.log (every ROS node, Nav2 BT,
#       costmap warnings, the works). Greppable after the fact when you need to
#       debug why a nav failed.
#    b) Filtered live stream → your terminal (noise removed).
#    Use `tee` to fork; `grep -v` only touches the terminal copy.
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"
LAUNCH_LOG="$LOG_DIR/launch_$(date +%Y%m%d_%H%M%S).log"
echo "[run.sh] launching air gazebo.launch.py"
echo "[run.sh] full launch log → $LAUNCH_LOG"
ros2 launch air gazebo.launch.py 2>&1 \
  | tee "$LAUNCH_LOG" \
  | grep --line-buffered -Ev \
    'worldToMap failed|out of map bounds|out of bounds of the costmap|controller_manager.*list_controllers|waiting for service /controller_manager'
