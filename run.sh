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
#
# Note: we can't `source .env` directly — python-dotenv is more lenient about
# syntax (spaces around =, unquoted values with special chars) and bash chokes.
# Instead, parse only well-formed KEY=VALUE lines, strip surrounding quotes,
# and export each one explicitly. Skip comments and blank lines.
if [ -f "$REPO_ROOT/.env" ]; then
    while IFS= read -r line || [ -n "$line" ]; do
        # Skip comments and blanks.
        case "$line" in
            ''|\#*) continue ;;
        esac
        # Only accept KEY=VALUE form (KEY is alphanumeric + underscore).
        if [[ "$line" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*=(.*)$ ]]; then
            key="${BASH_REMATCH[1]}"
            val="${BASH_REMATCH[2]}"
            # Strip optional surrounding double or single quotes.
            val="${val%\"}"; val="${val#\"}"
            val="${val%\'}"; val="${val#\'}"
            export "$key=$val"
        fi
    done < "$REPO_ROOT/.env"
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
