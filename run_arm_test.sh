#!/bin/bash
# Arm-tuning launcher: brings up Gazebo + bot + ros2_control + a ball at
# reach distance, then fires pick_up() once. No Nav2, no SLAM, no LLM.
#
# Iterate on POSE_PRE_GRASP / POSE_GRASP / POSE_LIFT in agent_node.py:
#   1. Edit the pose values.
#   2. ./run_arm_test.sh
#   3. Watch the arm in Gazebo.
#   4. Repeat.
#
# Usage:
#   ./run_arm_test.sh             # build + run
#   ./run_arm_test.sh --no-build  # skip build, just run

set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
WS_ROOT="$REPO_ROOT/ros"

# Always source ROS itself.
source /opt/ros/humble/setup.bash

# 1) Build (skippable). Same caveat as run.sh — plain copy install for safety.
if [ "$1" != "--no-build" ]; then
    echo "[run_arm_test.sh] building air..."
    cd "$WS_ROOT"
    colcon build
fi

# 2) Source the workspace.
source "$WS_ROOT/install/setup.bash"

# 3) Force LLM off so run_scan_only_loop fires the auto pick_up test.
export AIR_LLM_ENABLED=0

# 4) Launch. tee full output for debugging; grep the terminal copy for noise.
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"
LAUNCH_LOG="$LOG_DIR/arm_test_$(date +%Y%m%d_%H%M%S).log"
echo "[run_arm_test.sh] launching air arm_test.launch.py (LLM disabled)"
echo "[run_arm_test.sh] full log → $LAUNCH_LOG"
ros2 launch air arm_test.launch.py 2>&1 \
  | tee "$LAUNCH_LOG" \
  | grep --line-buffered -Ev \
    'worldToMap failed|out of map bounds|out of bounds of the costmap|controller_manager.*list_controllers|waiting for service /controller_manager'
