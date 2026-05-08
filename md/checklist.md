# Test Checklist

Run after `git pull` + `colcon build --symlink-install && source install/setup.bash`.

Launch first in one terminal: `ros2 launch air gazebo.launch.py` — wait ~15 s.

## Smoke tests (quick infra checks)

| # | Command | Pass = |
|---|---|---|
| A | `ros2 topic list \| grep map` | `/map` listed |
| B | `ros2 action list \| grep navigate_to_pose` | action listed |
| C | `ros2 run tf2_ros tf2_echo map camera_rgb_optical_frame` | transform prints (not "lookup failed") |

## Functional tests

| # | What | `.env` | Steps | Pass = |
|---|---|---|---|---|
| 1 | Nav2 drives the bot | any | `ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose "{pose: {header: {frame_id: 'map'}, pose: {position: {x: 1.0, y: 0.0}, orientation: {w: 1.0}}}}"` | bot drives forward ~1 m |
| 2 | `scan_scene` returns map-frame coords | `AIR_LLM_ENABLED=0` | `python3 -c "import time; from air.agent_node import get_node; import json; n=get_node(); time.sleep(3); print(json.dumps(n.scan_scene(), indent=2))"` | each detection has `position: {x, y, z=0}` (not `null`) |
| 3 | Full LLM-driven nav | `GROQ_API_KEY=<real>`, `AIR_LLM_ENABLED=1` | `ros2 topic pub --once /agent/answer std_msgs/String "data: 'go to the cup'"` | scan → navigate_to → check_nav_status loop → arrival reply on `/agent/response` |

## If something's off

| Symptom | Likely cause | Fix |
|---|---|---|
| `Nav2 action server unavailable` | Nav2 still booting | Bump `delayed_agent` in `gazebo.launch.py` past 12 s |
| `tf lookup map<-... failed` / `position: null` | SLAM not converged | Drive bot manually a few seconds, retry |
| Bot drives into walls | wrong `cam_frame_id` from CameraInfo | Hardcode `cam_frame = "camera_rgb_optical_frame"` in `_project_to_ground` call |
