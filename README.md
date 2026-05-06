bleh


# Robot Pipeline Flow
### Speech-Controlled Mobile Manipulator | ROS2 + Gazebo + YOLO + LLM Agent

---

## Overview

```
Voice Command → LLM Agent → [Tools] → Task Complete
```

The LLM agent is the brain of the system. It receives the voice command and
dynamically decides which tools to call, in what order, and how to recover
when something goes wrong. There is no hardcoded pipeline — the agent figures
it out.

---

## Architecture

```
                        ┌─────────────────────────────┐
                        │         LLM Agent            │
                        │                              │
  Voice Command ──────► │  - Reasons about the task    │
                        │  - Decides which tools to use │
                        │  - Handles failures           │
                        │  - Sequences waypoints        │
                        └────────────┬─────────────────┘
                                     │
                    ┌────────────────┼─────────────────┐
                    │                │                  │
             scan_scene()    navigate_to()         pick_up()
                    │                │                  │
                 YOLO            Nav2              MoveIt2
                    │                │                  │
                Gazebo           Gazebo             Gazebo
                Camera            Base               Arm
```

---

## Agent Tools

These are the tools the LLM agent can call. Each one maps to a ROS2 action
or service under the hood.

### `scan_scene()`
- Triggers the YOLO node to process the current camera frame
- Returns a list of detected objects with labels, confidence scores, and 3D positions
- Example output:
```json
{
  "detections": [
    { "label": "pen", "confidence": 0.91, "position": { "x": 1.2, "y": 0.4, "z": 0.01 } },
    { "label": "eraser", "confidence": 0.87, "position": { "x": 0.8, "y": 0.6, "z": 0.01 } }
  ]
}
```

### `navigate_to(position)`
- Sends a navigation goal to Nav2
- Robot base drives to the target position
- Returns success or failure with reason
- Example output:
```json
{ "status": "failed", "reason": "path blocked at (1.1, 0.3)" }
```

### `check_nav_status()`
- Polls the current Nav2 action status
- Used by the agent to decide if it should wait, replan, or escalate

### `pick_up(object_label)`
- Triggers MoveIt2 to plan and execute arm trajectory toward the object
- Closes gripper on success
- Returns success or failure
- Example output:
```json
{ "status": "success", "object": "pen" }
```

### `ask_user(question)`
- Publishes a question back to the user interface
- Used when the agent is stuck or uncertain
- Example: *"I can see two pens, which one should I fetch?"*

---

## 3D Object Localization

YOLO only gives a 2D bounding box in pixel space — it has no depth info on its
own. To get a usable 3D position, the `localizer_node` runs a three-step chain:

```
YOLO bounding box (pixels)
        ↓
depth image → 3D point in camera frame
        ↓
tf2 transform → 3D point in map / base frame
        ↓
Nav2 goal / MoveIt2 target
```

**Step 1 — Pixel → camera frame**
The RGB-D camera (simulated in Gazebo as an Intel RealSense) publishes both
`/camera/image_raw` and `/camera/depth/image_raw`. The localizer takes the
center pixel of YOLO's bounding box, looks up its depth value, and uses
`image_geometry` to back-project it into a 3D point in the camera's own frame
— e.g. *"1.2m ahead, 0.1m to the left, relative to the camera."*

**Step 2 — Camera frame → world frame**
`tf2` maintains live transforms between every frame on the robot — camera,
base, arm, map. The localizer calls tf2 to convert the camera-frame point into
whichever frame is needed:

- **Map frame** → passed to Nav2 so it can plan a path to the object
- **Robot base frame** → passed to MoveIt2 so the arm knows where to reach

**No global GPS or world coordinates are needed.** The robot only ever needs
to know where the object is *relative to itself*, and tf2 handles that
continuously as the robot moves.

> **Gazebo note:** The TurtleBot3 Manipulation simulation includes an RGB-D
> camera plugin by default, so depth data is available out of the box — no
> extra setup required.

---

## Example Agent Run

**Voice command:** *"fetch the pen"*

```
1. Agent receives: "fetch the pen"

2. Agent calls: scan_scene()
   → detects: pen (0.91) at (1.2, 0.4), eraser (0.87) at (0.8, 0.6)

3. Agent reasons: command matches "pen", one pen detected, navigate to it

4. Agent calls: navigate_to({ x: 1.2, y: 0.4 })
   → status: failed, path blocked at (1.1, 0.3)

5. Agent reasons: blocked, try approaching from a different angle

6. Agent calls: navigate_to({ x: 1.2, y: 0.2 })
   → status: success

7. Agent calls: scan_scene()
   → confirms pen still visible and reachable

8. Agent calls: pick_up("pen")
   → status: success

9. Agent reports: "I've fetched the pen."
```

---

## ROS2 Node Summary

| Node | Role | Subscribes | Publishes / Actions |
|---|---|---|---|
| `speech_node` | Captures voice input | mic / text input | `/speech/raw_command` |
| `agent_node` | LLM agent + tool orchestration | `/speech/raw_command` | calls all tools |
| `yolo_node` | YOLOv8 object detection | `/camera/image_raw` `/camera/depth/image_raw` | `/yolo/detections` |
| `localizer_node` | Pixel → 3D position via depth + tf2 | `/yolo/detections` `/camera/depth/image_raw` | `/task/target_pose` |
| `nav2` | Base navigation | `/task/target_pose` | `/cmd_vel` |
| `moveit2` | Arm motion planning | arm goal from agent | `/joint_trajectory_controller` |

---

## Tech Stack

| Layer | Tool |
|---|---|
| Robot | TurtleBot3 + OpenManipulator |
| Simulator | Gazebo (via The Construct) |
| Visualizer | RViz2 |
| ROS Version | ROS2 Humble |
| Object Detection | YOLOv8 via `yolov8_ros` |
| 3D Localization | `image_geometry` + `tf2` |
| LLM Agent | LangChain + Claude / GPT (tool calling) |
| Agent Tools | ROS2 actions + services wrapped as LangChain tools |

---

## What the Agent Handles

| Situation | Agent Behaviour |
|---|---|
| Object found, path clear | Navigate → pick up |
| Multiple matching objects | Picks closest, or asks user |
| Object not detected | Asks user to clarify or reposition |
| Nav2 blocked | Replans with different approach angle |
| Pick up fails | Retries once, then asks user |
| Ambiguous command | Asks user before acting |

---

> **Simulation note:** Voice input can be replaced with a simple text publisher
> during development — publish directly to `/speech/raw_command` to test the
> full agent loop without needing a microphone setup.