# Robot Pipeline Flow
### Speech-Controlled Mobile Manipulator | ROS2 + Gazebo + YOLO + LLM Agent

---

## Overview

```
Voice / Text Command → LLM Agent → [Tools] → Task Complete
```

The LLM agent is the brain. It receives a command and dynamically decides
which tools to call, in what order, and how to recover when something goes
wrong. There is no hardcoded pipeline — the agent figures it out, subject to
a phase-based tool palette that prevents it from issuing conflicting commands
mid-action (e.g. firing a new nav goal while the bot is already driving).

---

## Architecture

```
                        ┌─────────────────────────────┐
                        │         LLM Agent            │
                        │  (Groq Llama 3.3-70B,        │
                        │   LangGraph state machine)   │
  Voice / Text ───────► │                              │
                        │  - Reasons about the task    │
                        │  - Picks tools by phase      │
                        │  - Handles failures          │
                        │  - Sequences waypoints       │
                        └────────────┬─────────────────┘
                                     │
            ┌────────────┬───────────┼───────────┬──────────────┐
            │            │           │           │              │
       scan_scene  navigate_to  check_nav   pick_up      go_to_checkpoint
            │            │           │           │              │
          YOLO11       Nav2     (event-driven)  arm +          named
       + ground-                              gripper          pose
         plane                                actions          dict
        projection                          (scripted)
            │            │                      │
         Gazebo       Gazebo                 Gazebo
         Camera        Base                    Arm
```

---

## Event-Driven Outer Ring

The agent loop is not a tight polling loop. It blocks on an event queue and
wakes only on:

- `user_msg` — new command from the user
- `nav_done` — Nav2 reported success/failure
- `pick_done` — arm/gripper sequence finished

`check_nav_status` is therefore mostly a courtesy tool — Nav2 completion is
already pushed into the queue automatically, so the LLM rarely needs to poll.

### Phase-restricted tool palette

| Phase        | Tools available                                                       |
|--------------|-----------------------------------------------------------------------|
| `idle`       | scan_scene, navigate_to, pick_up, list_checkpoints, go_to_checkpoint, ask_user |
| `navigating` | check_nav_status, ask_user                                            |
| `holding`    | navigate_to, go_to_checkpoint, ask_user                               |

While navigating, the LLM cannot fire a new `navigate_to` — it can only ask
questions or wait. This is what makes the loop sane during multi-step fetches.

---

## Agent Tools

### `scan_scene()`
- Runs YOLO11 (`imgsz=1280` for small-object recall) on the current camera frame
- For each detection, back-projects the bbox bottom edge onto the ground plane
  via `image_geometry` + `tf2` to get a map-frame `(x, y)`
- Optionally dumps annotated debug frames to `logs/scans/` when `AIR_SAVE_SCANS=1`
- Returns label, confidence, bbox, and 3D position per detection

### `navigate_to(points: list, stop_distance: float = 0.0)`
- Accepts a list of waypoints (currently uses `points[0]`)
- `stop_distance` backs the goal off along the bot→target line — used to stop
  ~40cm short of a pick target so the arm has room
- Fires the goal, transitions phase to `navigating`, and returns immediately
- Nav2 completion is pushed into the event queue, not polled

#### Incremental midpoint retry
If Nav2 fails (e.g. "off global costmap" because the target sits in unmapped
space), the agent_node automatically:

1. Picks the midpoint between the current pose and the original target
2. Fires that as a fallback goal — succeeding here grows the costmap
3. On midpoint success, retries the **real** target (now likely in-map)
4. Repeats up to `_nav_max_attempts = 8` times

This unlocks navigation to far-side rooms without manual exploration.

### `check_nav_status()`
- Returns the latched last-known nav status — cheap, no polling

### `pick_up(object_label)`
- Scripted joint trajectory: open gripper → pre-grasp pose → grasp pose
  → close gripper → lift pose
- All poses + durations are ROS params, set by `arm_test.launch.py` for fast
  iteration without rebuilding `agent_node`
- Sets phase to `holding` on success
- MoveIt2 IK-based grasping is planned but not yet wired

### `list_checkpoints()` / `go_to_checkpoint(name)`
- Named places (kitchen, desk, couch) defined in `gazebo.launch.py`
- Spawned as no-collision visual markers in Gazebo and exported via the
  `AIR_CHECKPOINTS` env var so the LLM sees them at startup

### `ask_user(question)`
- Used when the agent is stuck or needs disambiguation
- Example: *"I see two pens, which one should I fetch?"*

---

## 3D Object Localization (ground-plane projection)

The TB3 sim's camera publishes RGB only — no depth. We get away with this
because everything the bot picks up sits on the floor: a single ground-plane
intersection gives a usable `(x, y)`.

```
YOLO11 bbox in pixel space
        ↓
take bbox bottom edge (where the object meets the floor)
        ↓
image_geometry.PinholeCameraModel.projectPixelTo3dRay
        ↓
intersect that ray with the z=0 plane (in camera frame)
        ↓
tf2 transform → map frame
        ↓
Nav2 goal
```

Notable quirks fixed along the way:

- **Camera `frame_id`** — the sim publishes `base_footprint` instead of the
  real optical frame, breaking tf lookups. Overridden in code to
  `camera_rgb_optical_frame`.
- **Use bbox bottom, not center** — projecting the bbox center assumes the
  object floats; on small objects this overshoots by ~1m. Bottom edge gives
  the floor contact point.

---

## Example Agent Run

**Command:** *"fetch the pen"*

```
1. user_msg → agent
2. agent → scan_scene()              → pen at (1.2, 0.4)
3. agent → navigate_to([(1.2, 0.4)], stop_distance=0.40)
                                     → phase: navigating
4. (Nav2 fails: off costmap)
   agent_node retries midpoint       → success
   agent_node retries real target    → success
   nav_done pushed to queue          → phase: idle
5. agent → scan_scene()              → confirms pen still visible
6. agent → pick_up("pen")            → phase: holding
7. agent → go_to_checkpoint("desk")  → phase: navigating
   ... nav_done ...                  → phase: holding
8. agent → "I've brought the pen to the desk."
```

---

## ROS2 Node Summary

| Node           | Role                                          | Notes                                  |
|----------------|-----------------------------------------------|----------------------------------------|
| `agent_node`   | LLM agent + tool orchestration + projection   | All ROS-side logic lives here          |
| `nav2`         | Base navigation                               | `allow_unknown: true`, `xy_goal_tolerance: 0.10` (patched at launch) |
| `slam_toolbox` | Online async SLAM                             | Builds the map as the bot drives       |
| `move_group`   | MoveIt2 (planned use for grasp IK)            | Launched but not yet driven by pick_up |
| (speech)       | Voice → text                                  | Stubbed; text input goes straight to `/agent/user` |

---

## Tech Stack

| Layer            | Tool                                                      |
|------------------|-----------------------------------------------------------|
| Robot            | TurtleBot3 Waffle Pi + OpenManipulator-X (4-DOF)          |
| Simulator        | Gazebo Classic 11                                         |
| ROS Version      | ROS 2 Humble                                              |
| Object Detection | YOLO11 (`imgsz=1280`)                                     |
| 3D Localization  | `image_geometry` ground-plane projection + `tf2`          |
| Navigation       | Nav2 (NavfnPlanner + DWBLocalPlanner) + slam_toolbox      |
| Manipulation     | ros2_control FollowJointTrajectory + GripperCommand; MoveIt2 wired (IK use TBD) |
| LLM Agent        | Groq Llama 3.3-70B via LangGraph                          |
| Agent Tools      | Python functions wrapped as LangGraph tools, calling ROS  |

---

## What the Agent Handles

| Situation                       | Behaviour                                                       |
|---------------------------------|-----------------------------------------------------------------|
| Object found, path clear        | Navigate (with stop_distance) → pick_up                         |
| Multiple matching objects       | Picks closest or asks the user                                  |
| Object not detected             | Asks the user to clarify or reposition                          |
| Nav2 goal in unmapped space     | Auto midpoint-retry, up to 8 attempts                           |
| Nav2 fully blocked              | Reports failure, falls back to ask_user                         |
| Pick up fails                   | Currently: reports failure; retry policy TBD                    |
| Ambiguous command               | Asks the user before acting                                     |

---

## Known issues / in progress

- **Gazebo grasp physics** — closed gripper doesn't reliably hold small
  objects; they squirt out. Buehler's `gazebo_grasp_plugin` is ROS 1 only,
  so the plan is `gazebo_ros_link_attacher` (community-maintained, has ROS 2
  forks) — call `/ATTACHLINK` after the gripper closes, `/DETACHLINK` on
  release.
- **MoveIt2** — `move_group` is launched but `pick_up` still uses scripted
  joint poses. Swap to IK-based planning is queued.
- **Arm vs base mass** — fast trajectories flip the light base. Durations
  currently padded (5/4/6s) until the grasp plugin lands.

---

## Iterating on the arm

`arm_test.launch.py` + `run_arm_test.sh` bring up Gazebo + the bot + a single
coke can at arm's reach, with `AIR_LLM_ENABLED=0` so `pick_up()` fires once
at startup. All grasp poses and durations are declared as launch params at
the top of `arm_test.launch.py` — edit, relaunch, watch the arm. No rebuild
needed unless you touched Python sources.

---

> **Simulation note:** Voice input is stubbed during development — publish
> text directly to `/agent/user` to drive the full agent loop without a
> microphone setup.
