# Setup Instructions

Get this running from scratch. Tested on Ubuntu 22.04 + ROS 2 Humble.


oh yeah wsl works on windows just gotta enable virtualization just ask gpt/claude if you get stuck on it construct works too

---

## 1. Install dependencies

### Linux (Ubuntu 22.04)

```bash
# ROS 2 Humble (skip if already installed)
sudo apt update
sudo apt install -y ros-humble-desktop python3-colcon-common-extensions

# Project-specific apt packages
sudo apt install -y \
    ros-humble-turtlebot3-manipulation \
    ros-humble-turtlebot3-manipulation-gazebo \
    ros-humble-turtlebot3-manipulation-navigation2 \
    ros-humble-nav2-bringup \
    ros-humble-slam-toolbox \
    ros-humble-cv-bridge \
    ros-humble-image-geometry \
    ros-humble-gazebo-ros-pkgs \
    portaudio19-dev libgl1
```

### The Construct

Skip the ROS install + the `ros-humble-*` apt block above — The Construct's images already have them. You still need the Python deps below.

---

## 2. Python deps

```bash
pip install --user --no-cache-dir -r requirements.txt
```

If pip drags in numpy ≥ 2 (cv_bridge breaks), pin it:
```bash
pip install --user --force-reinstall --no-cache-dir "numpy<2" "opencv-python<4.11"
```

---

## 3. Gazebo model cache

Test objects (Coke can, cup, ball) come from Gazebo's model database. Seed it once:

```bash
mv ~/.gazebo/models ~/.gazebo/models.bak 2>/dev/null
git clone https://github.com/osrf/gazebo_models.git ~/.gazebo/models
```

---

## 4. Clone the project

```bash
git clone <this-repo-url> ~/air/roooomba
```

Adjust the path if you want; the code auto-locates it.

---

## 5. `.env` file

Create `~/air/roooomba/.env`:

```dotenv
GROQ_API_KEY=gsk_your_real_key_here
```

That's the minimum. See [check.md](check.md) for all options.

---

## 6. Shell setup

Add to `~/.bashrc`:

```bash
export TURTLEBOT3_MODEL=waffle_pi
source /opt/ros/humble/setup.bash
source ~/air/roooomba/ros/install/setup.bash
```

Then `source ~/.bashrc`.

---

## 7. Build

```bash
cd ~/air/roooomba/ros
colcon build 
source install/setup.bash
```

---

## 8. Run

**Terminal 1** — bring up the whole sim + agent:
```bash
ros2 launch air gazebo.launch.py
```

Wait ~15 s. You'll see Gazebo open, objects spawn, agent_node log:
```
interactive loop ready — awaiting first command.
```

**Terminal 2** — send a command (this terminal stays open while the bot is running):
```bash
source /opt/ros/humble/setup.bash
source ~/air/roooomba/ros/install/setup.bash
ros2 topic pub --once /agent/answer std_msgs/String "data: 'go to the cup'"
```

Optional **Terminal 3** — watch the agent's questions / replies live:
```bash
ros2 topic echo /agent/question
ros2 topic echo /agent/response
```

To exit cleanly: `quit` as a command, or Ctrl-C in Terminal 1.

---

## Verifying it works

See [checklist.md](checklist.md) — three quick smoke tests + three functional tests.

---

## Notes

- Logs land in `~/air/roooomba/logs/agent_<timestamp>.log` per launch.
- All `.env` knobs (debug, test mode, LLM on/off) are documented in [check.md](check.md).
- The first launch downloads the YOLO model (~18 MB). Subsequent launches use the cache.
