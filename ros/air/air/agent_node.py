"""
ROS2 wrapper for the agent tools.

One rclpy node owns the YOLO model, camera subscriptions, and (eventually)
action clients + TF buffer. tools.py calls into the singleton via `get_node()`.
rclpy spins on a background thread so the synchronous agent loop can block on
`future.result()` without deadlocking the executor.

Currently implemented:
    - scan_scene()  → live YOLO11 detection on the Gazebo RGB camera, with
                      center-pixel depth → 3D point in the camera's optical
                      frame.

Stubs (raise NotImplementedError until wired to Nav2 / MoveIt2):
    - navigate_to(x, y)
    - check_nav_status()
    - pick_up(object_label)
    - ask_user(question)
"""

from __future__ import annotations

import datetime
import json
import logging
import math
import os
import queue
import sys
import threading
import time
from typing import Optional

import numpy as np

import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Twist
from nav2_msgs.action import NavigateToPose
from control_msgs.action import FollowJointTrajectory, GripperCommand
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration as DurationMsg
from cv_bridge import CvBridge
from image_geometry import PinholeCameraModel
from tf2_ros import Buffer, TransformListener

# Import the standalone YOLO loader from the project's OD module so we don't
# duplicate model config. OD.py lives at the repo root, which is several dirs
# above this file. Walk upward looking for it, then prepend that dir to
# sys.path — no PYTHONPATH export required, works under both colcon
# --symlink-install (realpath resolves through the symlink) and a plain copy
# install. Falls back to leaving OD as None so scan_scene can report cleanly.
def _find_repo_root_with_OD(start: str, max_depth: int = 8) -> Optional[str]:
    here = os.path.dirname(os.path.realpath(start))
    for _ in range(max_depth):
        if os.path.isfile(os.path.join(here, "OD.py")):
            return here
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent
    return None


_repo_root = _find_repo_root_with_OD(__file__)
if _repo_root and _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Load .env from the repo root so AIR_LLM_ENABLED, AIR_LLM_DEBUG, GROQ_API_KEY
# etc. are visible to this process. Topic env vars (AIR_RGB_TOPIC, ...) are
# read at module top below; if you want to override those, use a shell export
# instead — by the time agent_node imports those constants, .env is loaded.
if _repo_root:
    try:
        from dotenv import load_dotenv as _load_dotenv
        _load_dotenv(os.path.join(_repo_root, ".env"))
    except Exception:
        pass  # dotenv missing or .env absent — env-var fallback still works

# Distinguish "OD.py not on disk" from "OD.py loaded but a dep blew up". The
# latter is far more common (pydantic / cv2 / ultralytics in ~/.local) and
# deserves a different error message in the startup log.
try:
    import OD  # type: ignore
    _od_import_error: Optional[str] = None
except Exception as _e:
    OD = None
    _od_import_error = f"{type(_e).__name__}: {_e}"


# ---------- file logging ----------
# Each run writes a fresh timestamped log file under <repo>/logs/. We keep this
# OUR-side logger separate from rclpy's get_logger() (which prints to console
# and into ~/.ros/log/<...>/) so we have a copy alongside the project source.
# Stays a no-op if the repo root couldn't be located (e.g. OD.py was moved).
_air_log = logging.getLogger("air")
_air_log.setLevel(logging.INFO)
_air_log.propagate = False  # don't double-log via the root logger


def _setup_file_logging() -> Optional[str]:
    """Attach a FileHandler writing to <repo>/logs/agent_<UTC ts>.log.
    Idempotent — repeated calls don't stack handlers. Returns the log path,
    or None if no repo root was located."""
    if not _repo_root:
        return None
    # Skip if already configured (e.g. on hot-reload).
    if any(isinstance(h, logging.FileHandler) for h in _air_log.handlers):
        return next(h.baseFilename for h in _air_log.handlers
                    if isinstance(h, logging.FileHandler))

    logs_dir = os.path.join(_repo_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(logs_dir, f"agent_{ts}.log")

    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    _air_log.addHandler(handler)
    return log_path


# ---------- topic configuration ----------
# Defaults match the apt-built turtlebot3_manipulation_gazebo on Humble, which
# publishes the Pi-camera plugin under /pi_camera/* and (currently) ships no
# depth image at all. The depth subscription stays declared so the moment a
# depth plugin is added the 3D back-projection just starts working — no code
# change needed. Override via env vars if your sim publishes elsewhere.
RGB_TOPIC         = os.environ.get("AIR_RGB_TOPIC",         "/pi_camera/image_raw")
DEPTH_TOPIC       = os.environ.get("AIR_DEPTH_TOPIC",       "/pi_camera/depth/image_raw")
CAMERA_INFO_TOPIC = os.environ.get("AIR_CAMERA_INFO_TOPIC", "/pi_camera/camera_info")


_node_singleton: Optional["AgentNode"] = None
_lock = threading.Lock()


class AgentNode(Node):
    """Owns all robot I/O. One instance per process."""

    def __init__(self):
        super().__init__("agent_node")

        # Tunable parameters (also set in agent.launch.py). rclpy needs
        # explicit declaration before values from the launch file take effect.
        self.declare_parameter("nav_timeout_s",     60.0)
        self.declare_parameter("pick_timeout_s",    30.0)
        self.declare_parameter("ask_timeout_s",     60.0)
        self.declare_parameter("scan_cache_ttl_s",   2.0)
        self.declare_parameter("yolo_model_path",   "yolo11s.pt")
        self.declare_parameter("yolo_conf",          0.35)

        # Named checkpoints from gazebo.launch.py via the AIR_CHECKPOINTS env
        # var (JSON: {"name": [x, y], ...}). Single source of truth — that
        # launch file also spawns the visible markers. Empty if unset, in which
        # case go_to_checkpoint() returns a clean error and list_checkpoints()
        # returns an empty list. FUTURE: a save_checkpoint() tool could append
        # at runtime; for now this is launch-time only.
        self._checkpoints: dict[str, tuple[float, float]] = {}
        raw_cp = os.environ.get("AIR_CHECKPOINTS", "").strip()
        if raw_cp:
            try:
                parsed = json.loads(raw_cp)
                self._checkpoints = {
                    str(k): (float(v[0]), float(v[1])) for k, v in parsed.items()
                }
                self.get_logger().info(
                    f"loaded {len(self._checkpoints)} checkpoint(s): "
                    f"{list(self._checkpoints)}"
                )
            except Exception as e:
                self.get_logger().warn(
                    f"AIR_CHECKPOINTS parse failed: {e} (raw={raw_cp!r})"
                )

        # Sensor data + cv_bridge live behind a lock so scan_scene() (called
        # from the agent thread) and the subscription callbacks (executor
        # thread) don't race.
        self._bridge = CvBridge()
        self._frame_lock = threading.Lock()
        self._latest_rgb: Optional[np.ndarray] = None      # HxWx3 BGR
        self._latest_depth: Optional[np.ndarray] = None    # HxW float32 (meters)
        self._cam_model: Optional[PinholeCameraModel] = None
        self._cam_frame_id: str = "camera_link"

        # YOLO model — load once, reuse on every scan. Skipped if OD couldn't
        # be imported so the node still boots and can be poked over topics.
        self._yolo = None
        if OD is not None:
            try:
                self.get_logger().info("loading YOLO model (this can take a few seconds)...")
                self._yolo = OD.load_model()
                self.get_logger().info("YOLO model ready.")
            except Exception as e:
                self.get_logger().error(f"YOLO load failed: {e}")
        else:
            if _repo_root is None:
                self.get_logger().warn(
                    "OD module not importable: could not auto-locate OD.py by "
                    "walking up from this file. Is OD.py still at the project root?"
                )
            else:
                self.get_logger().warn(
                    f"OD module found on path ({_repo_root}) but failed to import: "
                    f"{_od_import_error}. scan_scene() will return an error. "
                    "Common cause: stale pydantic/cv2/ultralytics in ~/.local "
                    "(try: pip install --user --upgrade pydantic groq ultralytics)."
                )

        # Sensor data is high-volume; BEST_EFFORT keeps us from backing up
        # publishers. Camera plugins in Gazebo publish best-effort by default.
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(Image,       RGB_TOPIC,         self._on_rgb,        sensor_qos)
        self.create_subscription(Image,       DEPTH_TOPIC,       self._on_depth,      sensor_qos)
        self.create_subscription(CameraInfo,  CAMERA_INFO_TOPIC, self._on_camera_info, sensor_qos)

        self.get_logger().info(
            f"subscribed: rgb={RGB_TOPIC}  depth={DEPTH_TOPIC}  info={CAMERA_INFO_TOPIC}"
        )

        # ---- unified user channel + outer-ring event queue ----
        # ONE input topic for everything the user says — replies to ask_user,
        # unprompted commands, mid-drive chatter. The subscriber decides which
        # bucket each message lands in:
        #   - if ask_user is currently blocked (_expecting_answer=True) → fulfill
        #     it by waking _answer_event (existing tool semantics, unchanged).
        #   - else → push onto _event_queue as a "user" event. run_interactive_loop
        #     drains the queue and invokes the LLM with that text.
        #
        # The queue also carries non-user events (nav_done, future pick_done,
        # interrupt, ...) so the outer ring has ONE place to wait on anything
        # that might wake the LLM. Cheap, zero polling.
        self._question_pub = self.create_publisher(String, "/agent/question", 10)
        self._response_pub = self.create_publisher(String, "/agent/response", 10)
        self.create_subscription(String, "/agent/user", self._on_user, 10)
        self._event_queue: "queue.Queue[tuple]" = queue.Queue()
        self._expecting_answer = False
        self._answer_event = threading.Event()
        self._answer_lock  = threading.Lock()
        self._latest_answer: Optional[str] = None

        # ---- coarse task phase ----
        # IDLE / NAVIGATING / HOLDING. Used by the graph (via tools.get_phase)
        # to restrict the LLM's tool palette per phase — kills whole classes of
        # invalid actions ("pick_up while driving", "release empty-handed").
        # Mutated by navigate_to (→navigating), _on_nav_done (→idle), pick_up
        # (→holding, Stage 3), release (→idle, Stage 3). Lock so the LLM thread
        # and executor thread can't observe a torn read.
        self._phase: str = "idle"
        self._phase_lock = threading.Lock()

        # ---- tf2 buffer (used by ground-plane projection in scan_scene) ----
        # The TransformListener subscribes to /tf and /tf_static via this node's
        # executor; lookups in scan_scene/navigate_to are pure reads against
        # the buffer.
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # ---- Nav2 action client + nav goal state ----
        # Server name matches the default in turtlebot3_manipulation_navigation2.
        # check_nav_status() reads _nav_goal_handle / _nav_last_status —
        # navigate_to() fills them in.
        self._nav_client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self._nav_goal_handle = None
        self._nav_last_status: Optional[int] = None  # last terminal GoalStatus.

        # Event-driven nav: navigate_to() attaches a done-callback to Nav2's
        # result future; that callback fires on the executor thread the instant
        # Nav2 reports terminal, sets _nav_last_status + clears the handle, and
        # sets this event. check_nav_status(wait_seconds=N) blocks on it instead
        # of polling — one Groq call per drive instead of ~20.
        self._nav_done_event = threading.Event()

        # ---- arm + gripper action clients ----
        # These reach the controllers brought up by ros2_control. Action names
        # match what `ros2 action list` showed:
        #   /arm_controller/follow_joint_trajectory  (FollowJointTrajectory)
        #   /gripper_controller/gripper_cmd          (GripperCommand)
        self._arm_client = ActionClient(
            self, FollowJointTrajectory, "/arm_controller/follow_joint_trajectory",
        )
        self._gripper_client = ActionClient(
            self, GripperCommand, "/gripper_controller/gripper_cmd",
        )

        # OpenManipulator-X has 4 arm joints (joint1..4); ros2_control exposes
        # them under these names. Update if your URDF differs.
        self._arm_joint_names = ["joint1", "joint2", "joint3", "joint4"]

        # Incremental-nav state. On a failure, fire a midpoint between current
        # pose and the real target — that drive grows the SLAM map. After the
        # midpoint succeeds, retry the real target from the new position. Loop
        # until target succeeds, midpoint also fails (truly stuck), or we hit
        # _nav_max_attempts. LLM only sees the final result.
        #
        # _nav_attempt_kind:  "target" or "midpoint" — which kind of goal is in
        #                     flight, so _on_nav_done knows what to do on result.
        self._nav_original_goal: Optional[tuple[float, float]] = None
        self._nav_attempt_kind: str = "target"
        self._nav_attempt: int = 0
        self._nav_max_attempts: int = 8  # enough for ~4 target/midpoint cycles

        # ---- direct base velocity publisher ----
        # Used by look_around() to spin the bot in place. Bypasses Nav2 — fine
        # for in-place rotation. Don't issue both this and navigate_to at the
        # same time; the diff_drive plugin will use whatever arrived last.
        self._cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # Spin on a background thread so synchronous tool calls work.
        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

    # ---- subscription callbacks ----

    def _on_rgb(self, msg: Image):
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"rgb convert failed: {e}")
            return
        with self._frame_lock:
            self._latest_rgb = frame

    def _on_depth(self, msg: Image):
        # Gazebo's depth camera plugin typically publishes 32FC1 in meters.
        # If your sim uses 16UC1 (mm), the divide-by-1000 below converts it.
        try:
            depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().warn(f"depth convert failed: {e}")
            return
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) / 1000.0
        elif depth.dtype != np.float32:
            depth = depth.astype(np.float32)
        with self._frame_lock:
            self._latest_depth = depth

    def _on_user(self, msg: String):
        """Unified user-input handler — routes one /agent/user message to the
        right bucket based on whether ask_user is currently blocked.

        - ask_user pending → wake it; the LLM gets the text as the tool result.
        - otherwise → push onto the outer-ring event queue. run_interactive_loop
          picks it up and runs the LLM with the text as the next turn.

        Both buckets are non-blocking from the executor thread's POV.
        """
        text = msg.data
        if self._expecting_answer:
            with self._answer_lock:
                self._latest_answer = text
            self._answer_event.set()
        else:
            self._event_queue.put(("user", text))

    def _on_camera_info(self, msg: CameraInfo):
        # Build the pinhole model once; intrinsics rarely change at runtime.
        if self._cam_model is None:
            model = PinholeCameraModel()
            model.fromCameraInfo(msg)
            with self._frame_lock:
                self._cam_model = model
                # Gazebo's camera plugin publishes frame_id="base_footprint"
                # for this TB3 build, which is the bot's ground projection —
                # NOT the camera. Casting a ray "from base_footprint" already
                # sits on the floor, so every ground-plane projection collapses
                # to the bot's own (x, y). Override to the URDF's actual
                # optical frame so tf gives us the real camera pose.
                claimed = msg.header.frame_id
                self._cam_frame_id = "camera_rgb_optical_frame"
            self.get_logger().info(
                f"camera intrinsics received (msg.frame_id={claimed!r}, "
                f"using={self._cam_frame_id!r})."
            )

    # ---- tool methods (called from tools.py) ----

    def scan_scene(self) -> dict:
        """Run YOLO on the latest RGB frame; project each detection's center
        pixel onto the floor (z=0) in `map` frame via tf2 — gives the LLM real
        navigable coordinates without needing a depth camera.

        Returns:
            {"detections": [{"label", "confidence", "position": {x,y,z=0},
                             "bbox": [x1,y1,x2,y2], "frame_id": "map"}, ...]}
            or {"error": "..."} on failure. position is None if tf or camera
            intrinsics aren't ready yet (the LLM should retry or ask the user).
        """
        if self._yolo is None:
            return {"error": "YOLO model not loaded (see node startup logs)."}

        with self._frame_lock:
            rgb = None if self._latest_rgb is None else self._latest_rgb.copy()
            cam_model = self._cam_model
            cam_frame = self._cam_frame_id

        if rgb is None:
            return {"error": f"no RGB frame received yet on {RGB_TOPIC}."}

        # YOLO inference + per-detection extraction live in OD.py — agent_node
        # only adds the ROS-specific bit (project pixel → map frame via tf).
        try:
            result, annotated = OD.detect_frame(self._yolo, rgb)
            raw = OD.extract_detections(result)
        except Exception as e:
            return {"error": f"YOLO inference failed: {type(e).__name__}: {e}"}

        # Dump the annotated frame to disk for visual debugging. Opt-in via
        # AIR_SAVE_SCANS=1 so we don't pile up images on every run by default.
        # Each scan gets a timestamped jpg under logs/scans/.
        if os.environ.get("AIR_SAVE_SCANS") == "1" and _repo_root is not None:
            try:
                import cv2
                scans_dir = os.path.join(_repo_root, "logs", "scans")
                os.makedirs(scans_dir, exist_ok=True)
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                path = os.path.join(scans_dir, f"scan_{ts}.jpg")
                cv2.imwrite(path, annotated)
                self.get_logger().info(f"scan saved → {path}")
                _air_log.info(f"scan saved → {path}")
            except Exception as e:
                self.get_logger().warn(f"failed to save scan image: {e}")

        if not raw:
            return {"detections": []}

        detections = []
        for d in raw:
            # Ground-plane projection assumes the projected pixel corresponds
            # to a z=0 point. The bbox CENTER lies above the floor for any
            # object with height, so projecting through it overshoots (ray
            # continues past the object to the floor beyond).
            # The bbox BOTTOM-center is approximately where the object touches
            # the floor — that's the real z=0 contact point. Use it instead.
            cx = d["center"][0]
            cy = int(d["bbox"][3])  # y2 = bottom edge of bbox
            position = self._project_to_ground(cx, cy, cam_model, cam_frame, "map")
            detections.append({
                "label":      d["label"],
                "confidence": d["confidence"],
                "bbox":       d["bbox"],
                "position":   position,        # {x,y,z=0} in map frame, or None
                "frame_id":   "map",
            })

        return {"detections": detections}

    # DISABLED — broke things in testing. Re-enable by uncommenting and adding
    # `look_around` back into agent/tools.py DISPATCH and agent/agent.py TOOLS.
    # def look_around(
    #     self,
    #     num_scans: int = 8,
    #     scan_period_s: float = 1.5,
    #     angular_speed_rad_s: float = 0.6,
    # ) -> dict:
    #     """Spin in place while scanning periodically; return merged detections.
    #
    #     Default: 8 scans across one full rotation (~13 s). Useful when the LLM
    #     is asked "what do you see" and a single forward scan misses things off
    #     to the sides — also helps with low-confidence detections by giving
    #     YOLO multiple shots at each object from slightly different angles.
    #
    #     Detections are deduplicated by (label, position rounded to 10 cm). The
    #     first sighting wins (we keep its bbox + confidence).
    #
    #     Returns:
    #         {"detections": [...], "scans_done": N}
    #     """
    #     twist = Twist()
    #     twist.angular.z = float(angular_speed_rad_s)
    #     stop = Twist()  # zero by default
    #
    #     merged: list[dict] = []
    #     seen: set = set()
    #
    #     try:
    #         for _ in range(num_scans):
    #             # Drive rotation for one scan period before scanning. Publishing
    #             # cmd_vel at ~10 Hz keeps the diff_drive plugin happy (it
    #             # otherwise stops the bot if it stops hearing commands).
    #             end = time.time() + scan_period_s
    #             while time.time() < end:
    #                 self._cmd_vel_pub.publish(twist)
    #                 time.sleep(0.1)
    #
    #             snap = self.scan_scene()
    #             for det in snap.get("detections", []):
    #                 pos = det.get("position") or {}
    #                 key = (
    #                     det.get("label"),
    #                     round(pos.get("x", 0.0), 1),
    #                     round(pos.get("y", 0.0), 1),
    #                 )
    #                 if key in seen:
    #                     continue
    #                 seen.add(key)
    #                 merged.append(det)
    #     finally:
    #         # Always stop the bot on exit, even if scan_scene threw.
    #         for _ in range(3):
    #             self._cmd_vel_pub.publish(stop)
    #             time.sleep(0.05)
    #
    #     return {"detections": merged, "scans_done": num_scans}

    def _pixel_to_camera_xyz(
        self,
        u: int,
        v: int,
        depth: Optional[np.ndarray],
        cam_model: Optional[PinholeCameraModel],
        w: int,
        h: int,
    ) -> Optional[dict]:
        """Back-project pixel (u,v) to a 3D point in the camera optical frame.
        Returns None if depth or intrinsics are missing / invalid at that pixel.
        TODO: tf2 transform into map / base_link for Nav2 / MoveIt2 consumers.
        """
        if depth is None or cam_model is None:
            return None
        if not (0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]):
            return None

        z = float(depth[v, u])
        if not math.isfinite(z) or z <= 0.0:
            return None

        # PinholeCameraModel.projectPixelTo3dRay returns a unit ray (X,Y,1-ish);
        # multiply by z to land on the surface.
        ray = cam_model.projectPixelTo3dRay((u, v))
        # Normalise so ray.z == 1, then scale by depth.
        rz = ray[2] if ray[2] != 0 else 1.0
        x = ray[0] / rz * z
        y = ray[1] / rz * z
        return {"x": float(x), "y": float(y), "z": float(z)}

    @staticmethod
    def _quat_rotate(q, v):
        """Rotate 3-vec v by quaternion q=(x,y,z,w). Pure-Python, no scipy.

        Uses the standard t = 2 * (q.xyz × v); v' = v + qw*t + q.xyz × t form
        — equivalent to q * v * q^-1 for unit quaternions, ~6× faster than
        building a rotation matrix when we only rotate one vector.
        """
        qx, qy, qz, qw = q
        vx, vy, vz = v
        tx = 2.0 * (qy * vz - qz * vy)
        ty = 2.0 * (qz * vx - qx * vz)
        tz = 2.0 * (qx * vy - qy * vx)
        rx = vx + qw * tx + (qy * tz - qz * ty)
        ry = vy + qw * ty + (qz * tx - qx * tz)
        rz = vz + qw * tz + (qx * ty - qy * tx)
        return (rx, ry, rz)

    def _project_to_ground(
        self,
        u: int,
        v: int,
        cam_model: Optional[PinholeCameraModel],
        cam_frame: str,
        target_frame: str = "map",
    ) -> Optional[dict]:
        """Back-project pixel (u,v) and intersect the resulting ray with z=0
        in `target_frame`. Returns {"x","y","z":0} or None if intrinsics or
        the tf chain aren't ready yet.

        Floor-only assumption: the object is at z=0 in target_frame. Tall
        objects on tables will project further out (the ray keeps going past
        them until it hits the ground). For our floor-bot, fine.
        """
        if cam_model is None:
            return None
        try:
            t = self._tf_buffer.lookup_transform(
                target_frame, cam_frame,
                rclpy.time.Time(),                    # latest available
                timeout=Duration(seconds=0.5),
            )
        except Exception as e:
            self.get_logger().debug(f"tf lookup {target_frame}<-{cam_frame} failed: {e}")
            return None

        # Camera origin in target_frame.
        ox = t.transform.translation.x
        oy = t.transform.translation.y
        oz = t.transform.translation.z

        # Pixel → unit ray in CAMERA optical frame, then rotate into target.
        rcx, rcy, rcz = cam_model.projectPixelTo3dRay((u, v))
        q = (
            t.transform.rotation.x, t.transform.rotation.y,
            t.transform.rotation.z, t.transform.rotation.w,
        )
        rx, ry, rz = self._quat_rotate(q, (rcx, rcy, rcz))

        # Ray-plane intersection: ox + s*rx, oy + s*ry, oz + s*rz; solve for z=0.
        if abs(rz) < 1e-6:
            return None  # ray parallel to ground
        s = -oz / rz
        if s <= 0:
            return None  # ray points away from / above ground
        return {"x": float(ox + s * rx), "y": float(oy + s * ry), "z": 0.0}

    def navigate_to(self, points: list, stop_distance: float = 0.0) -> dict:
        """Send a NavigateToPose goal in `map` frame; return immediately.

        `points` is a list of [x, y] pairs. For now we only use points[0]
        (the primary target) — the bisect-retry logic still handles fallbacks
        at fractions 0.5, 0.25, 0.125 along bot→goal automatically.
        Multi-waypoint chains can be wired up later by iterating `points`.

        `stop_distance` (metres): if > 0, back the target off along the
        bot→target line by this much. Use when the goal is an object you want
        to be NEAR, not ON — e.g. stop_distance=0.20 lands the bot ~20 cm from
        the ball instead of driving over it. For checkpoint-style "land on
        this exact point" navigation, leave at 0.

        On failure, _on_nav_done auto-retries at points closer to the bot.
        LLM only sees the final outcome.
        """
        if not points:
            return {"status": "failed", "reason": "navigate_to: empty points list"}
        x, y = float(points[0][0]), float(points[0][1])
        if len(points) > 1:
            self.get_logger().info(
                f"navigate_to: {len(points)} points given, using points[0]=({x:.2f},{y:.2f}); "
                "multi-waypoint not wired yet"
            )

        # stop_distance: pull the goal back toward the bot's current pose.
        # If we can't get the pose (tf timeout), skip the adjustment rather
        # than fail the whole nav — losing 20cm of precision is better than
        # not driving at all.
        if stop_distance > 0:
            try:
                t = self._tf_buffer.lookup_transform(
                    "map", "base_footprint", rclpy.time.Time(),
                    timeout=Duration(seconds=0.5),
                )
                rx, ry = t.transform.translation.x, t.transform.translation.y
                dx, dy = x - rx, y - ry
                dist = math.sqrt(dx * dx + dy * dy)
                if dist > stop_distance:
                    scale = (dist - stop_distance) / dist
                    x_adj = rx + dx * scale
                    y_adj = ry + dy * scale
                    self.get_logger().info(
                        f"navigate_to: stop_distance={stop_distance:.2f}m → "
                        f"({x:.2f},{y:.2f}) adjusted to ({x_adj:.2f},{y_adj:.2f})"
                    )
                    x, y = x_adj, y_adj
                else:
                    self.get_logger().info(
                        f"navigate_to: already within stop_distance ({dist:.2f}m); "
                        "treating as no-op success"
                    )
                    # Synthesize a succeeded result via the same channel so the
                    # outer ring's nav_done event fires consistently.
                    self.set_phase("navigating")
                    self._nav_original_goal = (float(x), float(y))
                    self._nav_attempt = 0
                    self._nav_attempt_kind = "target"
                    self._finalize_nav("succeeded")
                    return {"status": "active", "target": {"x": float(x), "y": float(y)}}
            except Exception as e:
                self.get_logger().warn(
                    f"navigate_to: stop_distance tf lookup failed ({e}); using raw goal"
                )

        self._nav_original_goal = (float(x), float(y))
        self._nav_attempt = 0
        self._nav_attempt_kind = "target"
        return self._send_nav_goal(float(x), float(y))

    def _send_nav_goal(self, x: float, y: float) -> dict:
        """Lower-level: send one NavigateToPose goal and wire the callback.

        Called by navigate_to (first attempt) AND by _on_nav_done (bisect
        retries). Same Nav2 plumbing in both cases — only the (x, y) differs.
        """
        if not self._nav_client.wait_for_server(timeout_sec=2.0):
            return {"status": "failed", "reason": "Nav2 action server unavailable"}

        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(x)
        goal.pose.pose.position.y = float(y)
        goal.pose.pose.position.z = 0.0
        goal.pose.pose.orientation.w = 1.0  # identity quaternion

        send_future = self._nav_client.send_goal_async(goal)

        deadline = time.time() + 5.0
        while not send_future.done() and time.time() < deadline:
            time.sleep(0.05)
        if not send_future.done():
            return {"status": "failed", "reason": "send_goal timed out (Nav2 didn't acknowledge)"}

        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            return {"status": "failed", "reason": "Nav2 rejected goal"}

        self._nav_goal_handle = goal_handle
        self._nav_last_status = None
        self._nav_done_event.clear()
        goal_handle.get_result_async().add_done_callback(self._on_nav_done)

        self.set_phase("navigating")
        msg = (f"_send_nav_goal: target=({x:.2f}, {y:.2f}) "
               f"attempt={self._nav_attempt + 1}/{self._nav_max_attempts}")
        self.get_logger().info(msg)
        _air_log.info(msg)
        return {"status": "active", "target": {"x": float(x), "y": float(y)}}

    def _on_nav_done(self, future):
        """Result-future callback: drive the target/midpoint state machine.

        Logic:
          - target  succeeded  → done (we reached the real goal).
          - target  failed     → fire a midpoint (halfway between bot's CURRENT
                                 pose and target). That short drive grows the
                                 SLAM map.
          - midpoint succeeded → re-fire the target from new position; the map
                                 has expanded, target may now be reachable.
          - midpoint failed    → bot can't even make the easier midpoint; give
                                 up to avoid an infinite loop.
          - attempts hit cap   → give up regardless.
        """
        try:
            result = future.result()
            status = result.status if result is not None else GoalStatus.STATUS_UNKNOWN
        except Exception as e:
            self.get_logger().warn(f"_on_nav_done: result future raised: {e}")
            status = GoalStatus.STATUS_UNKNOWN

        self._nav_last_status = status
        self._nav_goal_handle = None
        status_str = self._format_nav_status(status).get("status", "unknown")
        kind = self._nav_attempt_kind

        # Canceled is terminal — user/system stopped us, don't retry.
        if status == GoalStatus.STATUS_CANCELED:
            self._finalize_nav(status_str)
            return

        succeeded = (status == GoalStatus.STATUS_SUCCEEDED)

        # Cap to avoid infinite loops if every step keeps failing in a way we
        # haven't anticipated. Catches pathological worlds; normal cases finish
        # well under this.
        self._nav_attempt += 1
        if self._nav_attempt >= self._nav_max_attempts:
            msg = f"_on_nav_done: max attempts ({self._nav_max_attempts}) reached; giving up"
            self.get_logger().info(msg); _air_log.info(msg)
            self._finalize_nav(status_str if succeeded else "failed")
            return

        if self._nav_original_goal is None:
            # Should not happen — finalize defensively.
            self._finalize_nav(status_str)
            return

        gx, gy = self._nav_original_goal

        # ---- success branch ----
        if succeeded:
            if kind == "target":
                # Done — we reached the real goal.
                self._finalize_nav("succeeded")
                return
            # midpoint succeeded → retry the real target from new pose.
            msg = (f"_on_nav_done: midpoint reached; retrying target=({gx:.2f},{gy:.2f}) "
                   f"attempt={self._nav_attempt + 1}/{self._nav_max_attempts}")
            self.get_logger().info(msg); _air_log.info(msg)
            self._nav_attempt_kind = "target"
            self._send_nav_goal(gx, gy)
            return

        # ---- failure branch ----
        if kind == "midpoint":
            # Even the midpoint failed — bot is genuinely stuck.
            msg = f"_on_nav_done: midpoint also failed; giving up"
            self.get_logger().info(msg); _air_log.info(msg)
            self._finalize_nav("failed")
            return

        # kind == "target" failed → compute midpoint from CURRENT pose so each
        # cycle starts from where the bot actually is, not a stale snapshot.
        try:
            t = self._tf_buffer.lookup_transform(
                "map", "base_footprint", rclpy.time.Time(),
                timeout=Duration(seconds=0.5),
            )
            rx, ry = t.transform.translation.x, t.transform.translation.y
        except Exception as e:
            msg = f"_on_nav_done: tf lookup failed ({e}); giving up"
            self.get_logger().info(msg); _air_log.info(msg)
            self._finalize_nav("failed")
            return

        mx = rx + (gx - rx) * 0.5
        my = ry + (gy - ry) * 0.5
        msg = (f"_on_nav_done: target failed at ({gx:.2f},{gy:.2f}); "
               f"bot=({rx:.2f},{ry:.2f}) → midpoint=({mx:.2f},{my:.2f})")
        self.get_logger().info(msg); _air_log.info(msg)
        self._nav_attempt_kind = "midpoint"
        self._send_nav_goal(mx, my)

    def _finalize_nav(self, status_str: str) -> None:
        """Terminal cleanup shared by success / cancel / exhausted-retries paths.
        Flips phase, sets the done event, pushes a nav_done event to the outer
        ring.
        """
        self.set_phase("idle")
        self._nav_done_event.set()
        self._event_queue.put(("nav_done", status_str))
        msg = (f"_finalize_nav: status={status_str} "
               f"final_target={self._nav_original_goal} "
               f"attempts_used={self._nav_attempt + 1}")
        self.get_logger().info(msg)
        _air_log.info(msg)

    # ---- phase accessors ----

    def get_phase(self) -> str:
        with self._phase_lock:
            return self._phase

    def set_phase(self, phase: str) -> None:
        """Mutate the coarse task phase. Logged so transitions are auditable."""
        with self._phase_lock:
            prev, self._phase = self._phase, phase
        if prev != phase:
            self.get_logger().info(f"phase: {prev} → {phase}")
            _air_log.info(f"phase: {prev} → {phase}")

    # approach() was merged into navigate_to(stop_distance=...) — see that
    # method. To restore the standalone tool, copy from git history.

    def list_checkpoints(self) -> dict:
        """Return all named checkpoints the bot knows about.

        Source: AIR_CHECKPOINTS env var, populated by gazebo.launch.py.
        """
        return {
            "checkpoints": [
                {"name": n, "x": x, "y": y}
                for n, (x, y) in self._checkpoints.items()
            ]
        }

    def go_to_checkpoint(self, name: str) -> dict:
        """Navigate to a named checkpoint. Thin wrapper over navigate_to().

        Returns a 'failed' dict (without contacting Nav2) if the name is
        unknown, so the LLM can recover by calling list_checkpoints() or
        ask_user() instead of waiting on a timeout.
        """
        if name not in self._checkpoints:
            return {
                "status": "failed",
                "reason": (
                    f"unknown checkpoint {name!r}; "
                    f"known: {list(self._checkpoints) or '(none)'}"
                ),
            }
        x, y = self._checkpoints[name]
        self.get_logger().info(f"go_to_checkpoint: {name} → ({x:.2f}, {y:.2f})")
        return self.navigate_to([[x, y]])

    def check_nav_status(self, wait_seconds: float = 60.0) -> dict:
        """Block up to wait_seconds for the active nav goal to terminate, then
        return its status. If no goal is active, returns immediately with the
        last cached terminal status (or 'idle' if none).

        States returned: idle | active | succeeded | failed | canceled.

        wait_seconds:
          - 60 (default) — happy case for a multi-metre drive. ~1 LLM call total.
          - small (e.g. 2-3) — when the LLM wants to interleave actions or check
            on the user; returns 'active' on timeout, LLM can re-call.
          - 0 — pure poll, no wait (matches old behaviour).
        """
        # Already terminal (callback fired before we got here, or never started)
        # → return cached status without waiting.
        if self._nav_goal_handle is None:
            return self._format_nav_status(self._nav_last_status)

        # Block on the done-event. The result-future callback (_on_nav_done)
        # sets it the instant Nav2 reports terminal — zero polling, wakes
        # immediately. Returns False on timeout.
        woke = self._nav_done_event.wait(wait_seconds)
        if not woke:
            return {"status": "active"}

        return self._format_nav_status(self._nav_last_status)

    def _format_nav_status(self, status: Optional[int]) -> dict:
        """Map a cached GoalStatus int to the coarse string the LLM expects."""
        if status is None:
            return {"status": "idle"}
        if status == GoalStatus.STATUS_SUCCEEDED:
            return {"status": "succeeded"}
        if status == GoalStatus.STATUS_CANCELED:
            return {"status": "canceled"}
        return {"status": "failed"}  # aborted / unknown / anything else terminal

    def pick_up(self, object_label: str = "object") -> dict:
        """Scripted-pose pickup. Sends a fixed arm trajectory + closes gripper.

        No IK / MoveIt — pick_up is a canned routine assuming the object is on
        the floor ~15 cm ahead of the bot. Sequence:
          1. Move arm to a pre-grasp pose (extended forward, low).
          2. Open gripper.
          3. Lower the wrist to the floor.
          4. Close gripper.
          5. Lift back to a stow pose.

        Returns {"status": "success"} / {"status": "failed", "reason": ...}.
        Replace with IK-driven planning once MoveGroup is wired up.
        """
        self.get_logger().info(f"pick_up: scripted routine for {object_label!r}")

        # Joint poses (joint1..4) in radians. Tuned for OpenManipulator-X
        # reaching forward to a floor-level object directly in front of the bot.
        # Adjust if the gripper isn't landing where you expect.
        POSE_HOME      = [0.0,  0.0,  0.0,  0.0]
        POSE_PRE_GRASP = [0.0,  0.8, -0.4, -0.4]   # extended low forward
        POSE_GRASP     = [0.0,  1.3, -0.8, -0.5]   # wrist near floor
        POSE_LIFT      = [0.0,  0.4,  0.0,  0.0]   # raised, holding

        if not self._arm_client.wait_for_server(timeout_sec=2.0):
            return {"status": "failed", "reason": "arm action server unavailable"}
        if not self._gripper_client.wait_for_server(timeout_sec=2.0):
            return {"status": "failed", "reason": "gripper action server unavailable"}

        try:
            self._send_arm_pose(POSE_PRE_GRASP, duration_s=2.0)
            self._send_gripper(0.019)                       # open (max ~0.019 rad)
            self._send_arm_pose(POSE_GRASP, duration_s=1.5)
            self._send_gripper(-0.01)                       # close on object
            self._send_arm_pose(POSE_LIFT, duration_s=1.5)
        except Exception as e:
            return {"status": "failed", "reason": f"{type(e).__name__}: {e}"}

        self.set_phase("holding")
        return {"status": "success", "object": object_label}

    def _send_arm_pose(self, positions: list, duration_s: float = 2.0) -> None:
        """Blocking send of a single-waypoint joint trajectory.
        Raises on rejection / failure so pick_up can short-circuit cleanly.
        """
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(self._arm_joint_names)

        pt = JointTrajectoryPoint()
        pt.positions = [float(p) for p in positions]
        secs = int(duration_s)
        nsecs = int((duration_s - secs) * 1e9)
        pt.time_from_start = DurationMsg(sec=secs, nanosec=nsecs)
        goal.trajectory.points = [pt]

        self.get_logger().info(f"arm → {positions} over {duration_s}s")
        send_fut = self._arm_client.send_goal_async(goal)
        deadline = time.time() + 5.0
        while not send_fut.done() and time.time() < deadline:
            time.sleep(0.05)
        if not send_fut.done():
            raise RuntimeError("arm send_goal timed out")
        handle = send_fut.result()
        if handle is None or not handle.accepted:
            raise RuntimeError("arm goal rejected")

        # Wait for completion.
        result_fut = handle.get_result_async()
        deadline = time.time() + duration_s + 5.0
        while not result_fut.done() and time.time() < deadline:
            time.sleep(0.05)
        if not result_fut.done():
            raise RuntimeError("arm trajectory result timed out")

    def _send_gripper(self, position: float) -> None:
        """Blocking send to the gripper. position: ~+0.019 open, ~-0.01 closed."""
        goal = GripperCommand.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = 2.0

        self.get_logger().info(f"gripper → {position}")
        send_fut = self._gripper_client.send_goal_async(goal)
        deadline = time.time() + 3.0
        while not send_fut.done() and time.time() < deadline:
            time.sleep(0.05)
        if not send_fut.done():
            raise RuntimeError("gripper send_goal timed out")
        handle = send_fut.result()
        if handle is None or not handle.accepted:
            raise RuntimeError("gripper goal rejected")

        result_fut = handle.get_result_async()
        deadline = time.time() + 3.0
        while not result_fut.done() and time.time() < deadline:
            time.sleep(0.05)
        if not result_fut.done():
            raise RuntimeError("gripper result timed out")

    def ask_user(self, question: str) -> dict:
        """Publish a question, block until /agent/user comes back (or timeout).

        While blocked, `_expecting_answer` is True — the /agent/user subscriber
        routes the next message to this call instead of the outer-ring event
        queue. Cleared on exit so subsequent unprompted messages go to the
        queue normally.

        Returns:
            {"answer": "<user reply>"}  on success
            {"error": "timeout after Xs"} if no reply within ask_timeout_s
        """
        timeout = float(self.get_parameter("ask_timeout_s").value)

        # Reset state under the lock so a stale answer from a previous call
        # can't satisfy this one.
        with self._answer_lock:
            self._latest_answer = None
            self._answer_event.clear()

        self._expecting_answer = True
        try:
            msg = String()
            msg.data = question
            self._question_pub.publish(msg)
            self.get_logger().info(f"asked user: {question!r}  (timeout {timeout}s)")

            if not self._answer_event.wait(timeout=timeout):
                return {"error": f"timeout after {timeout}s waiting for /agent/user"}

            with self._answer_lock:
                answer = self._latest_answer or ""
            return {"answer": answer}
        finally:
            self._expecting_answer = False

    # ---- scan-only loop (LLM disabled) ----

    def run_scan_only_loop(self, period_s: float = 5.0):
        """Periodically run scan_scene() and log the result. No LLM, no tokens.

        Used when AIR_LLM_ENABLED=0. Lets you verify the camera + YOLO path in
        isolation while you're still iterating on Gazebo models / lighting.

        Also fires pick_up() once at startup so the arm trajectory can be
        eyeballed without involving the LLM. Failure is logged but doesn't
        crash the loop — sim might not have the arm controllers ready yet.
        """
        self.get_logger().info(f"scan-only loop (LLM disabled) — every {period_s}s.")
        _air_log.info(f"scan-only loop started (period={period_s}s)")

        # One-shot arm-sweep test. Give the action servers a moment to come up.
        time.sleep(2.0)
        self.get_logger().info("scan-only: firing pick_up() once for arm test...")
        try:
            res = self.pick_up("test_object")
            self.get_logger().info(f"scan-only: pick_up returned {res}")
            _air_log.info(f"scan-only pick_up result: {res}")
        except Exception as e:
            self.get_logger().warn(f"scan-only: pick_up raised {type(e).__name__}: {e}")
            _air_log.warning(f"scan-only pick_up exception: {e}")
        while rclpy.ok():
            result = self.scan_scene()
            if "error" in result:
                self.get_logger().warn(f"scan_scene: {result['error']}")
                _air_log.warning(f"scan_scene error: {result['error']}")
            else:
                dets = result.get("detections", [])
                if dets:
                    summary = ", ".join(
                        f"{d['label']}({d['confidence']:.2f})" for d in dets
                    )
                    self.get_logger().info(f"detections: {summary}")
                    _air_log.info(f"detections: {summary}  raw={dets}")
                else:
                    self.get_logger().info("no detections")
                    _air_log.info("detections: (none)")
            # Sleep in small slices so Ctrl-C / rclpy.shutdown wakes us promptly.
            slept = 0.0
            while rclpy.ok() and slept < period_s:
                time.sleep(0.2)
                slept += 0.2

    # ---- top-level interactive loop ----

    def run_interactive_loop(self):
        """Event-driven outer ring. Blocks on the event queue; only invokes the
        LLM when something happens (user message on /agent/user, or Nav2 done).

        Lifecycle:
          1. Wait for first event. If it's a 'user' event, start a session
             with [SystemMessage, HumanMessage(text)] as history. nav_done
             events with no active session are dropped (nothing to react to).
          2. Call agent.run_step(history). The graph runs think → tool → ...
             until it either:
               (a) FINISHES — a natural-language reply, no further tools. Publish
                   it on /agent/response and reset the session.
               (b) YIELDS — fired a nav-firing tool (navigate_to / approach /
                   go_to_checkpoint). Keep history, go back to waiting on the
                   event queue. Next nav_done or user message resumes.
          3. Loop.

        Type 'quit' / 'exit' / 'q' to exit cleanly. Exceptions inside the graph
        surface as the reply, not a node crash.
        """
        try:
            from agent.agent import run_step as run_agent_step
            from langchain_core.messages import HumanMessage, SystemMessage
            from agent.prompts import SYSTEM_PROMPT
        except Exception as e:
            err = (f"could not import agent.agent.run_step "
                   f"({type(e).__name__}: {e}); the LLM loop is disabled. "
                   "Node will idle.")
            self.get_logger().error(err)
            _air_log.error(err)
            threading.Event().wait()
            return

        self.get_logger().info(
            "interactive loop ready — listening on /agent/user."
        )
        _air_log.info("interactive loop ready.")

        history: list = []  # SystemMessage + running turns; reset between tasks

        while rclpy.ok():
            try:
                ev_type, ev_payload = self._event_queue.get(timeout=1.0)
            except queue.Empty:
                continue  # lets us notice rclpy shutdown promptly

            # --- translate event → next message for the LLM ---
            if ev_type == "user":
                text = (ev_payload or "").strip()
                if not text:
                    continue
                if text.lower() in ("quit", "exit", "q"):
                    self.get_logger().info("user exited interactive loop.")
                    _air_log.info("user exited interactive loop.")
                    return
                if not history:
                    history = [SystemMessage(content=SYSTEM_PROMPT)]
                    self.get_logger().info(f"new session: {text!r}")
                    _air_log.info(f"new session: {text!r}")
                else:
                    self.get_logger().info(f"user (mid-session): {text!r}")
                    _air_log.info(f"user (mid-session): {text!r}")
                history.append(HumanMessage(content=text))

            elif ev_type == "nav_done":
                if not history:
                    # Nav finished but no live session — likely the LLM had
                    # already wrapped up (final_reply published) before the
                    # callback fired. Nothing to react to.
                    self.get_logger().info(
                        f"nav_done={ev_payload} ignored (no active session)"
                    )
                    continue
                # Inject as a HumanMessage tagged [system] so the LLM treats
                # it as an observation rather than a user instruction. Using
                # SystemMessage mid-conversation confuses some model APIs.
                note = f"[system] navigation finished: {ev_payload}"
                history.append(HumanMessage(content=note))
                self.get_logger().info(f"nav_done injected: {ev_payload}")
                _air_log.info(f"nav_done injected: {ev_payload}")

            else:
                self.get_logger().warn(f"unknown event type: {ev_type!r}")
                continue

            # --- run one graph step ---
            try:
                history, reply, yielded = run_agent_step(history)
            except Exception as e:
                err = f"[error] {type(e).__name__}: {e}"
                self.get_logger().error(err)
                _air_log.exception("agent.run_step failed")
                out = String(); out.data = err
                self._response_pub.publish(out)
                history = []
                continue

            # --- publish + decide whether to reset session ---
            if reply:
                self.get_logger().info(f"[agent reply] {reply}")
                _air_log.info(f"reply: {reply}")
                out = String(); out.data = reply
                self._response_pub.publish(out)

            if not yielded:
                # Task done (graph returned a final reply, no more tools queued).
                # Drop history so the next user message starts fresh.
                history = []

    def shutdown(self):
        try:
            self._executor.shutdown()
        except Exception:
            pass
        try:
            self.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


def get_node() -> AgentNode:
    """Lazy singleton. First call constructs the node + starts the executor."""
    global _node_singleton
    with _lock:
        if _node_singleton is None:
            if not rclpy.ok():
                rclpy.init(args=sys.argv)
            _node_singleton = AgentNode()
        return _node_singleton


def shutdown_node():
    global _node_singleton
    with _lock:
        if _node_singleton is not None:
            _node_singleton.shutdown()
            _node_singleton = None


def main():
    """Entry point for `ros2 run air agent_node` and the launch file.

    Brings up the node (camera subs + YOLO) then enters the interactive loop:
    ask the user → run the LLM agent → publish reply → repeat. The 2s grace
    sleep gives camera subscriptions a chance to receive their first frame so
    the LLM's opening scan_scene() doesn't return 'no RGB frame yet'.
    """
    log_path = _setup_file_logging()
    node = get_node()
    if log_path:
        node.get_logger().info(f"file logging → {log_path}")
        _air_log.info(f"agent_node started; log file = {log_path}")
    node.get_logger().info("agent_node up. Warming up subscriptions...")
    try:
        time.sleep(2.0)
        # AIR_LLM_ENABLED=0 in .env (or shell) skips the LLM entirely and just
        # logs periodic YOLO scans. Default: enabled.
        llm_enabled = os.environ.get("AIR_LLM_ENABLED", "1") == "1"
        if llm_enabled:
            node.run_interactive_loop()
        else:
            node.get_logger().info("AIR_LLM_ENABLED=0 — running scan-only loop.")
            _air_log.info("LLM disabled via AIR_LLM_ENABLED=0; scan-only mode.")
            node.run_scan_only_loop()
    except KeyboardInterrupt:
        _air_log.info("KeyboardInterrupt — shutting down.")
    finally:
        _air_log.info("agent_node shutting down.")
        shutdown_node()
