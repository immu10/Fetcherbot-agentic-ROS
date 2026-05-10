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
import logging
import math
import os
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
from nav2_msgs.action import NavigateToPose
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

        # ---- ask_user: question/answer over std_msgs/String ----
        # Publisher fires the question, the UI (or another node, or a
        # `ros2 topic pub` from a human) replies on /agent/answer. We block
        # the calling tool thread on _answer_event until the answer arrives
        # or ask_timeout_s elapses.
        self._question_pub = self.create_publisher(String, "/agent/question", 10)
        self._response_pub = self.create_publisher(String, "/agent/response", 10)
        self.create_subscription(String, "/agent/answer", self._on_answer, 10)
        self._answer_event = threading.Event()
        self._answer_lock  = threading.Lock()
        self._latest_answer: Optional[str] = None

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

    def _on_answer(self, msg: String):
        """User reply on /agent/answer. Stash it and wake whoever's waiting."""
        with self._answer_lock:
            self._latest_answer = msg.data
        self._answer_event.set()

    def _on_camera_info(self, msg: CameraInfo):
        # Build the pinhole model once; intrinsics rarely change at runtime.
        if self._cam_model is None:
            model = PinholeCameraModel()
            model.fromCameraInfo(msg)
            with self._frame_lock:
                self._cam_model = model
                self._cam_frame_id = msg.header.frame_id or self._cam_frame_id
            self.get_logger().info(
                f"camera intrinsics received (frame_id={self._cam_frame_id})."
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

        try:
            result, _ = OD.detect_frame(self._yolo, rgb)
        except Exception as e:
            return {"error": f"YOLO inference failed: {type(e).__name__}: {e}"}

        names = result.names  # class_id -> label
        detections = []

        # ultralytics Results — boxes is None when nothing detected.
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return {"detections": []}

        xyxy  = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss  = boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, clss):
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            label = names.get(int(cls_id), str(cls_id)) if isinstance(names, dict) else names[int(cls_id)]

            position = self._project_to_ground(cx, cy, cam_model, cam_frame, "map")

            detections.append({
                "label": label,
                "confidence": float(conf),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "position": position,            # {x,y,z=0} in map frame, or None
                "frame_id": "map",
            })

        return {"detections": detections}

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

    def navigate_to(self, x: float, y: float) -> dict:
        """Send a NavigateToPose goal in `map` frame; return immediately.

        The LLM polls completion via check_nav_status() — this method only
        confirms the goal was accepted (status='active'), or returns a
        descriptive failure if Nav2 isn't up / rejected the goal.

        Orientation is identity (face +X). Future enhancement: face the goal
        by reading current robot pose from tf and computing yaw.
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

        # The executor thread is already spinning; just wait on the future.
        deadline = time.time() + 5.0
        while not send_future.done() and time.time() < deadline:
            time.sleep(0.05)
        if not send_future.done():
            return {"status": "failed", "reason": "send_goal timed out (Nav2 didn't acknowledge)"}

        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            return {"status": "failed", "reason": "Nav2 rejected goal"}

        # Stash for check_nav_status() to read.
        self._nav_goal_handle = goal_handle
        self._nav_last_status = None
        self.get_logger().info(f"navigate_to: goal accepted target=({x:.2f}, {y:.2f})")
        return {"status": "active", "target": {"x": float(x), "y": float(y)}}

    def check_nav_status(self) -> dict:
        """Map the current Nav2 goal state to a coarse status string for the LLM.

        States returned: idle | active | succeeded | failed | canceled.
        Until navigate_to is implemented, _nav_goal_handle stays None and we
        report 'idle' (or the last terminal status if a goal already finished).
        """
        gh = self._nav_goal_handle
        if gh is not None:
            # Active goal: query its current status.
            status = gh.status
            if status in (GoalStatus.STATUS_ACCEPTED, GoalStatus.STATUS_EXECUTING):
                return {"status": "active"}
            # Goal finished — clear the handle so future calls report from the
            # cached terminal status instead of re-reading a stale handle.
            self._nav_last_status = status
            self._nav_goal_handle = None

        last = self._nav_last_status
        if last is None:
            return {"status": "idle"}
        if last == GoalStatus.STATUS_SUCCEEDED:
            return {"status": "succeeded"}
        if last == GoalStatus.STATUS_CANCELED:
            return {"status": "canceled"}
        # Aborted, unknown, or anything else terminal → failed.
        return {"status": "failed"}

    def pick_up(self, object_label: str) -> dict:
        raise NotImplementedError("pick_up: wire to MoveIt2 MoveGroup + GripperCommand.")

    def ask_user(self, question: str) -> dict:
        """Publish a question, block until /agent/answer comes back (or timeout).

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

        msg = String()
        msg.data = question
        self._question_pub.publish(msg)
        self.get_logger().info(f"asked user: {question!r}  (timeout {timeout}s)")

        if not self._answer_event.wait(timeout=timeout):
            return {"error": f"timeout after {timeout}s waiting for /agent/answer"}

        with self._answer_lock:
            answer = self._latest_answer or ""
        return {"answer": answer}

    # ---- scan-only loop (LLM disabled) ----

    def run_scan_only_loop(self, period_s: float = 5.0):
        """Periodically run scan_scene() and log the result. No LLM, no tokens.

        Used when AIR_LLM_ENABLED=0. Lets you verify the camera + YOLO path in
        isolation while you're still iterating on Gazebo models / lighting.
        """
        self.get_logger().info(f"scan-only loop (LLM disabled) — every {period_s}s.")
        _air_log.info(f"scan-only loop started (period={period_s}s)")
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
        """Ask → run agent → publish reply → repeat. Blocks the calling thread.

        Triggered from main() once the node is up. The first prompt is broadcast
        on /agent/question so a UI (or a `ros2 topic pub` from a human) can
        reply on /agent/answer. Each completed run's text reply is published on
        /agent/response. Type 'quit' / 'exit' / 'q' to exit cleanly.

        Every exception inside agent.run() is caught and reported as the reply
        — a bad GROQ_API_KEY surfaces as a one-line error, not a node crash.
        """
        try:
            from agent.agent import run as run_agent
            from agent.prompts import INTERACTIVE_PROMPT
        except Exception as e:
            err = f"could not import agent.agent.run ({type(e).__name__}: {e}); the LLM loop is disabled. Node will idle."
            self.get_logger().error(err)
            _air_log.error(err)
            threading.Event().wait()
            return

        self.get_logger().info("interactive loop ready — awaiting first command.")
        _air_log.info("interactive loop ready.")
        while rclpy.ok():
            ask = self.ask_user(INTERACTIVE_PROMPT)
            if "error" in ask:
                # Most likely an ask_timeout. Loop and re-prompt so the user
                # has another chance instead of dying silently.
                msg = f"ask_user: {ask['error']}; re-prompting."
                self.get_logger().warn(msg)
                _air_log.warning(msg)
                continue

            command = (ask.get("answer") or "").strip()
            if not command:
                continue
            if command.lower() in ("quit", "exit", "q"):
                self.get_logger().info("user exited interactive loop.")
                _air_log.info("user exited interactive loop.")
                return

            self.get_logger().info(f"running agent on: {command!r}")
            _air_log.info(f"command: {command!r}")
            try:
                reply = run_agent(command)
            except Exception as e:
                reply = f"[error] {type(e).__name__}: {e}"
                self.get_logger().error(reply)
                _air_log.exception("agent.run failed")

            self.get_logger().info(f"[agent reply] {reply}")
            _air_log.info(f"reply: {reply}")
            out = String()
            out.data = reply
            self._response_pub.publish(out)

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
