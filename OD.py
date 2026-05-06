"""
Object detection with Ultralytics YOLO11.

This module is responsible for the *webcam* input path. Other sources
(Gazebo / Unreal / Unity sim cameras, ROS topics, RTSP, etc.) will live in
their own modules and feed frames into `detect_frame` / `run_on_source`.

Model: yolo11s.pt — latest Ultralytics YOLO, 'small' variant. Good
accuracy/speed balance, real-time on CPU. Swap to n/m/l/x as needed.

Install:  pip install ultralytics opencv-python
"""

import cv2
from ultralytics import YOLO

MODEL_PATH = "yolo11s.pt"
CONF = 0.35
IOU = 0.5


def load_model(path: str = MODEL_PATH) -> YOLO:
    return YOLO(path)


def detect_frame(model: YOLO, frame):
    """Run detection on a single BGR frame. Returns (results, annotated_frame)."""
    results = model.predict(frame, conf=CONF, iou=IOU, verbose=False)
    return results[0], results[0].plot()


def run_on_source(model: YOLO, source, window_name: str = "YOLO11"):
    """Generic loop over any cv2.VideoCapture-compatible source.
    `source` can be an int (webcam index), a video path, or an RTSP/HTTP URL.
    Sim camera modules (Gazebo/Unreal/Unity) can call `detect_frame` directly
    with their own frame producers instead of going through this helper.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            _, annotated = detect_frame(model, frame)
            cv2.imshow(window_name, annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Webcam entry point. Other input sources will have their own entry files.
    model = load_model()
    run_on_source(model, source=0, window_name="YOLO11 - webcam (q to quit)")
