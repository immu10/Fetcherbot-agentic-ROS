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
CONF = 0.10  # Sim objects (Gazebo textures + sim lighting) score lower than
             # real-world photos. 0.35 was killing legit detections; bump back
             # up if you start seeing too many false positives.
IOU = 0.5


def load_model(path: str = MODEL_PATH) -> YOLO:
    return YOLO(path)


def detect_frame(model: YOLO, frame):
    """Run detection on a single BGR frame. Returns (results, annotated_frame)."""
    results = model.predict(frame, conf=CONF, iou=IOU, verbose=False)
    return results[0], results[0].plot()


def extract_detections(result) -> list[dict]:
    """Pull a clean per-detection dict list out of an ultralytics Results object.

    Returns:
        [
            {
                "label":      "cup",
                "confidence": 0.87,
                "bbox":       [x1, y1, x2, y2],   # pixel coords
                "center":     [cx, cy],           # pixel center, convenience
                "cls_id":     41,
            },
            ...
        ]
        Empty list if no detections.

    Keeps all ultralytics-specific shape-juggling (boxes.xyxy.cpu().numpy(),
    names dict vs list, cls_id → label lookup) on THIS side so callers don't
    need to know YOLO internals. Pair with your own projection / drawing /
    filtering code on top.
    """
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    xyxy  = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss  = boxes.cls.cpu().numpy().astype(int)
    names = result.names  # dict[int, str] in modern ultralytics; list in old.

    out = []
    for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, clss):
        cls_id = int(cls_id)
        label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
        out.append({
            "label":      label,
            "confidence": float(conf),
            "bbox":       [float(x1), float(y1), float(x2), float(y2)],
            "center":     [int((x1 + x2) / 2), int((y1 + y2) / 2)],
            "cls_id":     cls_id,
        })
    return out


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
