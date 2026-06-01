import cv2
import threading
from app.detector import detect
from app.config import CLASS_COLORS, DEFAULT_CLASS
from bot_telegram.config import send_detection_message, send_detection_image

# Camera state management
_state_lock = threading.Lock()
_camera_index = 0
_target_class = DEFAULT_CLASS
_camera = None


def get_available_cameras(max_index=5):
    """Detect available cameras (0 to max_index-1)."""
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
        else:
            cap.release()
    return available


def initialize_camera(index: int):
    """Initialize camera with given index."""
    global _camera
    with _state_lock:
        if _camera is not None:
            _camera.release()
        
        _camera = cv2.VideoCapture(index)
        if _camera.isOpened():
            _camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
            return True
        else:
            _camera = None
            return False


def set_camera(index: int) -> bool:
    """Switch to a different camera."""
    global _camera_index
    if initialize_camera(index):
        with _state_lock:
            _camera_index = index
        return True
    return False


def get_camera_index() -> int:
    """Get current camera index."""
    with _state_lock:
        return _camera_index


def set_target_class(class_name: str):
    """Set the target class for detection filtering."""
    global _target_class
    with _state_lock:
        _target_class = class_name


def get_target_class() -> str:
    """Get the current target class."""
    with _state_lock:
        return _target_class


def draw_detections(frame, detections):
    """Draw bounding boxes, labels, and confidence on frame."""
    for det in detections:
        x1, y1, x2, y2 = map(int, det["xyxy"])
        grouped_class = det["grouped_class"]
        confidence = det["confidence"]

        # Get color for class
        color = CLASS_COLORS.get(grouped_class, (255, 255, 255))

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Prepare label
        label = f"{grouped_class} {confidence * 100:.1f}%"

        # Draw label background and text
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
        )
        cv2.rectangle(
            frame,
            (x1, y1 - text_h - baseline - 5),
            (x1 + text_w, y1),
            color,
            -1
        )
        cv2.putText(
            frame,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )

    return frame


def get_frame():
    """Capture frame and apply YOLOv8 detection with filtering."""
    global _camera
    
    if _camera is None or not _camera.isOpened():
        return None, None
    
    success, frame = _camera.read()

    if not success:
        return None, None

    # Run detection
    detections = detect(frame)

    # Filter by target class
    target = get_target_class()
    filtered_detections = [
        d for d in detections if d["grouped_class"] == target
    ]

    # Draw filtered detections
    if filtered_detections:
        frame = draw_detections(frame, filtered_detections)
        
        # Send Telegram notification
        count = len(filtered_detections)
        avg_confidence = sum(d["confidence"] for d in filtered_detections) / count
        send_detection_message(target, avg_confidence, count)
        
        # Send Telegram image (2 minute throttle)
        send_detection_image(target, frame, avg_confidence, count)

    return frame, filtered_detections


# Initialize camera on startup
initialize_camera(0)