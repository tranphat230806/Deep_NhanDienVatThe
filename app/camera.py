import cv2
import threading
from app.detector import detect
from app.resnet_model import predict as resnet_predict
from app.config import CLASS_COLORS, DEFAULT_CLASS
from bot_telegram.config import send_detection_message, send_detection_image

# Camera state management
_state_lock = threading.Lock()
_camera_index = 0
_target_classes = {DEFAULT_CLASS}  # Set of active target classes
_camera = None
_resnet_result = None
_frame_count = 0
_resnet_interval = 5  # Run ResNet34 every 5 frames to save compute


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


def set_target_classes(classes_list: list):
    """Set the active target classes for detection filtering."""
    global _target_classes
    with _state_lock:
        _target_classes = set(classes_list)


def get_target_classes() -> list:
    """Get the list of current active target classes."""
    with _state_lock:
        return list(_target_classes)


def set_target_class(class_name: str):
    """Set a single target class (backwards compatibility)."""
    global _target_classes
    with _state_lock:
        _target_classes = {class_name}


def get_target_class() -> str:
    """Get the first active target class (backwards compatibility)."""
    with _state_lock:
        return list(_target_classes)[0] if _target_classes else DEFAULT_CLASS


def get_resnet_result():
    """Get last ResNet34 classification result."""
    with _state_lock:
        return _resnet_result


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
        if "sub_class" in det:
            label = f"{det['sub_class']} {det['resnet_confidence'] * 100:.1f}%"
        else:
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
    """Capture frame and apply YOLOv8 detection + ResNet34 classification."""
    global _camera, _frame_count, _resnet_result
    
    if _camera is None or not _camera.isOpened():
        return None, None, None
    
    success, frame = _camera.read()

    if not success:
        return None, None, None

    # Run YOLOv8 detection
    detections = detect(frame)

    # For animal and phone detections, crop bounding box and pass to ResNet
    for d in detections:
        if d["grouped_class"] in ["animal", "phone"]:
            x1, y1, x2, y2 = map(int, d["xyxy"])
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]
                resnet_res = resnet_predict(crop)
                if resnet_res:
                    d["sub_class"] = resnet_res["class"]
                    d["resnet_confidence"] = resnet_res["confidence"]
                    with _state_lock:
                        _resnet_result = resnet_res
                else:
                    # Fallback to YOLO class if ResNet is not loaded/trained
                    if d["class"] in ["dog", "cat"]:
                        d["sub_class"] = d["class"].capitalize()
                    elif d["grouped_class"] == "phone":
                        d["sub_class"] = "Phone"
                    else:
                        d["sub_class"] = "Animal"
                    d["resnet_confidence"] = d["confidence"]

    # Filter by active target classes
    targets = get_target_classes()
    filtered_detections = [
        d for d in detections if d["grouped_class"] in targets
    ]

    # Run generic ResNet34 classification on whole frame every N frames only if "animal" and "phone" are not active
    if "animal" not in targets and "phone" not in targets:
        _frame_count += 1
        if _frame_count % _resnet_interval == 0:
            resnet_result = resnet_predict(frame)
            if resnet_result:
                with _state_lock:
                    _resnet_result = resnet_result
    
    # Draw filtered detections
    if filtered_detections:
        frame = draw_detections(frame, filtered_detections)
        
        # Group detections by their category/sub-class to notify separately
        detections_by_category = {}
        for d in filtered_detections:
            if d["grouped_class"] in ["animal", "phone"]:
                category = d.get("sub_class", "Animal" if d["grouped_class"] == "animal" else "Phone")
            else:
                category = d["grouped_class"]
            
            detections_by_category[category] = detections_by_category.get(category, [])
            detections_by_category[category].append(d)
            
        for category, dets in detections_by_category.items():
            count = len(dets)
            avg_confidence = sum(d.get("resnet_confidence", d["confidence"]) for d in dets) / count
            send_detection_message(category, avg_confidence, count)
            send_detection_image(category, frame, avg_confidence, count)

    return frame, filtered_detections, _resnet_result


# Initialize camera on startup
initialize_camera(0)