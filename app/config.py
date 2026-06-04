"""Configuration and class mappings for object detection."""

# YOLOv8 COCO class to grouped category mapping
YOLO_CLASS_MAP = {
    "person": "person",
    "bicycle": "vehicle",
    "car": "vehicle",
    "motorcycle": "vehicle",
    "airplane": "vehicle",
    "bus": "vehicle",
    "train": "vehicle",
    "truck": "vehicle",
    "boat": "vehicle",
    "dog": "animal",
    "cat": "animal",
    "bird": "animal",
    "horse": "animal",
    "sheep": "animal",
    "cow": "animal",
    "elephant": "animal",
    "bear": "animal",
    "zebra": "animal",
    "giraffe": "animal",
    "cell phone": "phone",
}

# Target class colors
CLASS_COLORS = {
    "person": (0, 255, 0),      # Green
    "vehicle": (255, 0, 0),     # Blue
    "animal": (0, 0, 255),      # Red
    "phone": (0, 255, 255),     # Yellow / Cyan (BGR)
}

# Detection thresholds
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

# Available classes for UI selection
AVAILABLE_CLASSES = ["person", "vehicle", "animal", "phone"]

# Default target class
DEFAULT_CLASS = "person"
