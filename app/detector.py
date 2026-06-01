"""YOLOv8 object detector wrapper for efficient inference."""

import torch
from ultralytics import YOLO
from app.config import CONFIDENCE_THRESHOLD, IOU_THRESHOLD, YOLO_CLASS_MAP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load YOLOv8n (nano) for speed, use YOLOv8s/m for better accuracy
model = YOLO("yolov8n.pt")
model.to(device)


def detect(frame):
    """
    Run YOLOv8 detection on a frame.
    Returns list of detections with cls, conf, xyxy, and grouped_class.
    """
    results = model(
        frame,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        device=device,
        verbose=False
    )

    detections = []
    if results and len(results) > 0:
        result = results[0]
        boxes = result.boxes

        for box in boxes:
            class_id = int(box.cls.item())
            class_name = model.names[class_id]
            confidence = float(box.conf.item())
            xyxy = box.xyxy[0].cpu().numpy()

            # Map YOLO class to grouped category
            grouped_class = YOLO_CLASS_MAP.get(class_name, None)

            if grouped_class:
                detections.append({
                    "class": class_name,
                    "grouped_class": grouped_class,
                    "confidence": confidence,
                    "xyxy": xyxy  # [x1, y1, x2, y2]
                })

    return detections
