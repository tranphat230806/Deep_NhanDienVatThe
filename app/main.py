from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
import cv2
from pathlib import Path

from app.camera import get_frame, set_target_class, get_target_class, get_available_cameras, set_camera, get_camera_index, get_resnet_result
from app.config import AVAILABLE_CLASSES

app = FastAPI(title="AI Webcam Detection")


# =====================
# MODEL ENDPOINTS
# =====================
@app.get("/api/models")
def get_model_status():
    """Get status of both models."""
    resnet_result = get_resnet_result()
    return {
        "yolo": "YOLOv8n (nano)",
        "resnet34": "Classification model" if resnet_result else "Not loaded",
        "last_resnet_prediction": resnet_result
    }


# =====================
# CAMERA ENDPOINTS
# =====================
@app.get("/api/cameras")
def list_cameras():
    """Get list of available cameras."""
    available = get_available_cameras()
    current = get_camera_index()
    return {"cameras": available, "current": current}


@app.post("/api/set-camera/{camera_index}")
def switch_camera(camera_index: int):
    """Switch to a different camera."""
    available = get_available_cameras()
    if camera_index not in available:
        raise HTTPException(
            status_code=400,
            detail=f"Camera {camera_index} not available. Available: {available}"
        )
    
    if set_camera(camera_index):
        return {"status": "ok", "camera": camera_index}
    else:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize camera {camera_index}"
        )


# =====================
# CONFIGURATION ENDPOINTS
# =====================
@app.get("/api/classes")
def get_available_classes():
    """Get available detection classes."""
    return {"classes": AVAILABLE_CLASSES, "current": get_target_class()}


@app.post("/api/set-class/{class_name}")
def set_detection_class(class_name: str):
    """Set target detection class."""
    if class_name not in AVAILABLE_CLASSES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid class. Choose from {AVAILABLE_CLASSES}"
        )
    set_target_class(class_name)
    return {"status": "ok", "class": class_name}


# =====================
# HOME PAGE
# =====================
@app.get("/")
def home():
    """Serve main UI with class selector."""
    templates_dir = Path(__file__).parent.parent / "templates"
    index_path = templates_dir / "index.html"

    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())

    # Fallback UI
    return HTMLResponse("""
    <html>
        <head>
            <title>AI Webcam Detection</title>
            <style>
                body {
                    margin: 0;
                    background: #1a1a1a;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    font-family: Arial, sans-serif;
                    color: white;
                }
                .container {
                    text-align: center;
                }
                img {
                    border: 3px solid #00ff88;
                    border-radius: 10px;
                    width: 800px;
                    margin-top: 20px;
                }
                select {
                    padding: 10px 15px;
                    font-size: 16px;
                    border-radius: 5px;
                    background: #333;
                    color: white;
                    border: 2px solid #00ff88;
                    cursor: pointer;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎥 Real-time Object Detection</h1>
                <label for="classSelect">Detect: </label>
                <select id="classSelect" onchange="setClass()">
                    <option value="person">Person</option>
                    <option value="vehicle">Vehicle</option>
                    <option value="animal">Animal</option>
                </select>
                <img src="/video" alt="Video Stream">
            </div>
            <script>
                function setClass() {
                    const selectedClass = document.getElementById('classSelect').value;
                    fetch(`/api/set-class/${selectedClass}`, { method: 'POST' });
                }
            </script>
        </body>
    </html>
    """)


# =====================
# VIDEO STREAMING
# =====================
def gen_frames():
    """Generate video frames for streaming."""
    while True:
        frame, _, _ = get_frame()

        if frame is None:
            continue

        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.get("/video")
def video():
    """Video stream endpoint."""
    return StreamingResponse(
        gen_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )