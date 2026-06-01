# 🎥 AI Webcam Object Detection

Real-time object detection system using YOLOv8 + FastAPI + OpenCV with web-based UI for class filtering.

## Features

- **YOLOv8 Detection**: High-speed object detection using YOLOv8n (nano)
- **Class Filtering**: Select 1 of 3 target classes
  - 👤 **Person**: Human detection
  - 🚗 **Vehicle**: Cars, trucks, motorcycles, buses
  - 🐾 **Animal**: Dogs, cats, birds, wildlife
- **Real-time Streaming**: MJPEG video stream via FastAPI
- **Web UI**: Interactive class selector with live detection
- **Optimized Performance**: Minimal latency, efficient buffer management

## Architecture

```
app/
├── main.py          # FastAPI app + streaming endpoints
├── camera.py        # Camera capture + frame processing
├── detector.py      # YOLOv8 wrapper for inference
└── config.py        # Class mappings + thresholds
```

## Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   First run will download YOLOv8 model (~35MB).

2. **Run application**:
   ```bash
   python run.py
   ```
   Or with uvicorn:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

3. **Access**: Open `http://localhost:8000` in browser

## API Endpoints

### GET `/`
Serves web UI with class selector

### GET `/video`
MJPEG video stream with real-time detections (unchanged from original)

### GET `/api/classes`
```json
{
  "classes": ["person", "vehicle", "animal"],
  "current": "person"
}
```

### POST `/api/set-class/{class_name}`
Set detection filter class
```bash
curl -X POST http://localhost:8000/api/set-class/vehicle
```

## Configuration

Edit [app/config.py](app/config.py) to:
- Adjust confidence threshold (default: 0.5)
- Change IOU threshold (default: 0.45)
- Modify class colors or mappings
- Switch YOLOv8 model (nano/small/medium)

## Performance

- **Model**: YOLOv8n (nano) - ~6ms inference
- **Stream**: 30 FPS MJPEG
- **Buffer**: Minimal (1 frame) for <100ms latency
- **GPU**: Auto-detects CUDA, falls back to CPU

## Requirements

- Python 3.8+
- Webcam
- CUDA-capable GPU (optional, CPU supported)

## Notes

- Original classifier.py kept for backward compatibility (unused)
- Thread-safe class state management
- Graceful stream recovery on disconnection
- Production-style error handling
