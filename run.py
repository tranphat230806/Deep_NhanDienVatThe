#!/usr/bin/env python
"""
AI Webcam Detection - FastAPI + YOLOv8 + OpenCV
Real-time object detection with class filtering

Run with: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

import uvicorn
import sys

if __name__ == "__main__":
    print("Starting AI Webcam Detection Server...")
    print("   YOLOv8n | FastAPI | OpenCV")
    print("   Visit: http://localhost:8000")
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
