"""ResNet34 model for image classification."""

import torch
import torch.nn as nn
from torchvision import models, transforms
import os
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model path
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_PATH = MODEL_DIR / "resnet34_classifier.pth"
CLASSES_PATH = MODEL_DIR / "resnet34_classes.json"

# Class labels (loaded dynamically if exists)
CLASS_NAMES = ["person", "vehicle", "animal"]
if CLASSES_PATH.exists():
    try:
        import json
        with open(CLASSES_PATH, "r", encoding="utf-8") as f:
            CLASS_NAMES = json.load(f)
        print(f"[OK] Loaded dynamic classes: {CLASS_NAMES}")
    except Exception as e:
        print(f"[WARN] Error loading dynamic classes: {e}")

NUM_CLASSES = len(CLASS_NAMES)

# Image preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


class ResNet34Classifier:
    """ResNet34 classifier for person/vehicle/animal detection."""
    
    def __init__(self):
        self.model = None
        self.is_loaded = False
        self._load_model()
    
    def _load_model(self):
        """Load or create ResNet34 model."""
        try:
            self.model = models.resnet34(pretrained=True)
            # Replace last layer for 3 classes
            self.model.fc = nn.Linear(self.model.fc.in_features, NUM_CLASSES)
            
            # Try to load trained weights
            if MODEL_PATH.exists():
                self.model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
                self.is_loaded = True
                print(f"[OK] Loaded ResNet34 from {MODEL_PATH}")
            else:
                print(f"[WARN] ResNet34 weights not found at {MODEL_PATH}")
                print(f"   Run training first: python training/train_resnet34.py")
                self.is_loaded = False
            
            self.model.to(device)
            self.model.eval()
            
        except Exception as e:
            print(f"[FAIL] Error loading ResNet34: {e}")
            self.model = None
            self.is_loaded = False
    
    def predict(self, frame):
        """
        Classify a frame using ResNet34.
        
        Args:
            frame: OpenCV frame (numpy array, BGR format)
            
        Returns:
            Dict with class_name, confidence, and class_scores
        """
        if self.model is None or not self.is_loaded:
            return None
        
        try:
            # Preprocess frame
            input_tensor = transform(frame).unsqueeze(0).to(device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get predictions
            confidence, predicted = torch.max(probabilities, 1)
            class_id = predicted.item()
            class_name = CLASS_NAMES[class_id]
            confidence_score = confidence.item()
            
            # Get all class scores
            class_scores = {}
            for i, name in enumerate(CLASS_NAMES):
                class_scores[name] = probabilities[0, i].item()
            
            return {
                "class": class_name,
                "confidence": confidence_score,
                "class_scores": class_scores
            }
        
        except Exception as e:
            print(f"Error in ResNet34 prediction: {e}")
            return None


# Global model instance
resnet_classifier = ResNet34Classifier()


def predict(frame):
    """Public API to get ResNet34 prediction."""
    return resnet_classifier.predict(frame)
