"""ResNet50 model for image classification (Cats vs Dogs)."""

import torch
import torch.nn as nn
from torchvision import models, transforms
import os
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model path
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_PATH = MODEL_DIR / "resnet50_catsdogs.pth"
CLASSES_PATH = MODEL_DIR / "resnet34_classes.json"

# Class labels (loaded dynamically if exists)
CLASS_NAMES = ["Cat", "Dog"]
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


class ResNet50Classifier:
    """ResNet50 classifier for Cat/Dog sub-classification."""
    
    def __init__(self):
        self.model = None
        self.is_loaded = False
        self._load_model()
    
    def _load_model(self):
        """Load or create ResNet50 model."""
        try:
            # Create ResNet50 structure without downloading weights
            self.model = models.resnet50(weights=None)
            # Replace last layer for NUM_CLASSES (usually 2: Cat, Dog)
            self.model.fc = nn.Linear(self.model.fc.in_features, NUM_CLASSES)
            
            # Load trained ResNet50 weights
            if MODEL_PATH.exists():
                self.model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
                self.is_loaded = True
                print(f"[OK] Loaded ResNet50 from {MODEL_PATH}")
            else:
                print(f"[WARN] ResNet50 weights not found at {MODEL_PATH}")
                self.is_loaded = False
            
            self.model.to(device)
            self.model.eval()
            
        except Exception as e:
            print(f"[FAIL] Error loading ResNet50: {e}")
            self.model = None
            self.is_loaded = False
    
    def predict(self, frame):
        """
        Classify a frame using ResNet50.
        
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
            print(f"Error in ResNet50 prediction: {e}")
            return None


# Global model instance
resnet_classifier = ResNet50Classifier()


def predict(frame):
    """Public API to get ResNet50 prediction."""
    return resnet_classifier.predict(frame)
