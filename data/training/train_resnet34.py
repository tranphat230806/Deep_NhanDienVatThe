"""Training script for ResNet34 classifier."""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

# Configuration
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Cat", "Dog"]
NUM_CLASSES = len(CLASS_NAMES)

# Paths
# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data
DATA_DIR = PROJECT_ROOT / "data" / "training"

# Models
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "resnet34_classifier.pth"

# Create directories if needed
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


class ImageDataset(Dataset):
    """Custom dataset for loading images from folders."""
    
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir: Root directory with subdirectories for each class
            transform: Data augmentation transforms
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load images from class folders
        for class_id, class_name in enumerate(CLASS_NAMES):
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                print(f"⚠ Warning: {class_dir} does not exist")
                continue
            
            for img_path in class_dir.glob("*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                    self.images.append(str(img_path))
                    self.labels.append(class_id)
        
        print(f"✓ Loaded {len(self.images)} images from {data_dir}")
        if len(self.images) == 0:
            raise ValueError(
                f"No images found in {data_dir}\n"
                f"Expected structure:\n"
                f"  {data_dir}/\n"
                f"    person/\n"
                f"    vehicle/\n"
                f"    animal/"
            )
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


def create_model():
    """Create ResNet34 model with 3 output classes."""
    model = models.resnet34(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model.to(DEVICE)


def train_epoch(model, train_loader, criterion, optimizer):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for images, labels in pbar:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({"loss": loss.item(), "acc": correct/total})
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def main():
    """Main training function."""
    print("=" * 60)
    print("ResNet34 Classifier Training")
    print("=" * 60)
    
    # Data augmentation
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    print(f"\nLoading datasets from: {DATA_DIR}")
    
    # Load datasets
    try:
        train_dataset = ImageDataset(DATA_DIR / "train", transform=train_transform)
        val_dataset = ImageDataset(DATA_DIR / "val", transform=val_transform)
    except FileNotFoundError:
        print("\n✗ Error: Training data directories not found!")
        print(f"\nPlease organize your data as follows:")
        print(f"{DATA_DIR}/")
        print(f"  train/")
        print(f"    person/")
        print(f"    vehicle/")
        print(f"    animal/")
        print(f"  val/")
        print(f"    person/")
        print(f"    vehicle/")
        print(f"    animal/")
        return
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"\nDevice: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epochs: {NUM_EPOCHS}")
    
    # Create model
    model = create_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    best_val_acc = 0.0
    
    print("\nStarting training...\n")
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%\n")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"✓ Saved best model to {MODEL_PATH}\n")
    
    print("=" * 60)
    print(f"Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
