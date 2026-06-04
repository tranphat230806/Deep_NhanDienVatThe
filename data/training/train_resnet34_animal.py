"""Training script for ResNet34 classifier on animal dataset (Cat vs Dog)."""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from pathlib import Path
import cv2
from tqdm import tqdm

# Configuration
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 10  # Reduced epochs for quicker demo/validation, user can adjust
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
ANIMAL_DATA_DIR = Path(__file__).parent.parent / "data" / "training" / "train" / "animal"
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_PATH = MODEL_DIR / "resnet34_classifier.pth"
CLASSES_PATH = MODEL_DIR / "resnet34_classes.json"

# Create directories if needed
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def get_available_classes():
    """Detect available classes from animal folder."""
    if not ANIMAL_DATA_DIR.exists():
        return []
    
    classes = [d.name for d in ANIMAL_DATA_DIR.iterdir() if d.is_dir()]
    return sorted(classes)


class ImageDataset(Dataset):
    """Custom dataset for loading images from class subfolders."""
    
    def __init__(self, root_dir, classes, transform=None):
        """
        Args:
            root_dir: Directory containing subdirectories for each class
            classes: List of class names to use
            transform: Data augmentation transforms
        """
        self.root_dir = Path(root_dir)
        self.classes = classes
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load images from class folders
        for class_id, class_name in enumerate(classes):
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                print(f"[WARN] Warning: {class_dir} does not exist")
                continue
            
            image_count = 0
            corrupt_count = 0
            for img_path in class_dir.glob("*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                    # Robust check: attempt to read image to filter out corrupt ones
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        self.images.append(str(img_path))
                        self.labels.append(class_id)
                        image_count += 1
                    else:
                        corrupt_count += 1
            
            if corrupt_count > 0:
                print(f"  [OK] {class_name}: {image_count} images (skipped {corrupt_count} corrupt images)")
            else:
                print(f"  [OK] {class_name}: {image_count} images")
        
        print(f"\n[OK] Total loaded: {len(self.images)} images from {root_dir}")
        if len(self.images) == 0:
            raise ValueError(f"No images found in {root_dir}")
    
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


def create_model(num_classes):
    """Create ResNet34 model with N output classes."""
    model = models.resnet34(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
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
    print("ResNet34 Classifier Training (Animal Dataset)")
    print("=" * 60)
    
    # Get available classes
    classes = get_available_classes()
    
    if not classes:
        print(f"\n[FAIL] Error: No class directories found in {ANIMAL_DATA_DIR}")
        print("Please check that data/training/train/animal/Cat and data/training/train/animal/Dog exist.")
        return
    
    num_classes = len(classes)
    print(f"\n[Target] Target classes detected: {classes}")
    
    # Save the dynamic classes list to resnet34_classes.json
    try:
        with open(CLASSES_PATH, "w", encoding="utf-8") as f:
            json.dump(classes, f, ensure_ascii=False, indent=4)
        print(f"[OK] Saved class names to {CLASSES_PATH}")
    except Exception as e:
        print(f"[WARN] Warning: Failed to save class list: {e}")
    
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
    
    print("\nLoading animal dataset...")
    try:
        full_dataset = ImageDataset(ANIMAL_DATA_DIR, classes, transform=train_transform)
    except Exception as e:
        print(f"\n[FAIL] Error loading dataset: {e}")
        return
    
    # Automatically split into 80% train and 20% validation
    print("\nSplitting dataset (80% Train, 20% Val)...")
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Apply val transform to val_dataset
    val_dataset.dataset.transform = val_transform
    
    print(f"   Train samples: {train_size}")
    print(f"   Val samples:   {val_size}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"\nDevice: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epochs: {NUM_EPOCHS}")
    
    # Create model
    model = create_model(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
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
            print(f"[OK] Saved best model (Val Acc: {val_acc:.2f}%)\n")
    
    print("=" * 60)
    print(f"Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Classes: {classes}")
    print("=" * 60)


if __name__ == "__main__":
    main()
