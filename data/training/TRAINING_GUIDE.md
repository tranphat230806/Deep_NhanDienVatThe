# ResNet34 Training Guide

## Cấu Trúc Dự Án

```
Deep_NhanDienVatThe/
├── app/
│   ├── resnet_model.py       # ResNet34 model wrapper
│   └── ...
├── training/
│   ├── train_resnet34.py     # Training script
├── data/
│   └── training/
│       ├── train/
│       │   ├── person/       # 📁 Thêm ảnh vào đây
│       │   ├── vehicle/
│       │   └── animal/
│       └── val/
│           ├── person/
│           ├── vehicle/
│           └── animal/
├── models/
│   └── resnet34_classifier.pth  # Trained model (auto-saved)
```

## Hướng Dẫn Training

### 1. Chuẩn Bị Dữ Liệu

```bash
# Cấu trúc đã được tạo sẵn, bạn chỉ cần thêm ảnh:
data/training/train/person/     # ≈ 100+ ảnh
data/training/train/vehicle/    # ≈ 100+ ảnh
data/training/train/animal/     # ≈ 100+ ảnh
data/training/val/person/       # ≈ 20+ ảnh
data/training/val/vehicle/      # ≈ 20+ ảnh
data/training/val/animal/       # ≈ 20+ ảnh
```

### 2. Chạy Training

```bash
python training/train_resnet34.py
```

**Output:**

```
============================================================
ResNet34 Classifier Training
============================================================
Device: cuda (or cpu)
Batch size: 32
Learning rate: 0.001
Epochs: 50

Starting training...

Epoch 1/50
Training: 100%|████████| 10/10 [00:05<00:00, 0.50s/it]
Train Loss: 1.2345, Train Acc: 45.23%
Val Loss: 1.1234, Val Acc: 52.10%

...

✓ Saved best model to models/resnet34_classifier.pth
Best validation accuracy: 85.50%
============================================================
```

### 3. Sử Dụng Model

Model sẽ tự động load khi app chạy:

```bash
python run.py
```

Sau đó truy cập `http://localhost:8000` để xem:

- **YOLOv8**: Phát hiện vật thể real-time
- **ResNet34**: Classification kết quả (cập nhật mỗi 5 frame)

## API Endpoints

```bash
# Lấy trạng thái 2 model
curl http://localhost:8000/api/models

# Response:
{
  "yolo": "YOLOv8n (nano)",
  "resnet34": "Classification model",
  "last_resnet_prediction": {
    "class": "person",
    "confidence": 0.95,
    "class_scores": {
      "person": 0.95,
      "vehicle": 0.04,
      "animal": 0.01
    }
  }
}
```

## Tùy Chỉnh Training

Mở `training/train_resnet34.py` để chỉnh:

```python
BATCH_SIZE = 32          # Kích thước batch (↑ = dùng RAM hơn)
LEARNING_RATE = 0.001    # Tốc độ học (↓ = học chậm hơn)
NUM_EPOCHS = 50          # Số epoch training
```

## Troubleshooting

**Q: "No images found in data/training"**

- Kiểm tra đã thêm ảnh vào các thư mục chưa?

**Q: "CUDA out of memory"**

- Giảm `BATCH_SIZE` từ 32 xuống 16

**Q: ResNet34 hiển thị "Loading..."**

- Kiểm tra mô hình đã training xong chưa (tệp lưu tại `models/resnet34_classifier.pth`)

## Note

- Training model lần đầu tiên sẽ mất thời gian (30-60 phút tuỳ GPU)
- Càng nhiều dữ liệu training → độ chính xác cao hơn
- ResNet34 chạy mỗi 5 frame để tối ưu performance
