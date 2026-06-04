# ResNet34 Training Data Structure

Tổ chức dữ liệu của bạn theo cấu trúc sau để training ResNet34:

```
data/
├── training/
│   ├── train/
│   │   ├── person/
│   │   │   ├── person_1.jpg
│   │   │   ├── person_2.jpg
│   │   │   └── ...
│   │   ├── vehicle/
│   │   │   ├── car_1.jpg
│   │   │   ├── car_2.jpg
│   │   │   └── ...
│   │   └── animal/
│   │       ├── dog_1.jpg
│   │       ├── cat_1.jpg
│   │       └── ...
│   └── val/
│       ├── person/
│       ├── vehicle/
│       └── animal/
```

## Hướng dẫn:

1. **Tạo thư mục**:

   ```bash
   mkdir -p data/training/train/{person,vehicle,animal}
   mkdir -p data/training/val/{person,vehicle,animal}
   ```

2. **Thêm ảnh**:
   - Đưa ảnh của mỗi lớp vào thư mục tương ứng
   - Chia ≈80% vào `train/`, ≈20% vào `val/`
   - Tối thiểu 50-100 ảnh per class để training tốt

3. **Chạy training**:

   ```bash
   python training/train_resnet34.py
   ```

   Mô hình sẽ được lưu tại: `models/resnet34_classifier.pth`

4. **Kiểm tra kết quả**: Mô hình sẽ tự động load khi app chạy
