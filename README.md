# 🎥 Nhận Diện Vật Thể Từ Webcam

Hệ thống nhận diện vật thể real-time sử dụng YOLOv8 + FastAPI + OpenCV với giao diện web để lọc lớp vật thể.

## Tính Năng

- **YOLOv8 Detection**: Nhận diện vật thể tốc độ cao với YOLOv8n (nano)
- **Lọc Lớp Vật Thể**: Chọn 1 trong 3 lớp mục tiêu
  - 👤 **Người**: Phát hiện con người
  - 🚗 **Xe Cộ**: Ô tô, xe tải, xe máy, xe buýt
  - 🐾 **Động Vật**: Chó, mèo, chim, động vật hoang dã
- **Truyền Tải Real-time**: Luồng video MJPEG qua FastAPI
- **Giao Diện Web**: Bộ chọn lớp tương tác với nhận diện trực tiếp
- **Hiệu Suất Tối Ưu**: Độ trễ tối thiểu, quản lý bộ đệm hiệu quả
- **Thông Báo Telegram**: Tự động gửi tin nhắn khi phát hiện vật thể

## Kiến Trúc

```
app/
├── main.py          # Ứng dụng FastAPI + điểm cuối truyền tải
├── camera.py        # Chụp camera + xử lý khung hình
├── detector.py      # Bộ bao bọc YOLOv8 để suy luận
└── config.py        # Ánh xạ lớp + ngưỡng

bot_telegram/
├── bot.py           # Điểm vào chính của bot
└── config.py        # Cấu hình bot + hàm tiện ích
```

## Cài Đặt

### 1. Cài Đặt Thư Viện

```bash
pip install -r requirements.txt
```

Lần chạy đầu tiên sẽ tải xuống mô hình YOLOv8 (~35MB).

### 2. Cấu Hình Telegram Bot (Tùy Chọn)

Để nhận thông báo khi phát hiện vật thể, bạn cần:

1. **Tạo Bot Telegram**:
   - Mở Telegram và tìm `@BotFather`
   - Gửi lệnh `/newbot` và tuân theo hướng dẫn
   - Sao chép `BOT_TOKEN` nhận được

2. **Lấy Chat ID**:
   - Tìm bot bạn vừa tạo (search theo tên bot)
   - Gửi tin nhắn `/start`
   - Truy cập: `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
   - Sao chép `id` từ field `chat.id` trong phản hồi JSON

3. **Cấu Hình `.env`**:
   - Copy file `.env.example` thành `.env`:
   ```bash
   cp .env.example .env
   ```

   - Mở `.env` và thêm:
   ```env
   TELEGRAM_BOT_TOKEN=your_bot_token_here
   TELEGRAM_CHAT_ID=your_chat_id_here
   DETECTION_THROTTLE=5
   CONFIDENCE_THRESHOLD=0.5
   IOU_THRESHOLD=0.45
   ```

**Lưu ý**: File `.env` sẽ bị gitignore, hãy giữ token của bạn an toàn!

### 3. Chạy Ứng Dụng

**Cách 1** (đơn giản):

```bash
python run.py
```

**Cách 2** (với uvicorn):

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

3. **Truy Cập**: Mở `http://localhost:8000` trong trình duyệt

## Các Điểm Cuối API

### GET `/`

Phục vụ giao diện web với bộ chọn lớp

### GET `/video`

Luồng video MJPEG với các phát hiện real-time

### GET `/api/classes`

```json
{
  "classes": ["person", "vehicle", "animal"],
  "current": "person"
}
```

### POST `/api/set-class/{class_name}`

Đặt lớp lọc phát hiện

```bash
curl -X POST http://localhost:8000/api/set-class/vehicle
```

## Cấu Hình

Chỉnh sửa [app/config.py](app/config.py) để:

- Điều chỉnh ngưỡng tin cậy (mặc định: 0.5)
- Thay đổi ngưỡng IOU (mặc định: 0.45)
- Sửa đổi màu hoặc ánh xạ lớp
- Chuyển đổi mô hình YOLOv8 (nano/small/medium)

## Hiệu Suất

- **Mô Hình**: YOLOv8n (nano) - ~6ms suy luận
- **Luồng**: 30 FPS MJPEG
- **Bộ Đệm**: Tối thiểu (1 khung hình) để độ trễ <100ms
- **GPU**: Tự động phát hiện CUDA, quay lại CPU nếu cần

## Yêu Cầu

- Python 3.8+
- Webcam
- GPU hỗ trợ CUDA (tùy chọn, CPU được hỗ trợ)
- Telegram Bot Token (tùy chọn, để nhận thông báo)

## Ghi Chú

- Classifier.py gốc được giữ lại để tương thích ngược (không sử dụng)
- Quản lý trạng thái lớp an toàn cho luồng
- Phục hồi luồng nhẹ nhàng khi ngắt kết nối
- Xử lý lỗi theo kiểu sản xuất
- Cấu hình Telegram Bot từ file `.env` để bảo mật
