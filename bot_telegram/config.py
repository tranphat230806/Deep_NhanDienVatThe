import socket
import platform
import psutil
import requests
import time
import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# Load environment variables from .env file
load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
BASE_URL = f"https://api.telegram.org/bot{TOKEN}"

# Throttle detection messages (seconds)
DETECTION_THROTTLE = int(os.getenv("DETECTION_THROTTLE", "5"))
_last_detection_time = {}


def get_system_info():
    """Lấy thông tin cấu hình máy."""
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    os_name = platform.system()
    os_version = platform.version()
    os_release = platform.release()
    cpu_count = psutil.cpu_count(logical=True)
    cpu_freq = psutil.cpu_freq().current if psutil.cpu_freq() else "Unknown"
    ram = psutil.virtual_memory().total / (1024**3)  # GB

    return (
        f"Thông tin cấu hình máy:\n"
        f"- Tên máy (Hostname): {hostname}\n"
        f"- Địa chỉ IP: {ip_address}\n"
        f"- Hệ điều hành: {os_name} {os_release} (Version: {os_version})\n"
        f"- CPU: {cpu_count} lõi, Tần số: {cpu_freq:.2f} MHz\n"
        f"- RAM: {ram:.2f} GB"
    )



def send_detection_message(class_name: str, confidence: float, count: int = 1):
    """
    Gửi thông báo detection qua Telegram.
    
    Args:
        class_name: Loại vật thể (person, vehicle, animal)
        confidence: Độ tin cậy (0-1)
        count: Số lượng vật thể phát hiện
    """
    try:
        # Throttle messages để tránh spam
        current_time = time.time()
        last_time = _last_detection_time.get(class_name, 0)
        
        if current_time - last_time < DETECTION_THROTTLE:
            return False
        
        _last_detection_time[class_name] = current_time
        
        # Xây dựng tin nhắn
        emoji_map = {
            "person": "👤",
            "vehicle": "🚗",
            "animal": "🐾"
        }
        emoji = emoji_map.get(class_name, "📦")
        
        message = (
            f"{emoji} <b>Phát hiện vật thể!</b>\n"
            f"<b>Loại:</b> {class_name.upper()}\n"
            f"<b>Độ tin cậy:</b> {confidence * 100:.1f}%\n"
            f"<b>Số lượng:</b> {count}"
        )
        
        # Gửi qua Telegram API
        url = f"{BASE_URL}/sendMessage"
        params = {
            "chat_id": CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        }
        
        response = requests.post(url, json=params, timeout=5)
        return response.status_code == 200
        
    except Exception as e:
        print(f"Lỗi gửi tin nhắn Telegram: {e}")
        return False


# Hàm xử lý lệnh /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Chào bạn! Gửi lệnh /info để nhận thông tin cấu hình máy của bạn."
    )


# Hàm xử lý lệnh /info
async def send_info(update: Update, context: ContextTypes.DEFAULT_TYPE):
    info = get_system_info()
    await update.message.reply_text(info)


def main():

    # Khởi tạo bot
    application = Application.builder().token(TOKEN).build()

    # Gắn các lệnh với hàm xử lý
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("info", send_info))

    # Bắt đầu bot
    print("Bot đang chạy...")
    application.run_polling()


if __name__ == "__main__":
    main()
