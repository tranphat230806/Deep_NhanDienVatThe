"""Telegram Bot Runner - Point of entry for the bot."""

from telegram.ext import Application, CommandHandler
from bot_telegram.config import TOKEN, start, send_info


def main():
    """Start the bot."""
    if not TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN not found in .env file")
    
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

