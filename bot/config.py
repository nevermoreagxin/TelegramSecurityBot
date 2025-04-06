# bot/config.py
import os
from dotenv import load_dotenv

# Визначаємо шлях до папки проекту (де лежить папка bot)
project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Формуємо шлях до файлу .env
dotenv_path = os.path.join(project_folder, '.env')

if os.path.exists(dotenv_path):
    print(f"Завантаження конфігурації з: {dotenv_path}")
    load_dotenv(dotenv_path)
else:
    print(f"Попередження: Файл .env не знайдено за шляхом {dotenv_path}")

BOT_TOKEN = os.getenv('BOT_TOKEN')

if not BOT_TOKEN:
    print("КРИТИЧНА ПОМИЛКА: Токен бота ('BOT_TOKEN') не знайдено! Перевір файл .env.")
    exit() # Зупиняємо виконання, якщо токена немає

print("Токен бота успішно завантажено.")
# Тут можна додати інші налаштування в майбутньому
# MODEL_PATH_TEXT = os.path.join(project_folder, 'models', 'text_model')