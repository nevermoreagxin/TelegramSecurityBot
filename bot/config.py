import os
from dotenv import load_dotenv

# Визначаємо шлях до папки проекту (де лежать папки bot/ та ai_model/)
# Припускаємо, що config.py лежить у bot/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Завантажуємо змінні середовища з .env у корені проекту
dotenv_path = os.path.join(PROJECT_ROOT, '.env')
load_dotenv(dotenv_path)

# --- Основні Налаштування Бота ---
BOT_TOKEN = os.getenv('BOT_TOKEN')
if not BOT_TOKEN:
    print("КРИТИЧНА ПОМИЛКА: Токен бота ('BOT_TOKEN') не знайдено! Перевір .env або змінні середовища.")
    exit(1) # Вихід з помилкою
print("Токен бота завантажено.")

# --- Шляхи до Ресурсів ШІ (відносно кореня проекту) ---
print("Визначення шляхів до ресурсів ШІ...")
AI_MODEL_DIR = os.path.join(PROJECT_ROOT, 'ai_model') # Папка з кодом та ресурсами Ксенії

# Переконайся, що імена файлів ТОЧНО відповідають тим, що у папці ai_model/
DB_PATH = os.path.join(AI_MODEL_DIR, 'ipso_database_binary.db')
MODEL_WEIGHTS_DIR = os.path.join(AI_MODEL_DIR, 'model_weights_binary') # Ім'я папки з вагами
MODEL_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, 'ipso_rnn_binary_lstm_30k.weights.h5') # Ім'я файлу ваг
TOKENIZER_PATH = os.path.join(AI_MODEL_DIR, 'tokenizer_binary_30k.json') # Ім'я файлу токенізатора
LABEL_ENCODER_PATH = os.path.join(AI_MODEL_DIR, 'label_encoder_binary.pkl') # Ім'я файлу енкодера

print(f"  DB_PATH: {DB_PATH}")
print(f"  MODEL_WEIGHTS_PATH: {MODEL_WEIGHTS_PATH}")
print(f"  TOKENIZER_PATH: {TOKENIZER_PATH}")
print(f"  LABEL_ENCODER_PATH: {LABEL_ENCODER_PATH}")

# --- Параметри Моделі (Тільки ті, що потрібні боту напряму) ---
MAX_LENGTH = 200 # Має відповідати параметру навчання моделі
print(f"MAX_LENGTH для паддінгу: {MAX_LENGTH}")

# Інші параметри (VOCAB_SIZE, EMBEDDING_DIM, etc.) тепер використовуються всередині ai_model/ipso_model.py