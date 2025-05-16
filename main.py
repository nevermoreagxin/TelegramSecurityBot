import telebot
import os
import sys
import sqlite3 # Потрібен для визначення типу db_conn та обробки помилок
from functools import partial # Потрібен для передачі аргументів в обробники

# Імпортуємо конфігурацію
try:
    from bot.config import (
        BOT_TOKEN, DB_PATH, MODEL_WEIGHTS_PATH, TOKENIZER_PATH,
        LABEL_ENCODER_PATH, MAX_LENGTH
    )
    print("Конфігурацію завантажено.")
except ImportError as e:
     print(f"КРИТИЧНА ПОМИЛКА: Помилка імпорту конфігурації з bot.config: {e}")
     sys.exit(1)

# Імпортуємо функції реєстрації обробників
try:
    from bot.handlers.common import register_common_handlers
    from bot.handlers.text import register_text_handlers
    print("Обробники імпортовано.")
except ImportError as e:
     print(f"КРИТИЧНА ПОМИЛКА: Помилка імпорту обробників з bot.handlers: {e}")
     sys.exit(1)

# Імпортуємо ТІЛЬКИ функцію завантаження з модуля ШІ
try:
    # Припускаємо, що файл називається ipso_model.py
    from ai_model.ipso_model import load_ai_resources
    print("Функцію завантаження ШІ 'load_ai_resources' імпортовано успішно.")
except ImportError as e:
    print(f"КРИТИЧНА ПОМИЛКА: Помилка імпорту 'load_ai_resources' з ai_model.ipso_model: {e}")
    print("Перевірте наявність файлу __init__.py в папці ai_model та правильність імені файлу/функції.")
    sys.exit(1)
except Exception as e:
    print(f"КРИТИЧНА ПОМИЛКА: Інша помилка під час імпорту ШІ модуля: {e}")
    sys.exit(1)

# --- Основний блок запуску ---
if __name__ == '__main__':
    print("Запуск main.py...")
    model = None
    tokenizer = None
    label_encoder = None
    db_conn = None # Важливо ініціалізувати як None для блоку finally
    optimal_threshold = 0.5 # Встановлюємо стандартний поріг

    try:
        # 1. Завантаження ВСІХ ресурсів ШІ + ініціалізація БД через єдину функцію
        print("Завантаження ресурсів ШІ та підключення до БД...")
        model, tokenizer, label_encoder, db_conn = load_ai_resources(
            model_weights_path=MODEL_WEIGHTS_PATH,
            tokenizer_path=TOKENIZER_PATH,
            encoder_path=LABEL_ENCODER_PATH,
            db_path=DB_PATH
        )

        # Перевірка, чи все завантажено
        if not all([model, tokenizer, label_encoder, db_conn]):
             # Функція load_ai_resources вже має вивести деталі помилки
             raise RuntimeError("Не вдалося завантажити всі компоненти ШІ або підключитися до БД.")
        print("Ресурси ШІ та з'єднання з БД готові.")

        # 2. Встановлення порогу (можна залишити 0.5 або розрахувати)
        print(f"Встановлено поріг класифікації: {optimal_threshold}")

        # 3. Ініціалізація бота
        print("Ініціалізація екземпляру бота...")
        # Можна додати parse_mode='HTML' за замовчуванням, якщо часто використовуєш
        bot = telebot.TeleBot(BOT_TOKEN, parse_mode='HTML')

        # 4. Реєстрація обробників
        print("Реєстрація обробників...")
        register_common_handlers(bot)
        # Передаємо завантажені об'єкти ШІ та АКТИВНЕ з'єднання з БД
        register_text_handlers(
            bot_instance=bot,
            model=model,
            tokenizer=tokenizer,
            label_encoder=label_encoder,
            db_conn=db_conn, # Передаємо об'єкт з'єднання
            max_length=MAX_LENGTH, # MAX_LENGTH з config.py
            threshold=optimal_threshold
        )
        print("Обробники зареєстровано.")

        # 5. Запуск бота
        print("-" * 30)
        print("Запуск бота (полінг)... Натисніть Ctrl+C для зупинки.")
        print("-" * 30)
        # num_threads > 1 за замовчуванням, interval=0 і timeout=30 хороші значення
        bot.polling(none_stop=True, interval=0, timeout=30)

    # Блоки обробки помилок залишаються такими ж, як у попередньому прикладі
    except FileNotFoundError as e:
        print(f"КРИТИЧНА ПОМИЛКА ЗАПУСКУ: Не знайдено необхідний файл ресурсів ШІ або БД. {e}")
    except ConnectionError as e:
        print(f"КРИТИЧНА ПОМИЛКА ЗАПУСКУ: Проблема з базою даних. {e}")
    except ImportError as e:
         print(f"КРИТИЧНА ПОМИЛКА ЗАПУСКУ: Не вдалося імпортувати необхідний модуль. {e}")
    except RuntimeError as e:
         print(f"КРИТИЧНА ПОМИЛКА ЗАПУСКУ: Проблема ініціалізації ресурсів. {e}")
    except Exception as e:
        print(f"КРИТИЧНА ПОМИЛКА під час виконання main.py: {e}")
        import traceback
        traceback.print_exc() # Виводимо детальний трейсбек
    finally:
        # Завжди закриваємо з'єднання з БД при виході
        if db_conn: # Перевіряємо, чи з'єднання було успішно створено
            try:
                db_conn.close()
                print("З'єднання з БД закрито.")
            except Exception as e_close:
                 print(f"Помилка закриття БД: {e_close}")
        print("Роботу бота завершено (або сталася критична помилка).")