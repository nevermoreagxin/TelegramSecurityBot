import telebot
from telebot import types
import numpy as np
from functools import partial
import sqlite3 # Потрібен для типізації та обробки помилок
import os

# Імпортуємо функцію аналізу з модуля ШІ
try:
    from ai_model.ipso_model import predict_single_message
    print("Функцію predict_single_message імпортовано.")
except ImportError:
    print("ПОМИЛКА: Не вдалося імпортувати predict_single_message з ai_model.ipso_model")
    # Функція-заглушка
    def predict_single_message(model, tokenizer, label_encoder, db_conn, text, source_id, max_length, threshold):
        print("УВАГА: Використовується заглушка predict_single_message!")
        reliability = 1.0
        # Імітуємо отримання/оновлення надійності через передане з'єднання
        if db_conn and source_id:
             try:
                 cursor = db_conn.cursor()
                 cursor.execute('SELECT reliability FROM sources WHERE source_id = ?', (str(source_id),))
                 result = cursor.fetchone()
                 if result: reliability = result[0]
             except sqlite3.Error as e_stub:
                 print(f"Помилка БД у заглушці: {e_stub}")
        # Імітуємо виявлення IPSO для тесту оновлення надійності
        is_ipso_stub = "погане" in text.lower() if isinstance(text, str) else False
        if is_ipso_stub: reliability = max(0.1, reliability - 0.02)

        return is_ipso_stub, reliability # is_ipso, reliability


# --- Основний обробник текстових повідомлень ---
# Тепер приймає db_conn замість db_path
def handle_text_message(message: types.Message, bot_instance: telebot.TeleBot, model, tokenizer, label_encoder, db_conn: sqlite3.Connection, max_length, threshold):
    """Обробляє вхідне текстове повідомлення, викликає ШІ та надсилає результат."""
    chat_id = message.chat.id
    message_id = message.message_id
    text_to_analyze = message.text
    print(f"\nОтримано текст від chat_id={chat_id}, message_id={message_id}")

    # Визначаємо source_id
    source_id = None
    forward_info = ""
    if message.forward_from_chat:
        source_id = str(message.forward_from_chat.id)
        forward_info = f" (Переслано з: {message.forward_from_chat.title or source_id})"
    elif message.forward_from:
        source_id = str(message.forward_from.id)
        forward_info = f" (Переслано від: {message.forward_from.first_name or source_id})"
    else:
        forward_info = " (Пряме повідомлення)"

    print(f"Текст: '{text_to_analyze[:60]}...', Джерело ID: {source_id}")

    response = "На жаль, сталася помилка під час аналізу. Спробуйте пізніше." # Відповідь за замовчуванням

    try:
        # Перевірка ресурсів ШІ
        if not all([model, tokenizer, label_encoder]):
             raise ValueError("Ресурси ШІ не були належним чином ініціалізовані.")
        # Перевірка з'єднання з БД
        if db_conn is None:
             raise ConnectionError("Відсутнє з'єднання з базою даних.")

        # Виклик функції аналізу ШІ з переданим з'єднанням db_conn
        is_ipso, final_reliability = predict_single_message(
            model=model,
            tokenizer=tokenizer,
            label_encoder=label_encoder,
            db_conn=db_conn, # <--- Передаємо існуюче з'єднання
            text=text_to_analyze,
            source_id=source_id,
            max_length=max_length,
            threshold=threshold
        )

        # Формування відповіді (використовуємо HTML)
        reliability_percent = final_reliability * 100
        source_display = f"<code>{source_id or 'Пряме повідомлення'}</code>{forward_info}"
        if is_ipso:
            response = (f"⚠️ <b>Виявлено ознаки ІПСО!</b>\n"
                        f"Джерело: {source_display}\n"
                        f"Поточна надійність джерела: <b>{reliability_percent:.1f}%</b>") # Додав .1f для наочності змін
        else:
            response = (f"✅ Аналіз завершено. Ознак ІПСО не виявлено.\n"
                        f"Джерело: {source_display}\n"
                        f"Надійність джерела: {reliability_percent:.1f}%")

    except ValueError as ve:
        print(f"Помилка значення (ймовірно, ресурси ШІ) для chat_id={chat_id}: {ve}")
        response = "Помилка конфігурації аналізатора. Зверніться до адміністратора."
    except ConnectionError as ce:
         print(f"Помилка з'єднання з БД для chat_id={chat_id}: {ce}")
         response = "Проблема з доступом до бази даних аналізатора. Спробуйте пізніше."
    except sqlite3.Error as e_sql: # Ловимо специфічні помилки SQLite
         print(f"Помилка SQLite під час обробки для chat_id={chat_id}: {e_sql}")
         # Можна перевірити на 'database is locked'
         if "lock" in str(e_sql).lower():
              response = "База даних аналізатора зараз зайнята. Будь ласка, спробуйте через декілька секунд."
         else:
              response = "Виникла помилка при роботі з базою даних аналізу."
    except Exception as e:
        print(f"Загальна помилка під час аналізу для chat_id={chat_id}: {e}")
        import traceback
        traceback.print_exc()
        # response залишається стандартним

    # Надсилання відповіді
    try:
        bot_instance.reply_to(message, response, parse_mode='HTML')
    except Exception as e_send:
        print(f"Помилка надсилання HTML відповіді для chat_id={chat_id}: {e_send}")
        try:
             # Використовуємо функцію escape з telebot.util
             response_plain = telebot.util.escape(response)
             # Додатково видаляємо теги, які escape не прибирає
             response_plain = response_plain.replace('<b>','').replace('</b>','').replace('<code>','').replace('</code>','')
             bot_instance.reply_to(message, response_plain)
             print("Відповідь надіслано без HTML форматування.")
        except Exception as e_send_plain:
              print(f"Повторна помилка надсилання відповіді (plain) для chat_id={chat_id}: {e_send_plain}")


# --- Обробник для нетекстових повідомлень (без змін) ---
def handle_other_messages(message: types.Message, bot_instance: telebot.TeleBot):
    print(f"Отримано нетекстове повідомлення ({message.content_type}) від chat_id={message.chat.id}")
    try:
        bot_instance.reply_to(message, "Вибачте, я призначений для аналізу лише текстових повідомлень.")
    except Exception as e:
        print(f"Помилка відповіді на нетекстове повід. для chat_id={message.chat.id}: {e}")


# --- Функція реєстрації обробників ---
# Тепер приймає db_conn замість db_path
def register_text_handlers(bot_instance: telebot.TeleBot, model, tokenizer, label_encoder, db_conn: sqlite3.Connection, max_length, threshold):
    """Реєструє обробники для текстових та інших типів повідомлень."""

    print(f"Реєстрація обробників тексту з db_conn: {type(db_conn)}") # Додав лог

    text_processor = partial(handle_text_message,
                             bot_instance=bot_instance,
                             model=model,
                             tokenizer=tokenizer,
                             label_encoder=label_encoder,
                             db_conn=db_conn, # <--- Передаємо з'єднання
                             max_length=max_length,
                             threshold=threshold)

    bot_instance.register_message_handler(text_processor, content_types=['text'], pass_bot=False)
    print("Обробник текстових повідомлень зареєстровано.")

    other_processor = partial(handle_other_messages, bot_instance=bot_instance)

    bot_instance.register_message_handler(other_processor,
                                         content_types=['audio', 'photo', 'voice', 'video', 'document',
                                                        'location', 'contact', 'sticker', 'video_note',
                                                        'animation', 'game', 'poll', 'dice'], pass_bot=False)
    print("Обробник для нетекстових повідомлень зареєстровано.")