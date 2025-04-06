# bot/handlers/text.py
import telebot
from telebot import types
# НЕ ПОТРІБНО: from bot.analysis.text_analyzer import TextAnalyzer # Ми отримуємо екземпляр з main
from functools import partial # Для передачі аргументів у хендлери

# Функція, що викликається після register_next_step_handler
def process_text_for_analysis(message: types.Message, bot_instance: telebot.TeleBot, analyzer): # Додали analyzer
    """Обробляє текст, надісланий для аналізу."""
    if message.content_type == 'text':
        # !!! Виклик РЕАЛЬНОГО аналізатора !!!
        analysis_result = analyzer.analyze(message.text) # Використовуємо переданий analyzer

        # Формуємо відповідь
        confidence_percent = analysis_result.get('confidence', 0.0) * 100
        if analysis_result.get("is_disinformation"):
             response = f"Аналіз завершено:\n" \
                        f"🔴 Схоже на дезінформацію ({confidence_percent:.0f}%)"
        elif "error" in analysis_result:
             response = f"⚠️ Помилка аналізу: {analysis_result['error']}"
        else:
             response = f"Аналіз завершено:\n" \
                        f"🟢 Ознак дезінформації не виявлено ({100 - confidence_percent:.0f}% впевненості у протилежному)."

        bot_instance.send_message(message.chat.id, response)
    else:
        bot_instance.send_message(message.chat.id, "Будь ласка, надішліть текстове повідомлення для аналізу.")
        # Повторно викликаємо запит, передаючи той самий аналізатор
        request_text_input(message, bot_instance, analyzer)

# Функція, що реагує на кнопку або текстове повідомлення для аналізу
def request_text_input(message: types.Message, bot_instance: telebot.TeleBot, analyzer): # Додали analyzer
    """Запитує у користувача текст для аналізу."""
    invite_text = "Добре, надішли мені текст, який потрібно проаналізувати."
    markup = types.ForceReply(selective=False)
    sent_msg = bot_instance.send_message(message.chat.id, invite_text, reply_markup=markup)
    # Реєструємо наступний крок, передаючи bot_instance та analyzer
    # Використовуємо partial для фіксації аргументів bot_instance та analyzer
    process_func = partial(process_text_for_analysis, bot_instance=bot_instance, analyzer=analyzer)
    bot_instance.register_next_step_handler(sent_msg, process_func)

# Функція для реєстрації цих обробників
def register_text_handlers(bot_instance: telebot.TeleBot, analyzer): # Приймає analyzer
    # Використовуємо partial для передачі bot_instance та analyzer
    request_func = partial(request_text_input, bot_instance=bot_instance, analyzer=analyzer)
    bot_instance.register_message_handler(
        request_func,
        func=lambda message: message.text == 'Аналізувати текст'
    )