# bot/handlers/common.py
import telebot # Потрібен для типів
from telebot import types

# Функцію start переносимо сюди
def start_handler(message: types.Message, bot_instance: telebot.TeleBot):
    """Обробник команди /start"""
    instruction = (
        "Привіт! Я DisRaze - бот на основі штучного інтелекту, який допоможе тобі виявити дезинформацію у текстовому повідомлені.\n"
        "Ось що я вмію:\n"
        "1. Аналізувати текст на присутність дезинформації\n"
        "2. (Додати опис функції зображень пізніше)\n"
        "3. ...\n"
        "Обери одну з функцій нижче або надішли мені текст чи фото для аналізу:"
    )

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=1) # Змінено row_width
    btn1 = types.KeyboardButton('Аналізувати текст') # Змінено текст для простоти
    btn2 = types.KeyboardButton('Аналізувати зображення (скоро)') # Додамо кнопку для майбутнього
    # btn3 = types.KeyboardButton('Тест_3')
    markup.add(btn1, btn2) # Додаємо кнопки

    bot_instance.send_message(message.chat.id, instruction, reply_markup=markup)

# Функцію help_command переносимо сюди
def help_handler(message: types.Message, bot_instance: telebot.TeleBot):
    """Обробник команди /help"""
    help_text = (
        "Цей бот може допомогти виявити дезінформацію у текстових повідомленнях та (скоро) на зображеннях.\n"
        "Надішли мені текст або зображення, і я спробую його проаналізувати.\n\n"
        "Доступні команди:\n"
        "/start - Запуск бота та показ головного меню.\n"
        "/help - Ця довідка.\n"
    )
    bot_instance.send_message(message.chat.id, help_text)

# Функція для реєстрації цих обробників
def register_common_handlers(bot_instance: telebot.TeleBot):
    bot_instance.register_message_handler(
        lambda msg: start_handler(msg, bot_instance), # Передаємо bot_instance
        commands=['start']
    )
    bot_instance.register_message_handler(
        lambda msg: help_handler(msg, bot_instance), # Передаємо bot_instance
        commands=['help']
    )