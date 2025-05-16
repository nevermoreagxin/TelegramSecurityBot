# bot/handlers/common.py
import telebot
from telebot import types # Потрібен для ReplyKeyboardRemove

# --- Обробник команди /start ---
def start_handler(message: types.Message, bot_instance: telebot.TeleBot):
    """Обробник команди /start. Надсилає вітальне повідомлення та прибирає старі кнопки."""
    instruction = (
        "Привіт! Я DisRaze - бот на основі штучного інтелекту, який допоможе тобі виявити дезінформацію та ознаки ІПСО у текстових повідомленнях.\n\n"
        "**Як користуватися:**\n"
        "Просто надішли мені будь-яке текстове повідомлення або перешли його з іншого чату/каналу, і я спробую його проаналізувати, враховуючи також надійність джерела (для пересланих повідомлень).\n\n"
        "Для отримання детальнішої довідки використовуй команду /help."
    )

    # Створюємо об'єкт для видалення попередньої клавіатури
    remove_keyboard = types.ReplyKeyboardRemove(selective=False)

    try:
        # Надсилаємо повідомлення, вказуючи reply_markup=remove_keyboard
        bot_instance.send_message(message.chat.id, instruction,
                                  reply_markup=remove_keyboard, parse_mode='Markdown')
    except Exception as e:
        print(f"Помилка надсилання /start повідомлення для chat_id={message.chat.id}: {e}")
        try:
            # Спроба надіслати без форматування, але все одно прибираючи клавіатуру
            bot_instance.send_message(message.chat.id, instruction.replace('**',''),
                                      reply_markup=remove_keyboard)
        except Exception as e_plain:
            print(f"Повторна помилка надсилання /start (plain) для chat_id={message.chat.id}: {e_plain}")


# --- Обробник команди /help ---
def help_handler(message: types.Message, bot_instance: telebot.TeleBot):
    """Обробник команди /help. Надсилає довідкову інформацію."""
    help_text = (
        "**Довідка по боту DisRaze:**\n\n"
        "Я аналізую текстові повідомлення на наявність ознак інформаційно-психологічних операцій (ІПСО) та іншої дезінформації. Моя мета – допомогти тобі критично оцінювати інформацію.\n\n"
        "**Як це працює:**\n"
        "1. Ти надсилаєш мені текстове повідомлення (або пересилаєш з іншого джерела).\n"
        "2. Я аналізую текст за допомогою нейронної мережі.\n"
        "3. Якщо повідомлення було переслане, я також враховую та оновлюю показник \"надійності\" джерела. Чим більше ІПСО надходить від певного джерела, тим нижчою стає його надійність. Якщо джерело поширює безпечний контент, його надійність зростає.\n"
        "4. Ти отримуєш результат: висновок про наявність/відсутність ознак ІПСО та поточну надійність джерела (для пересланих повідомлень).\n\n"
        "**Що означає \"надійність джерела\":**\n"
        "Це динамічний показник від 10% до 100%, який відображає, наскільки часто повідомлення з даного джерела (каналу, групи, користувача) раніше класифікувалися як ІПСО. Він не є абсолютною істиною, але може слугувати додатковим індикатором.\n\n"
        "**Важливо:**\n"
        "- Я аналізую **тільки текст**. Зображення, відео, стікери тощо ігноруються.\n"
        "- Мої висновки базуються на роботі нейронної мережі і не є остаточним вердиктом. Завжди аналізуй інформацію з різних джерел та використовуй критичне мислення!\n\n"
        "**Доступні команди:**\n"
        "/start - Початок роботи, вітальне повідомлення.\n"
        "/help - Ця довідка."
    )
    try:
        bot_instance.send_message(message.chat.id, help_text, parse_mode='Markdown')
    except Exception as e:
        print(f"Помилка надсилання /help повідомлення для chat_id={message.chat.id}: {e}")
        try:
            bot_instance.send_message(message.chat.id, help_text.replace('**',''))
        except Exception as e_plain:
            print(f"Повторна помилка надсилання /help (plain) для chat_id={message.chat.id}: {e_plain}")


# --- Функція реєстрації обробників ---
def register_common_handlers(bot_instance: telebot.TeleBot):
    """Реєструє обробники для спільних команд."""
    # Використовуємо pass_bot=False, оскільки bot_instance передається через lambda
    # (хоча для команд це не так критично, як для message_handler з content_types)
    bot_instance.register_message_handler(
        lambda msg: start_handler(msg, bot_instance),
        commands=['start'],
        pass_bot=False # Явно вказуємо, що не треба передавати бота ще раз
    )
    bot_instance.register_message_handler(
        lambda msg: help_handler(msg, bot_instance),
        commands=['help'],
        pass_bot=False
    )
    print("Обробники спільних команд (/start, /help) зареєстровано.")