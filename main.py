# main.py (у корені проекту)
import telebot
from bot.config import BOT_TOKEN # Імпортуємо токен з нашого конфігу

# Імпортуємо функції реєстрації (будуть створені далі)
from bot.handlers.common import register_common_handlers
from bot.handlers.text import register_text_handlers
# from bot.handlers.photo import register_photo_handlers # Для майбутнього

# Імпортуємо аналізатор (буде створений далі)
from bot.analysis.text_analyzer import TextAnalyzer

if __name__ == '__main__': # Стандартний блок для запуску головного файлу
    print("Запуск main.py...")
    print("Ініціалізація бота...")
    bot = telebot.TeleBot(BOT_TOKEN)

    print("Ініціалізація аналізаторів...")
    # Створюємо екземпляр аналізатора тексту (поки без шляху до моделі)
    text_analyzer = TextAnalyzer(model_path=None)

    print("Реєстрація обробників...")
    # Реєструємо всі наші обробники, передаючи їм екземпляр бота
    # і аналізатор (де потрібно)
    register_common_handlers(bot)
    register_text_handlers(bot, text_analyzer) # Передаємо аналізатор
    # register_photo_handlers(bot) # Для майбутнього
    print("Обробники зареєстровано.")

    print("Запуск бота (полінг)...")
    try:
        # Запускаємо бота в режимі нескінченного очікування повідомлень
        bot.polling(none_stop=True)
    except Exception as e:
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"Сталася помилка під час роботи бота: {e}")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    finally:
        # Цей код виконається, якщо полінг зупиниться (наприклад, через помилку або Ctrl+C)
        print("Роботу бота зупинено.")