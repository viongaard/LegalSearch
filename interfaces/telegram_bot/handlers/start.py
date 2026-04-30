from aiogram import Router, F
from aiogram.filters import CommandStart, Command
from aiogram.types import Message

# Создаём роутер для этой группы хендлеров
start_router = Router()


@start_router.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer(
        "Привет! Я - бот для поиска судебных решений.\n\n"
        "Опиши мне свою ситуацию, и я найду для тебя близкие по смыслу судебные решения."
    )


@start_router.message(F.text)
async def handle_search_query(message: Message):
    user_query = message.text
    # Отправляем уведомление о начале поиска
    waiting_msg = await message.answer("Начал поиск...")

    # TODO: сюда добавить вызов системы поиска
    # result_text = await your_search_system.search(user_query)

    result_text = (
        f"Запрос: '{user_query}'\n\n"
        "**Аналитика (тестовый режим):**\n"
        "Система поиска пока не подключена. Здесь будет результат анализа.\n\n"
    )

    await waiting_msg.edit_text(result_text)