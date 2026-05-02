import asyncio
from create_bot import bot, dp
from handlers.start import start_router, init_search_engine

async def main():
    # 1. Принудительная инициализация поиска при старте бота
    print("Загрузка поисковой системы (модели, индекс)...")
    init_search_engine()
    print("Поисковая система готова")

    # 2. Подключение роутеров
    dp.include_router(start_router)

    # 3. Запуск бота
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())