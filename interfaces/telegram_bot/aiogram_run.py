import asyncio
from create_bot import bot, dp, scheduler
from handlers.start import start_router

async def main():
    # Подключение роутеров
    dp.include_router(start_router)

    # Удаление вебхуков
    await bot.delete_webhook(drop_pending_updates=True)
    # Запуск бота
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())