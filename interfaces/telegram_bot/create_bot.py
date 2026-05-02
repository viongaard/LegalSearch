import logging
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.enums import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage
from decouple import config

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Загрузка переменных из .env
ADMINS = [int(admin_id) for admin_id in config('TELEGRAM_BOT_ADMINS').split(',')]
TOKEN = config('TELEGRAM_BOT_TOKEN')

# Инициализация бота и диспетчера
session = AiohttpSession(timeout=120)
bot = Bot(token=TOKEN, session=session, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher(storage=MemoryStorage())

logger.info("Бот инициализирован")