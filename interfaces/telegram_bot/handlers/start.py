import asyncio
import logging
import sys
from pathlib import Path

from aiogram import Router, F
from aiogram.filters import CommandStart, Command
from aiogram.types import Message

# Добавляем корень проекта в путь для импорта src
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.search import Search

logger = logging.getLogger(__name__)

# Глобальный объект поисковой системы
_search_engine = None


def init_search_engine():
    """Принудительная инициализация при старте бота (вызывается один раз)."""
    global _search_engine
    if _search_engine is None:
        import yaml
        with open('config/search.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        _search_engine = Search(config)
        logger.info("Поисковая система инициализирована при старте")


def get_search_engine():
    """Возвращает уже инициализированный поисковый движок."""
    global _search_engine
    if _search_engine is None:
        # fallback (на случай, если забыли вызвать init_search_engine)
        init_search_engine()
    return _search_engine


def format_search_result(result: dict) -> str:
    """Форматирует результат поиска для Telegram."""
    text = ""

    # Категория
    cls = result.get('classification', {})
    if cls:
        primary = cls.get('primary_category', 'unknown')
        confidence = cls.get('primary_confidence', 0)
        text += f"Категория: {primary} (уверенность: {confidence:.0%})\n\n"

    # LLM-аналитика (обрезаем, если слишком длинная)
    llm = result.get('llm_analysis')
    if llm and llm.get('summary'):
        summary = llm.get('summary', 'Нет данных')
        # Если аналитика слишком длинная, обрезаем
        if len(summary) > 1500:
            summary = summary[:1500] + "..."
        text += f"Аналитика:\n{summary}\n\n"

    # Связи между делами
    conn = result.get('connections', {})
    if conn:
        judges = conn.get('common_judges', [])
        articles = conn.get('common_articles', [])
        if judges:
            judges_clean = [j for j in judges if j.lower() not in ['постановил', 'решил', 'определил']]
            if judges_clean:
                text += f"Судьи: {', '.join(judges_clean[:3])}\n"
        if articles:
            text += f"Статьи: {', '.join(articles[:3])}\n"

    # Документы
    docs = result.get('documents', [])[:5]
    if docs:
        text += f"\nНайденные документы (топ-{len(docs)}):\n"
        for i, doc in enumerate(docs, 1):
            score = doc.get('rerank_score', doc.get('faiss_score', 0))
            doc_id = doc.get('doc_id', 'N/A')
            text += f"{i}. Дело № {doc_id} — релевантность: {score:.2f}\n"

    # Время выполнения
    stats = result.get('stats', {})
    time_sec = stats.get('time_sec', 0)
    if time_sec:
        text += f"\nВремя: {time_sec:.1f} сек"
    else:
        text += f"\nВремя: {result.get('time_sec', 0):.1f} сек"

    return text


# Создаём роутер
start_router = Router()


@start_router.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer(
        "👋 Привет! Я — бот для поиска судебных решений.\n\n"
        "Как я работаю:\n"
        "Опиши свою ситуацию простыми словами, и я найду похожие судебные решения.\n"
    )


@start_router.message(Command('help'))
async def cmd_help(message: Message):
    await message.answer(
        "📖 Помощь\n\n"
        "Команды:\n"
        "/start — приветствие\n"
        "/help — справка\n"
        "/stats — статистика (в разработке)\n\n"
        "Как задать запрос:\n"
        "Просто напишите описание вашей ситуации. Чем подробнее — тем точнее результат.\n\n"
        "Что я умею:\n"
        "• Искать по смыслу, а не по ключевым словам\n"
        "• Анализировать судебную практику\n"
        "• Показывать связи между делами\n\n"
        "Время ответа до 20 секунд."
    )


@start_router.message(Command('stats'))
async def cmd_stats(message: Message):
    await message.answer(
        "📊 Статистика\n\n"
        "Функция в разработке.\n"
        "Скоро здесь будет информация об остатке запросов по ключу."
    )


@start_router.message(F.text)
async def handle_search_query(message: Message):
    user_query = message.text.strip()

    if not user_query:
        await message.answer("Опишите вашу ситуацию")
        return

    waiting_msg = await message.answer("🔍 Ищу судебные решения...")

    try:
        searcher = get_search_engine()
        result = await asyncio.get_event_loop().run_in_executor(
            None, searcher.search_with_analysis, user_query
        )

        result_text = format_search_result(result)

        # Обрезаем длинные сообщения (Telegram лимит ~4096 символов)
        if len(result_text) > 3500:
            result_text = result_text[:3500] + "\n\n... (ответ обрезан из-за длины)"

        # Отправляем результат (без Markdown, чтобы не было ошибок парсинга)
        await waiting_msg.edit_text(result_text)

    except Exception as e:
        logger.error(f"Ошибка поиска: {e}", exc_info=True)
        await waiting_msg.edit_text("❌ Произошла ошибка при поиске.\nПопробуйте позже.")