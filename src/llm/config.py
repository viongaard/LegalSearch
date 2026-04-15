"""
Конфигурация LLM клиента для YandexGPT 5 Lite
"""

import os
from pathlib import Path

# Загрузка переменных окружения из .env файла
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / "config" / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

# Yandex GPT настройки
YANDEX_CLOUD_API_KEY = os.getenv("YANDEX_CLOUD_API_KEY", "xxxxxxxx")
YANDEX_CLOUD_FOLDER = os.getenv("YANDEX_CLOUD_FOLDER", "xxxxxxxx")
YANDEX_CLOUD_MODEL = os.getenv("YANDEX_CLOUD_MODEL", "yandexgpt-5-lite/latest")
BASE_URL = "https://ai.api.cloud.yandex.net/v1"

# Настройки генерации
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 500

# Флаг доступности
LLM_ENABLED = bool(YANDEX_CLOUD_API_KEY and YANDEX_CLOUD_FOLDER)