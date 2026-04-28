from decouple import config

# Yandex GPT настройки
YANDEX_CLOUD_API_KEY = config("YANDEX_CLOUD_API_KEY")
YANDEX_CLOUD_FOLDER = config("YANDEX_CLOUD_FOLDER")
YANDEX_CLOUD_MODEL = config("YANDEX_CLOUD_MODEL")
BASE_URL = "https://ai.api.cloud.yandex.net/v1"

# Настройки генерации
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 500

# Флаг доступности
LLM_ENABLED = bool(YANDEX_CLOUD_API_KEY and YANDEX_CLOUD_FOLDER)