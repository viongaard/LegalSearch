"""
LLM-КЛИЕНТ ДЛЯ RAG-ОТВЕТОВ С YANDEX GPT 5 LITE
Использует OpenAI-совместимый интерфейс
"""

import openai
import logging
from typing import List, Dict, Any, Optional

from src.llm.config import (
    BASE_URL,
    YANDEX_CLOUD_API_KEY,
    YANDEX_CLOUD_FOLDER,
    YANDEX_CLOUD_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    LLM_ENABLED
)
from src.llm.prompts import (
    SYSTEM_PROMPT,
    build_user_prompt,
    build_fallback_answer
)

logger = logging.getLogger(__name__)


class YandexGPTClient:
    """Клиент для генерации ответов через Yandex GPT 5 Lite API"""

    def __init__(self):
        self.api_key = YANDEX_CLOUD_API_KEY
        self.folder = YANDEX_CLOUD_FOLDER
        self.model = f"gpt://{self.folder}/{YANDEX_CLOUD_MODEL}"

        # Инициализация OpenAI-совместимого клиента
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=BASE_URL,
            project=self.folder
        )

        self.available = LLM_ENABLED and self._check_connection()

        if self.available:
            logger.info("YandexGPT 5 Lite клиент инициализирован")
        else:
            logger.warning("YandexGPT клиент недоступен")

    def _check_connection(self) -> bool:
        """Проверка доступности YandexGPT API"""
        if not self.api_key or not self.folder:
            logger.warning("Не настроены API ключ или folder ID")
            return False

        try:
            # Тестовый запрос
            response = self.client.responses.create(
                model=self.model,
                temperature=0.1,
                instructions="Ответь одним словом: OK",
                input="Тест",
                max_output_tokens=10
            )
            logger.info("YandexGPT API доступен")
            return True
        except Exception as e:
            logger.error(f"Ошибка при проверке YandexGPT: {e}")
            return False

    def generate_answer(
            self,
            query: str,
            documents: List[Dict[str, Any]],
            classification: Dict[str, Any]
    ) -> Optional[str]:
        """
    Генерация юридического ответа через YandexGPT

    Args:
        query: исходный запрос пользователя
        documents: список найденных документов (с полями text, category, rerank_score и т.д.)
        classification: результат классификации (primary_category, primary_confidence, expanded_categories)

    Returns:
        Сгенерированный ответ или None
    """
        if not self.available:
            logger.warning("LLM недоступен, возвращаю fallback ответ")
            return build_fallback_answer(documents, query, classification)

        # Строим контекст из документов
        context = self._build_context(documents)

        # Формируем пользовательский промпт
        user_prompt = build_user_prompt(
            query=query,
            case_type=classification.get('primary_category', 'unknown'),
            topics=classification.get('expanded_categories', []),
            context=context
        )

        try:
            logger.info("Отправка запроса к YandexGPT...")

            response = self.client.responses.create(
                model=self.model,
                temperature=DEFAULT_TEMPERATURE,
                instructions=SYSTEM_PROMPT,
                input=user_prompt,
                max_output_tokens=DEFAULT_MAX_TOKENS
            )

            answer = response.output_text
            logger.info("Ответ от YandexGPT получен успешно")
            return answer.strip()

        except Exception as e:
            logger.error(f"Ошибка при вызове YandexGPT: {e}")
            return build_fallback_answer(documents, query, classification)

    def _build_context(self, documents: List[Dict[str, Any]], max_docs: int = 3, max_chars: int = 1500) -> str:
        """Строит контекст из топ документов"""
        context_parts = []

        for i, doc in enumerate(documents[:max_docs]):
            # Получаем текст документа
            text = doc.get('text_full', doc.get('text', ''))
            if not text:
                text = doc.get('text_preview', '')

            # Ограничиваем длину
            if len(text) > max_chars:
                text = text[:max_chars] + "..."

            context_parts.append(f"""
ДОКУМЕНТ {i + 1}:
ID дела: {doc.get('doc_id', 'Не указан')}
Категория: {doc.get('category', 'Не указана')}
Релевантность: {doc.get('rerank_score', doc.get('faiss_score', 0)):.3f}
Текст: {text}
""")

        return "\n".join(context_parts)


# Функция для быстрого создания клиента
def get_llm_client() -> Optional[YandexGPTClient]:
    """Создаёт и возвращает LLM клиент, если он доступен"""
    if LLM_ENABLED:
        return YandexGPTClient()
    return None