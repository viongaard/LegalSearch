import pandas as pd
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    Doc
)
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class NER:
    """Извлечение именованных сущностей из юридических текстов с помощью Natasha."""

    def __init__(self):
        """Инициализация компонентов Natasha."""
        logger.info("Инициализация NER...")

        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.emb)
        self.syntax_parser = NewsSyntaxParser(self.emb)
        self.ner_tagger = NewsNERTagger(self.emb)

        logger.info("NER инициализирован")

    def fill_dataset_entities(self, df: pd.DataFrame, text_column_name: str = 'text') -> pd.DataFrame:
        """Обработка датасета и добавление колонки с сущностями."""
        logger.info(f"Начало извлечения сущностей: {len(df)} записей")

        texts = df[text_column_name].tolist()
        all_entities = []
        total_entities = 0

        for i, text in enumerate(texts):
            entities = self.extract_entities_from_text(text)
            all_entities.append(entities)
            total_entities += len(entities)

            # Логируем прогресс каждые 100 записей
            if (i + 1) % 100 == 0:
                logger.info(f"Обработано {i + 1}/{len(df)} записей, найдено {total_entities} сущностей")

        result_df = df.copy()
        result_df['entities'] = all_entities

        logger.info(f"Извлечение завершено. Всего записей: {len(df)}, сущностей: {total_entities}")
        return result_df

    def extract_entities_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Извлечение сущностей из фрагмента текста."""

        if not text or not isinstance(text, str):
            return []

        # Создаём документ Natasha и обрабатываем
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)
        doc.parse_syntax(self.syntax_parser)
        doc.tag_ner(self.ner_tagger)

        # Нормализация
        for span in doc.spans:
            span.normalize(self.morph_vocab)

        # Сборка результатов
        entities = []
        for span in doc.spans:
            entities.append({
                'type': span.type,
                'text': span.text,
                'normalized': span.normal
            })

        if entities:
            logger.debug(f"Найдено {len(entities)} сущностей в тексте длиной {len(text)}")

        return entities