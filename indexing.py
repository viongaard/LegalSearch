from data_processing.builder import Builder
from data_processing.chunker import Chunker
from data_processing.ner import NER
import logging
import pandas as pd
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class Indexing:
    def __init__(self, config: dict):
        self.builder = Builder(
            dataset_name=config['builder']['dataset_name'],
            split=config['builder']['split']
        )
        self.chunker = Chunker(
            chunk_size=config['chunker']['chunk_size'],
            overlap=config['chunker']['overlap']
        )
        self.ner = NER()

        self.text_field = config['text_field']
        self.category_field = config['category_field']
        self.output_dir = Path(config['output_dir'])

        logger.info("Indexing pipeline инициализирован")

    def run(self, max_docs: Optional[int] = None) -> pd.DataFrame:
        """Запуск индексации."""

        logger.info("Запуск пайплайна индексации...")

        # 1. Загрузка и подготовка датасета
        logger.info("Шаг 1/3: Загрузка датасета")
        self.builder.fill_target_dataset(
            text_field=self.text_field,
            category_field=self.category_field
        )

        docs_df = self.builder.target_df
        if max_docs:
            docs_df = docs_df.head(max_docs)
            logger.info(f"Ограничение: {max_docs} документов")

        logger.info(f"Загружено {len(docs_df)} документов")

        # 2. Чанкинг
        logger.info("Шаг 2/3: Разбиение на чанки")
        chunks_df = self.chunker.run_chunking(docs_df)
        logger.info(f"Создано {len(chunks_df)} чанков")

        # 3. Извлечение сущностей
        logger.info("Шаг 3/3: Извлечение сущностей")
        chunks_with_ner_df = self.ner.fill_dataset_entities(
            chunks_df,
            text_column_name='text'
        )

        # Статистика по сущностям
        total_entities = chunks_with_ner_df['entities'].apply(len).sum()
        logger.info(f"Всего найдено сущностей: {total_entities}")

        # 4. Сохранение результата
        self._save_results(chunks_with_ner_df)

        logger.info("Индексация завершена.")
        return chunks_with_ner_df

    def _save_results(self, df: pd.DataFrame):
        """Сохранение результатов в Parquet."""

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / 'chunks_with_ner.parquet'

        df.to_parquet(output_path, index=False)

        # Информация о сохранённом файле
        file_size = output_path.stat().st_size / (1024 * 1024)  # в MB
        logger.info(f"Результат сохранён: {output_path}")
        logger.info(f"Размер файла: {file_size:.2f} MB")
