from src.data_processing.builder import Builder
from src.data_processing.chunker import Chunker
from src.data_processing.embedder import Embedder
from src.data_processing.ner import NER
from src.data_processing.vector_store import VectorStore
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class Indexing:
    """Пайплайн индексации документов."""

    def __init__(self, config: dict):
        self.config = config

        # Инициализация компонентов
        self.builder = Builder(
            dataset_name=config['builder']['dataset_name'],
            split=config['builder']['split']
        )
        self.chunker = Chunker(
            chunk_size=config['chunker']['chunk_size'],
            overlap=config['chunker']['overlap']
        )
        self.ner = NER()
        self.embedder = Embedder(**config.get('embedder', {}))
        self.vector_store = None # Инициализация будет произведена после определения размерности

        # Параметры данных
        self.text_field = config['text_field']
        self.category_field = config['category_field']
        self.output_dir = Path(config['output_dir'])

        # Параметры индекса
        self.index_type = config.get('vector_store', {}).get('index_type', 'Flat')

        logger.info("Индексация: инициализация произведена")

    def run(self,
            max_docs: Optional[int] = None,
            use_dataset: bool = True,
            folder_path: Optional[str] = None,
            folder_extensions: Optional[list] = None,
            max_docs_folder: Optional[int] = None
            ) -> pd.DataFrame:
        """Запуск индексации из датасета и/или папки."""
        logger.info("Запуск индексации...")

        all_docs = []

        # 1. Загружаем из датасета
        if use_dataset:
            logger.info("Загрузка датасета")
            self.builder.build_from_dataset(
                text_field=self.text_field,
                category_field=self.category_field
            )
            docs_df = self.builder.target_df
            if max_docs:
                docs_df = docs_df.head(max_docs)
            logger.info(f"Из датасета: {len(docs_df)} документов")
            all_docs.append(docs_df)

        # 2. Загружаем из папки
        if folder_path:
            logger.info("Загрузка из папки")
            from src.query_classifier import QueryClassifier
            classifier = QueryClassifier()
            folder_docs = self.builder.build_from_folder(
                folder_path=folder_path,
                classifier=classifier,
                extensions=folder_extensions,
                max_docs=max_docs_folder
            )
            logger.info(f"Из папки: {len(folder_docs)} документов")
            all_docs.append(folder_docs)

        if not all_docs:
            raise ValueError("Нет источников данных")

        # Объединяем
        combined_docs = pd.concat(all_docs, ignore_index=True)
        logger.info(f"Всего документов: {len(combined_docs)}")

        # 3. Чанкинг
        logger.info("Чанкинг")
        chunks_df = self.chunker.run_chunking(combined_docs)
        logger.info(f"Создано {len(chunks_df)} чанков")

        # 4. Извлечение сущностей
        logger.info("Извлечение сущностей")
        chunks_with_ner_df = self.ner.fill_dataset_entities(chunks_df, 'text')

        # 5. Векторизация
        logger.info("Векторизация")
        embeddings = self.embedder.encode(chunks_with_ner_df['text'].tolist(), show_progress=True)

        # 6. Построение индекса
        logger.info("Построение FAISS индекса")
        self.vector_store = VectorStore(
            dimension=embeddings.shape[1],
            index_type=self.index_type
        )
        self.vector_store.build(embeddings, chunks_with_ner_df)

        # Сохранение
        self._save_results(chunks_with_ner_df, embeddings)
        logger.info("Индексация завершена")

    def _save_results(self, df: pd.DataFrame, embeddings: np.ndarray):
        """Сохранение всех результатов."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Сохраняем метаданные чанков
        chunks_path = self.output_dir / 'chunks.parquet'
        df.to_parquet(chunks_path, index=False)

        # 2. Сохраняем эмбеддинги (опционально, для отладки)
        embeddings_path = self.output_dir / 'embeddings.npy'
        np.save(embeddings_path, embeddings)

        # 3. Сохраняем FAISS индекс
        index_path = self.output_dir / 'index.faiss'
        self.vector_store.save(index_path, chunks_path)

        # Информация о размерах
        chunks_size = chunks_path.stat().st_size / (1024 * 1024)
        embeddings_size = embeddings_path.stat().st_size / (1024 * 1024)

        logger.info(f"Чанки сохранены: {chunks_path} ({chunks_size:.2f} MB)")
        logger.info(f"Эмбеддинги сохранены: {embeddings_path} ({embeddings_size:.2f} MB)")

        csv_path = self.output_dir / 'chunks.csv'
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        logger.info(f"CSV сохранён: {csv_path}")
