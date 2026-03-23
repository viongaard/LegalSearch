# src/vector_store/vector_store.py
import faiss
import numpy as np
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, dimension: int, index_type: str = "flat"):
        """
        :param dimension: размерность векторов
        :param index_type: тип индекса faiss
        """

        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.df = None

        logger.info(f"VectorStore инициализирован: dimension={dimension}, index_type={index_type}")

    def build(self, embeddings: np.ndarray, df: pd.DataFrame):
        """Строит FAISS индекс из эмбеддингов."""

        logger.info(f"Построение индекса для {len(embeddings)} векторов...")
        index_type_lower = self.index_type.lower()
        if index_type_lower == "flat":
            self.index = faiss.IndexFlatIP(self.dimension)
        elif index_type_lower == "ivf":
            # IVF с 100 кластерами для ускорения
            quantizer = faiss.IndexFlatIP(self.dimension)
            nlist = min(100, len(embeddings) // 10)  # не больше 10% от числа векторов
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            # Обучаем индекс (нужно минимум nlist * 39 векторов)
            if len(embeddings) > nlist * 39:
                logger.info(f"Обучение IVF индекса с {nlist} кластерами...")
                self.index.train(embeddings)
            else:
                logger.warning(f"Слишком мало векторов для IVF, использую Flat")
                self.index = faiss.IndexFlatIP(self.dimension)
        else:
            raise ValueError(f"Неизвестный тип индекса: {index_type_lower}")

        self.index.add(embeddings)
        self.df = df.reset_index(drop=True)
        logger.info(f"Индекс построен: {self.index.ntotal} векторов")

    def save(self, index_path: Path, metadata_path: Path):
        """Сохраняет индекс и метаданные."""

        if self.index is None or self.df is None:
            raise ValueError("Индекс не построен. Сначала вызовите build().")

        index_path.parent.mkdir(parents=True, exist_ok=True) # Создаём директорию, если её нет
        faiss.write_index(self.index, str(index_path)) # Сохраняем индекс
        self.df.to_parquet(metadata_path, index=False) # Сохраняем метаданные

        index_size = index_path.stat().st_size / (1024 * 1024)
        meta_size = metadata_path.stat().st_size / (1024 * 1024)

        logger.info(f"Индекс сохранён: {index_path} ({index_size:.2f} MB)")
        logger.info(f"Метаданные сохранены: {metadata_path} ({meta_size:.2f} MB)")

    def load(self, index_path: Path, metadata_path: Path):
        """Загружает FAISS индекс и метаданные."""
        self.index = faiss.read_index(str(index_path))
        self.df = pd.read_parquet(metadata_path)

        logger.info(f"Индекс загружен: {self.index.ntotal} векторов")
        logger.info(f"Метаданные загружены: {len(self.df)} записей")

        # Проверка соответствия
        if self.index.ntotal != len(self.df):
            logger.warning(f"Несоответствие: в индексе {self.index.ntotal} векторов, "
                          f"в метаданных {len(self.df)} записей")

    def search(self, query_embeddings: np.ndarray, k: int = 10) -> pd.DataFrame:
        """Поиск k-ближайших соседей"""

        if self.index is None or self.df is None:
            raise ValueError("Индекс не загружен. Сначала вызовите load()")

        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)

        # Поиск
        scores, indices = self.index.search(query_embeddings, k)
        results = self.df.iloc[indices[0]].copy()
        results['score'] = scores[0]

        return results