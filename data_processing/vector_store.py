# src/vector_store/vector_store.py
import faiss
import numpy as np
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = None
        self.df = None

    def build(self, embeddings: np.ndarray, df: pd.DataFrame):
        """Строит FAISS индекс из эмбеддингов."""
        
        self.index = faiss.IndexFlatIP(self.dimension)  # inner product для нормализованных векторов
        self.index.add(embeddings)
        self.df = df.reset_index(drop=True)
        logger.info(f"Индекс построен: {self.index.ntotal} векторов")

    def save(self, index_path: Path, metadata_path: Path):
        """Сохраняет индекс и метаданные."""

        faiss.write_index(self.index, str(index_path))
        self.df.to_parquet(metadata_path, index=False)
        logger.info(f"Индекс сохранён: {index_path}")

    def load(self, index_path: Path, metadata_path: Path):
        """Загружает индекс и метаданные."""

        self.index = faiss.read_index(str(index_path))
        self.df = pd.read_parquet(metadata_path)
        logger.info(f"Индекс загружен: {self.index.ntotal} векторов")