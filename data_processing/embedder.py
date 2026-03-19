from sentence_transformers import SentenceTransformer
import numpy as np
import logging

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self, model_name: str = "cointegrated/rubert-tiny2"):
        self.model = SentenceTransformer(model_name)
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedder инициализирован: {model_name}, размерность={self.embedding_dimension}")

    def encode(self, texts: list, batch_size: int = 16) -> np.ndarray:
        """Преобразует список текстов в матрицу эмбеддингов."""

        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)
