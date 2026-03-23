
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
import logging

logger = logging.getLogger(__name__)

class Search:
    def __init__(self, config: dict):
        # Пути к данным
        self.index_path = Path(config['index_path'])
        self.metadata_path = Path(config['metadata_path'])

        # Загрузка моделей TODO: добавить классификатор
        self.bi_encoder = SentenceTransformer(

        )
        self.cross_encoder = CrossEncoder(

        )

        # Параметры поиска
        self.retrieve_k =
        self.rank_k =

        # Загрузка индекса и метаданных
        self._load_index

        logging.info(f"Search инициализирован. retrieve_k={self.retrieve_k}, rank_k={self.rank_k}")

