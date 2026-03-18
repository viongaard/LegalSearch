import logging
from tqdm import tqdm
import pandas as pd
import numpy as np


# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Indexing:
    def __init__(self, loader, preprocessor, ner, embedder, vector_store):
        """

        :param loader:
        :param preprocessor:
        :param ner:
        :param embedder:
        :param vector_store:
        """

    def run(self):
        """
        Indexing pipeline
        :return:
        """
        logger.info("Starting indexing pipeline...")

        # Загрузка датасета документов судебных дел
        logger.info("Loading dataset...")
        documents = self.loader.load()
        logger.info(f"Loaded {len(documents)} documents.")

        # Обработка документов (чанкирование, извлечение сущностей, перевод в векторную БД)
        logger.info("Processing documents...")
        all_chunks = []
        for doc in tqdm(documents, desc="Documents"):
            doc_id = doc['id']
            doc_source_text = doc['source']
            # Chunking
            logger.info("Chunking...")
            chunks = self.preprocessor.process_document(doc)
            logger.info("NER...")
            # NER
            texts



