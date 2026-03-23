import pandas as pd
import logging

logger = logging.getLogger(__name__)


class Chunker:
    """Разбиение документов на чанки для последующей обработки."""

    def __init__(self, chunk_size: int = 512, overlap: int = 2):
        """
        :param chunk_size: размер фрагмента текста
        :param overlap: размер пересечения между фрагментами
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        logger.info(f"Chunker инициализирован: chunk_size={chunk_size}, overlap={overlap}")

    def run_chunking(self, df: pd.DataFrame) -> pd.DataFrame:
        """Разбивает документы на чанки."""
        logger.info(f"Начало чанкинга: {len(df)} документов")

        chunks = []
        total_chunks = 0

        for idx, row in df.iterrows():
            doc_id = row['id']
            category = row['category']
            text = row['text']

            # Пропускаем пустые документы
            if not text or not isinstance(text, str):
                logger.warning(f"Документ {doc_id} пропущен: пустой текст")
                continue

            text_len = len(text)
            text_chunks = self._split_text(text)

            for i, chunk_text in enumerate(text_chunks):
                chunks.append({
                    'chunk_id': f"{doc_id}_{i}",
                    'doc_id': doc_id,
                    'category': category,
                    'text': chunk_text,
                    'entities': None  # будет заполнено позже
                })

            num_chunks = len(text_chunks)
            total_chunks += num_chunks
            logger.debug(f"Документ {doc_id} (длина {text_len}) -> {num_chunks} чанков")

        result_df = pd.DataFrame(chunks)
        logger.info(f"Чанкинг завершен: {total_chunks} чанков из {len(df)} документов")

        return result_df

    def _split_text(self, text: str) -> list:
        """Разбиение текста на фрагменты (чанки) с перекрытием."""
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        step = self.chunk_size - self.overlap

        for i in range(0, len(text), step):
            chunk = text[i:i + self.chunk_size]
            if chunk and len(chunk) > 10:  # отсекаем слишком короткие чанки
                chunks.append(chunk)

        logger.debug(f"Текст длиной {len(text)} -> {len(chunks)} чанков")
        return chunks
