from datasets import load_dataset
import pandas as pd
from pathlib import Path
import logging
import uuid
from tqdm import tqdm

from src.data_processing.chunker import Chunker
from src.query_classifier import QueryClassifier

logger = logging.getLogger(__name__)


class Builder:
    """Загрузчик и преобразователь датасета судебных решений."""

    def __init__(self, dataset_name: str = "lawful-good-project/sud-resh-benchmark", split: str = 'train'):
        logger.info(f"Загрузка датасета {dataset_name}, сплит {split}")
        self.dataset = load_dataset(dataset_name, split=split)
        self.df = pd.DataFrame(self.dataset)
        self.target_df = None
        logger.info(f"Загружено {len(self.df)} документов")

    def print_info(self):
        """Вывод информации о загруженном датасете."""

        logger.debug("Вывод информации о датасете")
        print("\n--- Информация о датасете ---")
        print(f"Колонки: {self.df.columns.tolist()}")
        print(f"Размер: {self.df.shape}")
        print(f"\nПервые 3 строки:")
        print(self.df.head(3))
        print(f"\nСтатистика:")
        print(self.df.describe(include='all'))

    def build_from_dataset(self, text_field: str = 'source', category_field: str = 'category'):
        """Создание и заполнение целевого датасета для работы системы."""

        logger.info("Создание целевого датасета")

        # Проверка наличия полей
        if text_field not in self.df.columns:
            raise ValueError(
                f"Поле '{text_field}' не найдено. Доступные колонки: {self.df.columns.tolist()}")

        if category_field not in self.df.columns:
            raise ValueError(
                f"Поле '{category_field}' не найдено. Доступные колонки: {self.df.columns.tolist()}")

        # Извлечение данных
        logger.debug(f"Извлечение текста из поля '{text_field}'")
        texts = self.df[text_field].fillna('').tolist()

        logger.debug(f"Извлечение категорий из поля '{category_field}'")
        categories = self.df[category_field].fillna('').tolist()

        # Создание целевого датасета
        self.target_df = pd.DataFrame({
            'id': range(len(self.df)),
            'category': categories,
            'text': texts,
            'entities': [[] for _ in range(len(self.df))]
        })

        # Удаление дубликатов
        original_count = len(self.target_df)
        self.target_df = self.target_df.drop_duplicates(subset=['text'], keep='first')
        removed_count = original_count - len(self.target_df)

        if removed_count > 0:
            logger.info(
                f"Удалено дубликатов документов: {removed_count} (было {original_count}, стало {len(self.target_df)})")
        else:
            logger.info(f"Дубликатов документов не найдено")

        logger.info(f"Создан датасет: {len(self.target_df)} записей, "
                    f"уникальных категорий: {self.target_df['category'].nunique()}")

        return self.target_df

    def build_from_folder(
            self,
            folder_path: str,
            classifier,
            extensions: list = None,
            max_docs: int = None,
            start_id: int = 0  # новый параметр
    ):
        """Загружает документы из папки, классифицирует документ (не чанки).
        Возвращает DataFrame в том же формате, что и build_from_dataset.
        """
        if extensions is None:
            extensions = ['.txt']

        folder = Path(folder_path)
        if not folder.exists():
            raise ValueError(f"Папка не найдена: {folder_path}")

        files = []
        for ext in extensions:
            files.extend(list(folder.glob(f"*{ext}")))

        if max_docs:
            files = files[:max_docs]

        print(f"Найдено {len(files)} файлов")

        docs = []
        current_id = start_id

        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()

                if len(text) < 100:
                    print(f"Пропущен (короткий): {file_path.name}")
                    continue

                # Классификация документа (первые 2000 символов)
                res = classifier.predict(text[:2000])

                docs.append({
                    'id': current_id,  # теперь число
                    'category': res['primary_category'],
                    'text': text,
                    'entities': []
                })
                current_id += 1

            except Exception as e:
                print(f"Ошибка {file_path.name}: {e}")

        self.target_df = pd.DataFrame(docs)
        print(f"Загружено {len(self.target_df)} документов из папки")
        return self.target_df

    def save(self, path: str = 'data/processed/documents.parquet'):
        """Сохранение целевого датасета в Parquet."""

        if self.target_df is None:
            raise ValueError("Целевой датасет не создан. Сначала вызовите fill_target_dataset().")

        logger.info(f"Сохранение датасета в {path}")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.target_df.to_parquet(path, index=False)

        size_mb = path.stat().st_size / (1024 * 1024)
        logger.info(f"Сохранено: {path} ({size_mb:.2f} MB)")

    def load(self, path: str = "data/processed/documents.parquet"):
        """Загрузка ранее сохранённого целевого датасета."""

        logger.info(f"Загрузка датасета из {path}")
        self.target_df = pd.read_parquet(path)
        logger.info(f"Загружено {len(self.target_df)} записей")
        return self.target_df
