from datasets import load_dataset
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Builder:
    """Загрузчик и преобразователь датасета судебных решений."""

    def __init__(self, dataset_name: str = "lawful-good-project/sud-resh-benchmark", split: str = 'train'):
        logger.info(f"Загрузка датасета {dataset_name}, сплит {split}")
        self.dataset = load_dataset(dataset_name, split=split)
        self.df = pd.DataFrame(self.dataset)
        self.target_df = None
        logger.info(f"Загружено {len(self.df)} документов")

    def print_dataset_info(self):
        """Вывод информации о загруженном датасете."""
        logger.debug("Вывод информации о датасете")
        print("\n--- Информация о датасете ---")
        print(f"Колонки: {self.df.columns.tolist()}")
        print(f"Размер: {self.df.shape}")
        print(f"\nПервые 3 строки:")
        print(self.df.head(3))
        print(f"\nСтатистика:")
        print(self.df.describe(include='all'))

    def fill_target_dataset(self, text_field: str = 'source', category_field: str = 'category'):
        """
        Создание целевого датасета для работы системы.

        Args:
            text_field: название поля с текстом документа
            category_field: название поля с категорией документа
        """
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

        logger.info(f"Создан датасет: {len(self.target_df)} записей, "
                    f"уникальных категорий: {self.target_df['category'].nunique()}")
        return self.target_df

    def save_target_dataset(self, path: str = 'data/processed/dataset.parquet'):
        """Сохранение целевого датасета в Parquet."""
        if self.target_df is None:
            raise ValueError("Целевой датасет не создан. Сначала вызовите fill_target_dataset().")

        logger.info(f"Сохранение датасета в {path}")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.target_df.to_parquet(path, index=False)

        size_mb = path.stat().st_size / (1024 * 1024)
        logger.info(f"Сохранено: {path} ({size_mb:.2f} MB)")

    def load_target_dataset(self, path: str = "data/processed/dataset.parquet"):
        """Загрузка ранее сохранённого целевого датасета."""
        logger.info(f"Загрузка датасета из {path}")
        self.target_df = pd.read_parquet(path)
        logger.info(f"Загружено {len(self.target_df)} записей")
        return self.target_df
