"""
Классификатор запросов на основе fine-tuned rubert-tiny2.
Без обучения на unknown — используется порог уверенности.
"""

import torch
import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)


class QueryClassifier:
    """Классификатор запросов"""

    def __init__(
        self,
        model_path: str = "models/classifier",
        device: str = "cpu",
        threshold: float = 0.5,
        neighbor_threshold: float = 0.15
    ):
        self.model_path = Path(model_path)
        self.device = device
        self.threshold = threshold
        self.neighbor_threshold = neighbor_threshold

        self.model = None
        self.tokenizer = None
        self.classes = None
        self.id_to_class = None

        self._load()

    def _load(self):
        """Загрузка fine-tuned модели."""
        if not self.model_path.exists():
            logger.warning(f"Модель не найдена: {self.model_path}")
            logger.info("Сначала запустите train_classifier.py")
            return

        logger.info(f"Загрузка классификатора из {self.model_path}")

        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_path)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.model.to(self.device)
        self.model.eval()

        # Загрузка маппинга классов
        mapping_path = self.model_path / "class_mapping.json"
        if mapping_path.exists():
            with open(mapping_path, "r", encoding="utf-8") as f:
                mapping = json.load(f)
            self.id_to_class = {int(k): v for k, v in mapping["id_to_class"].items()}
            self.classes = mapping["classes"]
        else:
            self.id_to_class = self.model.config.id2label
            self.classes = list(self.id_to_class.values())

        logger.info(f"Загружено {len(self.classes)} юридических классов")

    def predict(self, query: str) -> Dict[str, Any]:
        """
        Предсказание категории запроса с порогом.

        :returns
            {
                'primary_category': str,      # один из 10 классов или 'unknown'
                'primary_confidence': float,  # степень уверенности (0-1)
                'expanded_categories': list,  # все классы > neighbor_threshold
                'all_probabilities': dict,    # полное распределение
                'is_reliable': bool           # уверенность >= threshold
            }
        """
        if self.model is None:
            return self._empty_result()

        # Токенизация
        inputs = self.tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding="max_length"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Инференс
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        # Формирование результата
        all_probs = {
            self.id_to_class[i]: float(probs[i])
            for i in range(len(probs))
        }

        # Сортировка по убыванию
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)

        best_class = sorted_probs[0][0]
        best_confidence = sorted_probs[0][1]

        # ===== КЛЮЧЕВОЙ МОМЕНТ: порог для unknown =====
        if best_confidence < self.threshold:
            primary_category = "unknown"
            is_reliable = False
        else:
            primary_category = best_class
            is_reliable = True

        # Расширенные категории (все, что выше порога neighbor_threshold)
        expanded = [
            cat for cat, conf in sorted_probs
            if conf >= self.neighbor_threshold
        ]

        return {
            'primary_category': primary_category,
            'primary_confidence': best_confidence,
            'expanded_categories': expanded,
            'all_probabilities': all_probs,
            'is_reliable': is_reliable
        }

    def filter_by_categories(self, df: pd.DataFrame, categories: List[str]) -> pd.DataFrame:
        """
        Фильтрация DataFrame по списку категорий.
        """
        if df.empty:
            return df

        if not categories or categories[0] == 'unknown':
            return df

        # Прямое совпадение
        mask = df['category'].isin(categories)

        # Если не нашлось — пробуем частичное (для случаев, когда категории в данных шире)
        if not mask.any():
            for cat in categories:
                part_mask = df['category'].str.contains(cat, case=False, na=False)
                mask = mask | part_mask

        return df[mask]

    def _empty_result(self) -> Dict[str, Any]:
        return {
            'primary_category': 'unknown',
            'primary_confidence': 0.0,
            'expanded_categories': [],
            'all_probabilities': {},
            'is_reliable': False
        }