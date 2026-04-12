"""
Тестирование дообученного классификатора на test выборке.
Запуск: python tests/test_classifier.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import logging
import argparse
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CLASSES = [
    'sud_resh_trud_pravo', 'sud_resh_sem_pravo', 'sud_resh_admin_pravo',
    'sud_resh_ugol_pravo', 'sud_resh_grazhd_pravo', 'sud_resh_eco_pravo',
    'sud_resh_const_pravo', 'sud_resh_soc_pravo', 'sud_resh_zhil_pravo',
    'sud_resh_fin_pravo'
]
CLASS_TO_ID = {cls: i for i, cls in enumerate(CLASSES)}


def predict_batch(model, tokenizer, texts, batch_size: int = 32, device: str = "cpu"):
    model.eval()
    all_predictions = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            batch_preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_predictions.extend(batch_preds)

    return np.array(all_predictions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='models/classifier_v2')
    parser.add_argument('--test-data-path', type=str, default='data/test_from_dataset.csv')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    # 1. Загрузка тестовых данных
    logger.info(f"Загрузка тестовых данных из {args.test_data_path}")
    test_df = pd.read_csv(args.test_data_path)

    test_texts = test_df['query_text'].tolist()
    test_labels = [CLASS_TO_ID[c] for c in test_df['category'].tolist()]

    logger.info(f"Загружено {len(test_texts)} тестовых примеров")

    # Статистика по классам
    unique, counts = np.unique(test_labels, return_counts=True)
    for u, c in zip(unique, counts):
        logger.info(f"  {CLASSES[u]}: {c}")

    # 2. Загрузка модели
    logger.info(f"Загрузка модели из {args.model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Устройство: {device}")

    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # 3. Предсказания
    logger.info(f"Предсказание для {len(test_texts)} примеров...")
    predictions = predict_batch(model, tokenizer, test_texts, args.batch_size, device)

    # 4. Метрики
    acc = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(test_labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(test_labels, predictions, average='weighted', zero_division=0)

    logger.info("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ НА ТЕСТОВОЙ ВЫБОРКЕ")
    logger.info(f"Accuracy:  {acc:.4f} ({acc:.2%})")
    logger.info(f"Precision: {precision:.4f} ({precision:.2%})")
    logger.info(f"Recall:    {recall:.4f} ({recall:.2%})")
    logger.info(f"F1-score:  {f1:.4f} ({f1:.2%})")

    logger.info(f"\n{classification_report(test_labels, predictions, target_names=CLASSES, zero_division=0)}")

    # 5. Анализ ошибок
    errors = []
    for i, (true, pred) in enumerate(zip(test_labels, predictions)):
        if true != pred:
            errors.append({
                'text': test_texts[i][:100] + "...",
                'true': CLASSES[true],
                'pred': CLASSES[pred]
            })

    logger.info(f"\nОшибок: {len(errors)} из {len(test_texts)} ({len(errors)/len(test_texts)*100:.1f}%)")

    if errors:
        logger.info("\nТоп-5 ошибок:")
        for err in errors[:5]:
            logger.info(f"  {err['true']} → {err['pred']}: {err['text']}")


if __name__ == "__main__":
    main()