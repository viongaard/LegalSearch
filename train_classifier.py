"""
Продолжение дообучения rubert-tiny2 на запросном датасете.
Запуск: python train_classifier.py
"""

import json
import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== КОНФИГУРАЦИЯ ==========
# БАЗОВАЯ МОДЕЛЬ — твоя дообученная!
BASE_MODEL_PATH = "models/classifier_v2"  # ← загружаем твою модель
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 40  # мало эпох, чтобы не переобучить
LEARNING_RATE = 1e-5
OUTPUT_DIR = "models/classifier_v3"  # сохраняем в новую папку, не затираем старую
DATA_PATH = "data/queries_categories.csv"

CLASSES = [
    'sud_resh_trud_pravo', 'sud_resh_sem_pravo', 'sud_resh_admin_pravo',
    'sud_resh_ugol_pravo', 'sud_resh_grazhd_pravo', 'sud_resh_eco_pravo',
    'sud_resh_const_pravo', 'sud_resh_soc_pravo', 'sud_resh_zhil_pravo',
    'sud_resh_fin_pravo'
]

NUM_CLASSES = len(CLASSES)
CLASS_TO_ID = {cls: i for i, cls in enumerate(CLASSES)}
ID_TO_CLASS = {i: cls for i, cls in enumerate(CLASSES)}


def load_data(csv_path: str):
    """Загрузка размеченных запросов из CSV"""
    df = pd.read_csv(csv_path)

    if 'query_text' not in df.columns:
        raise ValueError(f"CSV должен содержать колонку 'query_text'")
    if 'category' not in df.columns:
        raise ValueError(f"CSV должен содержать колонку 'category'")

    texts = []
    labels = []
    skipped = 0

    for _, row in df.iterrows():
        cls = row['category']
        if cls == 'unknown':
            skipped += 1
            continue
        if cls not in CLASS_TO_ID:
            logger.warning(f"Неизвестный класс: {cls}, пропускаем")
            skipped += 1
            continue
        texts.append(row['query_text'])
        labels.append(CLASS_TO_ID[cls])

    logger.info(f"Загружено {len(texts)} примеров, пропущено {skipped}")
    return texts, labels


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
        "precision": precision_score(labels, predictions, average="weighted"),
        "recall": recall_score(labels, predictions, average="weighted")
    }


def main():
    # 1. Загрузка данных
    logger.info(f"Загрузка данных из {DATA_PATH}...")
    texts, labels = load_data(DATA_PATH)

    # 2. Разделение на train/val
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    logger.info(f"Train: {len(train_texts)}, Val: {len(val_texts)}")

    train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})

    # 3. Токенизация
    logger.info("Загрузка токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)  # из твоей модели

    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True
    )
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True
    )

    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # 4. Модель — ЗАГРУЖАЕМ ТВОЮ ДООБУЧЕННУЮ МОДЕЛЬ!
    logger.info(f"Загрузка модели из {BASE_MODEL_PATH} (продолжение обучения)...")
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_PATH,  # ← твоя модель, а не оригинальная!
        num_labels=NUM_CLASSES,
        id2label=ID_TO_CLASS,
        label2id=CLASS_TO_ID
    )

    # 5. Настройки обучения (меньший learning rate для fine-tuning)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,  # 1e-5 вместо 2e-5
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=False,
        no_cuda=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 6. Продолжение обучения
    logger.info("Начало продолжения обучения...")
    trainer.train()

    # 7. Оценка
    eval_results = trainer.evaluate()
    logger.info(f"Результаты: {eval_results}")

    # 8. Сохранение
    logger.info(f"Сохранение модели в {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    with open(f"{OUTPUT_DIR}/class_mapping.json", "w", encoding="utf-8") as f:
        json.dump({
            "class_to_id": CLASS_TO_ID,
            "id_to_class": ID_TO_CLASS,
            "classes": CLASSES
        }, f, ensure_ascii=False, indent=2)

    logger.info("Готово!")


if __name__ == "__main__":
    main()