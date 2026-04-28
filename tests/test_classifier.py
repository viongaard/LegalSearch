"""
Тестирование классификатора в разных режимах.

Режимы работы:
1. test   - тестирование на test выборке (CSV)
2. single - ручной ввод текста
3. file   - классификация текста из файла
4. folder - классификация всех файлов в папке

Запуск:
  python tests/test_classifier.py --mode test
  python tests/test_classifier.py --mode single
  python tests/test_classifier.py --mode file --path ./document.txt
  python tests/test_classifier.py --mode folder --path ./my_docs/
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
import torch
from pathlib import Path
import pandas as pd
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

# Категории классификатора
CLASSES = [
    'sud_resh_trud_pravo', 'sud_resh_sem_pravo', 'sud_resh_admin_pravo',
    'sud_resh_ugol_pravo', 'sud_resh_grazhd_pravo', 'sud_resh_eco_pravo',
    'sud_resh_const_pravo', 'sud_resh_soc_pravo', 'sud_resh_zhil_pravo',
    'sud_resh_fin_pravo'
]

# Читабельные названия на русском
CLASS_NAMES_RU = {
    'sud_resh_trud_pravo': 'Трудовое право',
    'sud_resh_sem_pravo': 'Семейное право',
    'sud_resh_admin_pravo': 'Административное право',
    'sud_resh_ugol_pravo': 'Уголовное право',
    'sud_resh_grazhd_pravo': 'Гражданское право',
    'sud_resh_eco_pravo': 'Экологическое право',
    'sud_resh_const_pravo': 'Конституционное право',
    'sud_resh_soc_pravo': 'Социальное право',
    'sud_resh_zhil_pravo': 'Жилищное право',
    'sud_resh_fin_pravo': 'Финансовое право'
}

CLASS_TO_ID = {cls: i for i, cls in enumerate(CLASSES)}


def load_model(model_path: str, device: str):
    """Загружает модель и токенизатор."""
    logger.info(f"Загрузка модели из {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()
    return model, tokenizer


def predict_single(model, tokenizer, text: str, device: str = "cpu", max_length: int = 512):
    """
    Классифицирует один текст.
    Возвращает: predicted_class, confidence, все вероятности
    """
    # Токенизация с урезанием длинных текстов
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        confidence, pred = torch.max(probs, dim=1)

    predicted_class = CLASSES[pred.item()]
    confidence_value = confidence.item()

    # Все вероятности по классам
    all_probs = {CLASSES[i]: probs[0][i].item() for i in range(len(CLASSES))}

    return predicted_class, confidence_value, all_probs


def predict_batch(model, tokenizer, texts, batch_size: int = 32, device: str = "cpu", max_length: int = 512):
    """Пакетная классификация."""
    all_predictions = []
    all_confidences = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            confidences, preds = torch.max(probs, dim=1)
            all_predictions.extend(preds.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())

    predicted_classes = [CLASSES[p] for p in all_predictions]
    return predicted_classes, all_confidences


def print_result(predicted_class, confidence, all_probs=None):
    """Красивый вывод результата классификации."""
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТ КЛАССИФИКАЦИИ")
    print("=" * 60)
    print(f"Предсказанная категория: {predicted_class}")
    print(f"Название на русском:    {CLASS_NAMES_RU.get(predicted_class, predicted_class)}")
    print(f"Уверенность:            {confidence:.2%}")

    if all_probs:
        print("\nВероятности по всем категориям:")
        print("-" * 40)
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        for cls, prob in sorted_probs:
            bar = "█" * int(prob * 30)
            ru_name = CLASS_NAMES_RU.get(cls, cls)
            print(f"  {ru_name:25} {bar} {prob:.2%}")
    print("=" * 60)


# ============================================================================
# РЕЖИМ 1: Тестирование на test выборке (CSV)
# ============================================================================
def run_test_mode(model_path: str, test_data_path: str, batch_size: int = 32):
    """Режим тестирования: сравнивает с эталонными метками из CSV."""
    logger.info(f"=== РЕЖИМ ТЕСТИРОВАНИЯ ===")
    logger.info(f"Загрузка тестовых данных из {test_data_path}")

    test_df = pd.read_csv(test_data_path)

    # Проверка наличия нужных колонок
    if 'query_text' not in test_df.columns or 'category' not in test_df.columns:
        logger.error("CSV должен содержать колонки 'query_text' и 'category'")
        return

    test_texts = test_df['query_text'].tolist()
    test_labels = [CLASS_TO_ID[c] for c in test_df['category'].tolist()]

    logger.info(f"Загружено {len(test_texts)} тестовых примеров")

    # Статистика по классам
    unique, counts = np.unique(test_labels, return_counts=True)
    for u, c in zip(unique, counts):
        logger.info(f"  {CLASSES[u]}: {c}")

    # Загрузка модели
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Устройство: {device}")
    model, tokenizer = load_model(model_path, device)

    # Предсказания
    logger.info(f"Предсказание для {len(test_texts)} примеров...")
    predictions, _ = predict_batch(model, tokenizer, test_texts, batch_size, device)
    pred_ids = [CLASS_TO_ID[p] for p in predictions]

    # Метрики
    acc = accuracy_score(test_labels, pred_ids)
    precision = precision_score(test_labels, pred_ids, average='weighted', zero_division=0)
    recall = recall_score(test_labels, pred_ids, average='weighted', zero_division=0)
    f1 = f1_score(test_labels, pred_ids, average='weighted', zero_division=0)

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ НА ТЕСТОВОЙ ВЫБОРКЕ")
    print("=" * 60)
    print(f"Accuracy:  {acc:.4f} ({acc:.2%})")
    print(f"Precision: {precision:.4f} ({precision:.2%})")
    print(f"Recall:    {recall:.4f} ({recall:.2%})")
    print(f"F1-score:  {f1:.4f} ({f1:.2%})")

    print(f"\n{classification_report(test_labels, pred_ids, target_names=CLASSES, zero_division=0)}")

    # Анализ ошибок
    errors = []
    for i, (true, pred) in enumerate(zip(test_labels, pred_ids)):
        if true != pred:
            errors.append({
                'text': test_texts[i][:100] + "...",
                'true': CLASSES[true],
                'pred': CLASSES[pred]
            })

    logger.info(f"\nОшибок: {len(errors)} из {len(test_texts)} ({len(errors) / len(test_texts) * 100:.1f}%)")

    if errors:
        logger.info("\nТоп-10 ошибок:")
        for err in errors[:10]:
            logger.info(f"  {err['true']} → {err['pred']}: {err['text']}")


# ============================================================================
# РЕЖИМ 2: Ручной ввод текста
# ============================================================================
def run_single_mode(model_path: str):
    """Режим ручного ввода: пользователь вводит текст в консоли."""
    logger.info("=== РЕЖИМ РУЧНОГО ВВОДА ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(model_path, device)

    print("\nВведите текст для классификации (или 'exit' для выхода):")
    print("Подсказка: можно вставить полный текст судебного решения")
    print("-" * 60)

    while True:
        print("\n> ", end="")
        text = input().strip()

        if text.lower() in ['exit', 'quit', 'выход']:
            print("До свидания!")
            break

        if not text:
            print("Текст не может быть пустым. Попробуйте ещё раз.")
            continue

        # Классификация
        predicted_class, confidence, all_probs = predict_single(model, tokenizer, text, device)
        print_result(predicted_class, confidence, all_probs)


# ============================================================================
# РЕЖИМ 3: Классификация текста из файла
# ============================================================================
def read_text_from_file(file_path: str) -> str:
    """Читает текст из файла (поддерживает .txt, .docx, .pdf частично)."""
    file_path = Path(file_path)

    if file_path.suffix.lower() == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        # Для других форматов пока только txt
        raise ValueError(f"Поддерживаются только .txt файлы. Получен: {file_path.suffix}")
        # TODO: добавить поддержку .docx и .pdf при необходимости


def run_file_mode(model_path: str, file_path: str):
    """Режим файла: классифицирует текст из указанного файла."""
    logger.info(f"=== РЕЖИМ ФАЙЛА ===")
    logger.info(f"Файл: {file_path}")

    if not os.path.exists(file_path):
        logger.error(f"Файл не найден: {file_path}")
        return

    try:
        text = read_text_from_file(file_path)
        logger.info(f"Загружено {len(text)} символов, ~{len(text) // 4} токенов")

        # Обрезаем слишком длинные тексты для отображения
        preview = text[:500] + "..." if len(text) > 500 else text
        logger.info(f"Превью:\n{preview}")

    except Exception as e:
        logger.error(f"Ошибка чтения файла: {e}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(model_path, device)

    predicted_class, confidence, all_probs = predict_single(model, tokenizer, text, device)
    print_result(predicted_class, confidence, all_probs)


# ============================================================================
# РЕЖИМ 4: Классификация всех файлов в папке
# ============================================================================
def run_folder_mode(model_path: str, folder_path: str, extensions: list = None):
    """Режим папки: классифицирует все .txt файлы в папке."""
    if extensions is None:
        extensions = ['.txt']

    logger.info(f"=== РЕЖИМ ПАПКИ ===")
    logger.info(f"Папка: {folder_path}")

    if not os.path.exists(folder_path):
        logger.error(f"Папка не найдена: {folder_path}")
        return

    # Собираем все файлы с нужными расширениями
    files = []
    for ext in extensions:
        files.extend(Path(folder_path).rglob(f"*{ext}"))

    if not files:
        logger.warning(f"Не найдено файлов с расширениями {extensions} в {folder_path}")
        return

    logger.info(f"Найдено {len(files)} файлов")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(model_path, device)

    results = []

    for file_path in files:
        try:
            text = read_text_from_file(file_path)
            if not text or len(text) < 50:
                logger.warning(f"Пропущен (слишком короткий): {file_path.name}")
                continue

            predicted_class, confidence, _ = predict_single(model, tokenizer, text, device)
            results.append({
                'filename': file_path.name,
                'category': predicted_class,
                'category_ru': CLASS_NAMES_RU.get(predicted_class, predicted_class),
                'confidence': confidence,
                'length_chars': len(text)
            })
            logger.info(f"  {file_path.name:40} → {predicted_class:30} ({confidence:.2%})")

        except Exception as e:
            logger.error(f"Ошибка при обработке {file_path.name}: {e}")

    # Сводная статистика
    print("\n" + "=" * 60)
    print("СВОДНАЯ СТАТИСТИКА ПО ПАПКЕ")
    print("=" * 60)

    if results:
        df_results = pd.DataFrame(results)
        print(f"\nВсего обработано: {len(results)} файлов")
        print("\nРаспределение по категориям:")
        category_counts = df_results['category_ru'].value_counts()
        for cat, count in category_counts.items():
            print(f"  {cat}: {count} файлов ({count / len(results) * 100:.1f}%)")

        # Сохраняем результаты
        output_path = Path(folder_path) / "classification_results.csv"
        df_results.to_csv(output_path, index=False, encoding='utf-8')
        print(f"\nРезультаты сохранены в: {output_path}")
    else:
        print("Нет успешно обработанных файлов")


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Тестирование классификатора судебных решений')

    parser.add_argument(
        '--mode', '-m',
        type=str,
        required=True,
        choices=['test', 'single', 'file', 'folder'],
        help='Режим работы: test (CSV-тест), single (ручной ввод), file (файл), folder (папка)'
    )

    parser.add_argument(
        '--model-path',
        type=str,
        default='models/classifier_v2',
        help='Путь к папке с обученной моделью'
    )

    parser.add_argument(
        '--test-data-path',
        type=str,
        default='data/test_from_dataset.csv',
        help='Путь к CSV с тестовыми данными (для режима test)'
    )

    parser.add_argument(
        '--path', '-p',
        type=str,
        help='Путь к файлу или папке (для режимов file и folder)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Размер батча для режима test'
    )

    parser.add_argument(
        '--extensions',
        type=str,
        nargs='+',
        default=['.txt'],
        help='Расширения файлов для режима folder (например .txt .docx)'
    )

    args = parser.parse_args()

    # Проверка путей для режимов file и folder
    if args.mode in ['file', 'folder'] and not args.path:
        parser.error(f"Для режима {args.mode} требуется указать --path")

    # Запуск соответствующего режима
    if args.mode == 'test':
        run_test_mode(args.model_path, args.test_data_path, args.batch_size)

    elif args.mode == 'single':
        run_single_mode(args.model_path)

    elif args.mode == 'file':
        run_file_mode(args.model_path, args.path)

    elif args.mode == 'folder':
        run_folder_mode(args.model_path, args.path, args.extensions)


if __name__ == "__main__":
    main()