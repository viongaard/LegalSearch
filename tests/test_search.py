"""
Тестирование поисковой системы с расчётом метрик
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import time

from src.search import SearchEngine


def load_test_queries(csv_path: str) -> pd.DataFrame:
    """
    Загрузка тестовых запросов с разметкой

    Ожидаемый формат CSV:
    query_text,expected_category,expected_doc_ids,is_legal
    """
    df = pd.read_csv(csv_path)
    # Парсим expected_doc_ids (если строка вида "doc1,doc2,doc3")
    if 'expected_doc_ids' in df.columns:
        df['expected_doc_ids'] = df['expected_doc_ids'].apply(
            lambda x: [d.strip() for d in str(x).split(',')] if pd.notna(x) else []
        )
    return df


def calculate_classification_metrics(
        queries_df: pd.DataFrame,
        search_engine: SearchEngine
) -> Dict[str, Any]:
    """
    Расчёт метрик классификации на тестовых запросах
    """
    print("ТЕСТИРОВАНИЕ КЛАССИФИКАТОРА")

    y_true = []
    y_pred = []
    y_conf = []
    unknown_stats = {'legal_recognized': 0, 'legal_as_unknown': 0, 'illegal_as_legal': 0, 'illegal_correct': 0}

    for _, row in queries_df.iterrows():
        query = row['query_text']
        true_category = row.get('expected_category', 'unknown')
        is_legal = row.get('is_legal', True)

        # Получаем предсказание
        classification = search_engine.query_classifier.predict(query)
        pred_category = classification['primary_category']
        confidence = classification['primary_confidence']
        is_reliable = classification['is_reliable']

        y_true.append(true_category)
        y_pred.append(pred_category)
        y_conf.append(confidence)

        # Статистика по unknown
        if is_legal:
            if pred_category != 'unknown':
                unknown_stats['legal_recognized'] += 1
            else:
                unknown_stats['legal_as_unknown'] += 1
        else:
            if pred_category != 'unknown':
                unknown_stats['illegal_as_legal'] += 1
            else:
                unknown_stats['illegal_correct'] += 1

    # Расчёт метрик
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'avg_confidence': np.mean(y_conf),
        'unknown_stats': unknown_stats
    }

    # Вывод
    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"F1 (weighted): {metrics['f1_weighted']:.4f}")
    print(f"Precision (weighted): {metrics['precision_weighted']:.4f}")
    print(f"Recall (weighted): {metrics['recall_weighted']:.4f}")
    print(f"Средняя уверенность: {metrics['avg_confidence']:.4f}")

    print(f"\nСтатистика unknown:")
    print(f"  Юридические запросы распознаны: {unknown_stats['legal_recognized']}")
    print(f"  Юридические запросы ошибочно unknown: {unknown_stats['legal_as_unknown']}")
    print(f"  Мусорные запросы ошибочно классифицированы: {unknown_stats['illegal_as_legal']}")
    print(f"  Мусорные запросы верно unknown: {unknown_stats['illegal_correct']}")

    return metrics


def calculate_search_metrics(
        queries_df: pd.DataFrame,
        search_engine: SearchEngine,
        k: int = 5
) -> Dict[str, Any]:
    """
    Расчёт метрик поиска (MRR, Recall@K, Precision@K)
    """
    print(f"ТЕСТИРОВАНИЕ ПОИСКА (k={k})")

    mrr_sum = 0.0
    recall_sum = 0.0
    precision_sum = 0.0

    ndcg_sum = 0.0

    query_times = []
    filter_ratios = []
    n_queries = 0

    for _, row in queries_df.iterrows():
        query = row['query_text']
        expected_doc_ids = row.get('expected_doc_ids', [])

        if not expected_doc_ids:
            continue

        n_queries += 1

        # Поиск
        start = time.time()
        result = search_engine.search(query, k=k)
        elapsed = time.time() - start
        query_times.append(elapsed)

        # Получаем ID найденных документов
        retrieved_ids = [r['chunk_id'] for r in result['results']]

        # MRR
        reciprocal_rank = 0.0
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_doc_ids:
                reciprocal_rank = 1.0 / (i + 1)
                break
        mrr_sum += reciprocal_rank

        # Recall@K
        relevant_in_topk = len([d for d in retrieved_ids if d in expected_doc_ids])
        recall_sum += relevant_in_topk / len(expected_doc_ids)

        # Precision@K
        precision_sum += relevant_in_topk / k

        # NDCG@K
        dcg = 0.0
        idcg = 0.0
        for i, doc_id in enumerate(retrieved_ids):
            relevance = 1 if doc_id in expected_doc_ids else 0
            dcg += relevance / np.log2(i + 2)
        for i in range(min(len(expected_doc_ids), k)):
            idcg += 1 / np.log2(i + 2)
        ndcg_sum += dcg / idcg if idcg > 0 else 0

        # Фильтрация
        if result.get('filtered_categories'):
            before = result['metrics']['candidates_before_filter']
            after = result['metrics']['candidates_after_filter']
            if before > 0:
                filter_ratios.append((before - after) / before * 100)

    metrics = {
        f'mrr@{k}': mrr_sum / n_queries if n_queries > 0 else 0,
        f'recall@{k}': recall_sum / n_queries if n_queries > 0 else 0,
        f'precision@{k}': precision_sum / n_queries if n_queries > 0 else 0,
        f'ndcg@{k}': ndcg_sum / n_queries if n_queries > 0 else 0,
        'avg_query_time_sec': np.mean(query_times) if query_times else 0,
        'avg_filter_ratio_percent': np.mean(filter_ratios) if filter_ratios else 0,
        'queries_processed': n_queries
    }

    print(f"\nMRR@{k}: {metrics[f'mrr@{k}']:.4f}")
    print(f"Recall@{k}: {metrics[f'recall@{k}']:.4f}")
    print(f"Precision@{k}: {metrics[f'precision@{k}']:.4f}")
    print(f"NDCG@{k}: {metrics[f'ndcg@{k}']:.4f}")
    print(f"Среднее время запроса: {metrics['avg_query_time_sec']:.2f} сек")
    print(f"Среднее сокращение поиска: {metrics['avg_filter_ratio_percent']:.1f}%")

    return metrics


def run_full_test(config_path: str, test_data_path: str):
    """
    Полный цикл тестирования
    """
    print("\n" + "=" * 70)
    print("НАЧАЛО ТЕСТИРОВАНИЯ ПОИСКОВОЙ СИСТЕМЫ")
    print("=" * 70)

    # Загрузка конфига
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Инициализация поисковой системы
    print("\nИнициализация...")
    search_engine = SearchEngine(config)

    # Загрузка тестовых запросов
    test_queries = load_test_queries(test_data_path)
    print(f"\nЗагружено {len(test_queries)} тестовых запросов")

    # Тестирование классификатора
    if search_engine.use_classifier:
        class_metrics = calculate_classification_metrics(test_queries, search_engine)
    else:
        class_metrics = None

    # Тестирование поиска
    search_metrics = calculate_search_metrics(test_queries, search_engine, k=5)

    # Итоговый отчёт
    print("\n" + "=" * 70)
    print("ИТОГОВЫЙ ОТЧЁТ ТЕСТИРОВАНИЯ")
    print("=" * 70)

    if class_metrics:
        print(f"\nКлассификатор:")
        print(f"  Accuracy: {class_metrics['accuracy']:.4f}")
        print(f"  F1 (weighted): {class_metrics['f1_weighted']:.4f}")

    print(f"\nПоиск (RAG):")
    print(f"  MRR@5: {search_metrics['mrr@5']:.4f}")
    print(f"  Recall@5: {search_metrics['recall@5']:.4f}")
    print(f"  Precision@5: {search_metrics['precision@5']:.4f}")

    print(f"\nПроизводительность:")
    print(f"  Среднее время ответа: {search_metrics['avg_query_time_sec']:.2f} сек")
    print(f"  Сокращение поиска: {search_metrics['avg_filter_ratio_percent']:.1f}%")

    return {
        'classification_metrics': class_metrics,
        'search_metrics': search_metrics
    }


if __name__ == '__main__':
    # Путь к конфигу и тестовым данным
    CONFIG_PATH = "../config/search.yaml"
    TEST_DATA_PATH = "tests/test_data/test_queries.csv"

    results = run_full_test(CONFIG_PATH, TEST_DATA_PATH)