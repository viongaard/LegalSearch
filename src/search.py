import time
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

import re
from collections import Counter
from typing import Optional, List, Dict, Any, Tuple

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder

from src.data_processing.embedder import Embedder
from src.data_processing.vector_store import VectorStore
from src.query_classifier import QueryClassifier
from src.llm.client import get_llm_client

logger = logging.getLogger(__name__)


class Search:
    def __init__(self, config: dict):
        # Пути к данным
        self.index_path = Path(config['index_path'])
        self.metadata_path = Path(config['metadata_path'])

        # Параметры поиска
        search_config = config.get('search', {})
        self.retrieve_k = search_config.get('retrieve_k', 100)
        self.rank_k = search_config.get('rank_k', 10)
        self.use_classifier = search_config.get('use_classifier', True)

        # Эмбеддер
        embedder_config = config.get('embedder', {})
        self.embedder = Embedder(
            model_name=embedder_config.get('model_name', 'cointegrated/rubert-tiny2'),
            device=embedder_config.get('device', 'cpu')
        )

        # Ранкер
        ranker_config = config.get('ranker', {})
        self.ranker = CrossEncoder(
            ranker_config.get('model_name', 'DiTy/cross-encoder-russian-msmarco'),
            device=ranker_config.get('device', 'cpu')
        )

        # Классификатор запроса
        query_classifier_config = config.get('query_classifier')
        self.query_classifier = QueryClassifier(
            model_path=query_classifier_config.get('model_path'),
            device=query_classifier_config.get('device', 'cpu'),
            threshold=query_classifier_config.get('threshold'),
            neighbor_threshold=query_classifier_config.get('neighbor_threshold')
        )

        # LLM-клиент
        self.llm_client = get_llm_client()
        if self.llm_client and self.llm_client.available:
            logger.info("LLM клиент инициализирован")
        else:
            logger.info("LLM клиент не доступен")

        # Индекс и метаданные
        self.index = None
        self.metadata = None
        self._load_index()

        logger.info(f"Search инициализирован: retrieve_k={self.retrieve_k}, rank_k={self.rank_k}")

    def _load_index(self):
        """Загрузка индекса faiss и метаданных"""

        if not self.index_path.exists():
            raise FileNotFoundError(f"Индекс не найден: {self.index_path}")

        self.index = faiss.read_index(str(self.index_path))
        self.metadata = pd.read_parquet(self.metadata_path)

        logger.info(f"Загружено {self.index.ntotal} векторов, {len(self.metadata)} чанков")

    def _create_filtered_index(self, target_categories: List[str]) -> Any:
        """
        Создаёт faiss индекс только для векторов нужных категорий.
        Использует reconstruct для извлечения векторов.
        """
        # Маска для фильтрации
        mask = self.metadata['category'].isin(target_categories)
        local_indices = np.where(mask)[0]

        if len(local_indices) == 0:
            logger.warning(f"Нет документов для категорий: {target_categories}")
            return None, []

        # Размерность векторов
        d = self.index.d

        # Реконструкция векторов (можно батчами, но для 10-20k векторов ок)
        vectors = np.zeros((len(local_indices), d), dtype='float32')
        for i, local_idx in enumerate(local_indices):
            vectors[i] = self.index.reconstruct(int(local_idx))

        # Создаём мини-индекс
        mini_index = faiss.IndexFlatIP(d)
        mini_index.add(vectors)

        # Сохраняем соответствие позиция в мини-индексе -> chunk_id
        chunk_ids = self.metadata.iloc[local_indices]['chunk_id'].tolist()

        logger.info(f"Создан мини-индекс для категорий {target_categories}: {len(vectors)} векторов")

        return mini_index, chunk_ids

    def _search_raw(self, query: str) -> Tuple[Dict, pd.DataFrame, Dict, int]:
        """Внутренний метод "сырого" поиска"""
        start_time = time.time()
        logger.info(f"Начало поиска по запросу: {query[:100]}")

        # 1. Классификация запроса
        classification = self.query_classifier.predict(query)
        target_categories = classification.get('expanded_categories', [])
        logger.info(f"Категория: {classification['primary_category']} "
                    f"(уверенность: {classification['primary_confidence']:.2f})")
        logger.info(f"Целевые категории: {target_categories}")

        # 2. Векторизация запроса
        query_emb = self.embedder.encode([query])

        # 3. Создание отфильтрованного индекса
        if target_categories and target_categories[0] != 'unknown':
            mini_index, chunk_ids = self._create_filtered_index(target_categories)

            if mini_index is None or mini_index.ntotal == 0:
                logger.warning("После фильтрации пусто, ищу по всем документам")
                mini_index = self.index
                chunk_ids = self.metadata['chunk_id'].tolist()
        else:
            mini_index = self.index
            chunk_ids = self.metadata['chunk_id'].tolist()

        # 4. Поиск кандидатов
        search_k = min(self.retrieve_k, mini_index.ntotal)
        scores, local_indices = mini_index.search(query_emb, search_k)

        candidate_chunk_ids = [chunk_ids[i] for i in local_indices[0]]
        candidates = self.metadata[self.metadata['chunk_id'].isin(candidate_chunk_ids)].copy()
        score_map = {chunk_ids[i]: scores[0][i] for i in range(len(local_indices[0]))}
        candidates['faiss_score'] = candidates['chunk_id'].map(score_map)
        candidates = candidates.sort_values('faiss_score', ascending=False)

        # 5. Реранжирование
        if len(candidates) > 0:
            candidates = self._rerank(query, candidates)

        elapsed = time.time() - start_time
        metrics = {
            'time_sec': round(elapsed, 2),
            'candidates_found': len(candidates),
            'filtered_index_size': mini_index.ntotal if hasattr(mini_index, 'ntotal') else 0
        }
        filtered_size = mini_index.ntotal if hasattr(mini_index, 'ntotal') else 0
        return classification, candidates, metrics, filtered_size

    def _rerank(self, query: str, candidates: pd.DataFrame) -> pd.DataFrame:
        """Реранжирование через CrossEncoder"""
        pairs = [(query, text) for text in candidates['text'].tolist()]
        scores = self.ranker.predict(pairs, show_progress_bar=False)
        candidates['rerank_score'] = scores
        return candidates.sort_values('rerank_score', ascending=False)

    def _group_chunks_to_documents(self, candidates_df: pd.DataFrame) -> List[Dict]:
        """
        Группирует чанки по doc_id, объединяет текст и сущности.
        Возвращает список документов, отсортированный по rerank_score.
        """
        docs = {}
        for _, row in candidates_df.iterrows():
            doc_id = row['doc_id']
            if doc_id not in docs:
                docs[doc_id] = {
                    'doc_id': doc_id,
                    'category': row['category'],
                    'chunks': [],
                    'rerank_score': row.get('rerank_score', 0),
                    'faiss_score': row.get('faiss_score', 0),
                    'text_full': '',
                    'entities': []
                }
            docs[doc_id]['chunks'].append(row)
            if 'entities' in row and isinstance(row['entities'], list):
                docs[doc_id]['entities'].extend(row['entities'])

        # Объединяем текст и дедуплицируем сущности
        for doc_id in docs:
            chunks_sorted = sorted(docs[doc_id]['chunks'], key=lambda x: x.get('chunk_id', ''))
            docs[doc_id]['text_full'] = '\n'.join([c['text'] for c in chunks_sorted])
            if len(docs[doc_id]['text_full']) > 3000:
                docs[doc_id]['text_full'] = docs[doc_id]['text_full'][:3000] + '...'

            # Дедупликация сущностей
            unique = {}
            for e in docs[doc_id]['entities']:
                key = (e.get('type', ''), e.get('normalized', e.get('text', '')))
                unique[key] = e
            docs[doc_id]['entities'] = list(unique.values())

        result = list(docs.values())
        result.sort(key=lambda x: x['rerank_score'], reverse=True)
        return result[:5]  # топ-5 документов

    def _extract_connections(self, documents: List[Dict]) -> Dict[str, List[str]]:
        """
        Анализирует список документов и возвращает общие сущности.
        Работает с сущностями в формате Natasha (type: PER, ORG, LOC)
        """
        connections = {
            'common_participants': set(),  # PER + ORG
            'common_courts': set(),  # LOC + содержит слово "суд"
            'common_judges': set(),  # PER + рядом слово "судья"
            'common_articles': set(),  # статьи законов
            'common_amounts': set()  # суммы
        }

        article_pattern = r'ст\.?\s*\d+(?:\.\d+)?'
        amount_pattern = r'\b\d{1,3}(?:[ \d]*\d)?\s*(?:млн|тыс|руб|₽|USD|EUR)\b'

        for doc in documents:
            text = doc.get('text_full', '')

            # 1. Обработка сущностей из NER (Natasha)
            for ent in doc.get('entities', []):
                etype = ent.get('type', '')  # PER, ORG, LOC
                value = ent.get('normalized', ent.get('text', ''))
                if not value:
                    continue

                if etype in ('PER', 'ORG'):
                    connections['common_participants'].add(value)
                elif etype == 'LOC':
                    # Проверяем, относится ли локация к суду
                    if 'суд' in value.lower() or 'суд' in ent.get('text', '').lower():
                        connections['common_courts'].add(value)

            # 2. Поиск судей (по тексту, т.к. NER не всегда их выделяет)
            judge_pattern = r'судья\s+([А-Я][а-я]+(?:\s+[А-Я][а-я]+)?)'
            judge_matches = re.findall(judge_pattern, text, re.IGNORECASE)
            for judge in judge_matches:
                connections['common_judges'].add(judge.strip())

            # 3. Поиск статей
            articles = re.findall(article_pattern, text)
            for art in articles:
                connections['common_articles'].add(art.strip())

            # 4. Поиск сумм
            amounts = re.findall(amount_pattern, text, re.IGNORECASE)
            for amt in amounts:
                connections['common_amounts'].add(amt.strip())

        # Превращаем сеты в списки (топ-10)
        return {k: list(v)[:10] for k, v in connections.items()}

    def _generate_llm_analysis(self, query: str, documents: List[Dict], classification: Dict) -> Dict:
        """Генерация аналитического ответа через YandexGPT"""
        if not self.llm_client or not self.llm_client.available:
            # Заглушка если LLM нет
            return {
                'summary': 'LLM клиент не доступен. Подключите YandexGPT для аналитики.',
                'key_findings': [],
                'legal_basis': []
            }

        answer = self.llm_client.generate_answer(query, documents, classification)

        # Парсим ответ LLM (можно попросить LLM возвращать JSON, пока просто возвращаем текст)
        return {
            'summary': answer if answer else 'Не удалось сгенерировать ответ',
            'key_findings': [],
            'legal_basis': [],
            'raw_answer': answer
        }

    def search_simple(self, query: str) -> Dict[str, Any]:
        """Базовый поиск (только список чанков)"""
        classification, candidates, metrics, _ = self._search_raw(query)

        if candidates.empty:
            return {
                'query': query,
                'classification': classification,
                'results': [],
                'metrics': metrics
            }

        return {
            'query': query,
            'classification': classification,
            'results': self._format_results(candidates.head(self.rank_k)),
            'metrics': metrics
        }

    def search_with_analysis(self, query: str) -> Dict[str, Any]:
        """
        Полноценный поиск с:
        - группировкой чанков в цельные документы
        - извлечением связей между делами
        - LLM-аналитикой
        """
        classification, candidates_df, metrics, filtered_size = self._search_raw(query)

        if candidates_df.empty:
            return {
                'query': query,
                'classification': classification,
                'documents': [],
                'connections': {},
                'llm_analysis': None,
                'stats': metrics,
                'error': 'Ничего не найдено'
            }

        # 1. Группируем чанки в документы
        documents = self._group_chunks_to_documents(candidates_df)

        # 2. Извлекаем связи
        connections = self._extract_connections(documents)

        # 3. Генерируем LLM-аналитику
        llm_analysis = self._generate_llm_analysis(query, documents, classification)

        return {
            'query': query,
            'classification': classification,
            'llm_analysis': llm_analysis,
            'documents': documents,
            'connections': connections,
            'stats': {
                'time_sec': metrics['time_sec'],
                'total_candidates': metrics['candidates_found'],
                'filtered_index_size': filtered_size,
                'documents_found': len(documents)
            }
        }

    def _format_results(self, df: pd.DataFrame) -> List[Dict]:
        """Форматирование результатов для базового поиска"""
        results = []
        for i, (_, row) in enumerate(df.iterrows()):
            text = row.get('text', '')
            results.append({
                'rank': i + 1,
                'chunk_id': row.get('chunk_id', ''),
                'doc_id': row.get('doc_id', ''),
                'category': row.get('category', ''),
                'text': text[:500] + '...' if len(str(text)) > 500 else text,
                'faiss_score': float(row.get('faiss_score', 0)),
                'rerank_score': float(row.get('rerank_score', 0)) if 'rerank_score' in row else 0
            })
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Статистика по индексу"""
        return {
            'total_vectors': self.index.ntotal if self.index else 0,
            'total_chunks': len(self.metadata) if self.metadata is not None else 0,
            'categories': self.metadata['category'].value_counts().to_dict() if self.metadata is not None else {}
        }