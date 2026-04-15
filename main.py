import argparse
import yaml
import logging
import sys

from src.indexing import Indexing
from src.search import Search

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _print_analysis_report(result: dict):
    """Печать аналитического отчёта search_with_analysis"""
    print(f"\n{'=' * 70}")
    print(f"🔍 ЗАПРОС: {result['query']}")
    print('=' * 70)

    # Классификация
    cls = result.get('classification', {})
    if cls:
        print(
            f"\n📌 КАТЕГОРИЯ: {cls.get('primary_category', 'unknown')} (уверенность: {cls.get('primary_confidence', 0):.2%})")
        if cls.get('expanded_categories'):
            print(f"   Расширенные: {', '.join(cls['expanded_categories'])}")

    # LLM-аналитика
    llm = result.get('llm_analysis')
    if llm:
        print("\n🤖 АНАЛИТИКА (LLM):")
        print(f"📝 {llm.get('summary', 'Нет данных')}")
        if llm.get('key_findings'):
            print("\n🔑 КЛЮЧЕВЫЕ ВЫВОДЫ:")
            for f in llm['key_findings']:
                print(f"   • {f}")
        if llm.get('legal_basis'):
            print(f"\n📚 ПРАВОВАЯ ОСНОВА: {', '.join(llm['legal_basis'])}")

    # Связи
    conn = result.get('connections', {})
    if any(conn.values()):
        print("\n🔗 СВЯЗИ МЕЖДУ ДЕЛАМИ:")
        if conn.get('common_participants'):
            print(f"   • Участники: {', '.join(conn['common_participants'][:5])}")
        if conn.get('common_courts'):
            print(f"   • Суды: {', '.join(conn['common_courts'][:5])}")
        if conn.get('common_judges'):
            print(f"   • Судьи: {', '.join(conn['common_judges'][:5])}")
        if conn.get('common_articles'):
            print(f"   • Статьи: {', '.join(conn['common_articles'][:5])}")
        if conn.get('common_amounts'):
            print(f"   • Суммы: {', '.join(conn['common_amounts'][:5])}")

    # Документы
    docs = result.get('documents', [])
    if docs:
        print(f"\n📄 НАЙДЕННЫЕ ДОКУМЕНТЫ ({len(docs)}):")
        for i, doc in enumerate(docs, 1):
            print(f"\n{i}. Дело № {doc.get('doc_id', 'N/A')} (релевантность: {doc.get('rerank_score', 0):.3f})")
            print(f"   Категория: {doc.get('category', 'N/A')}")
            text = doc.get('text_full', '')[:300]
            if text:
                print(f"   {text}...")

    # Статистика
    stats = result.get('stats', {})
    print(f"\n⚡ Время: {stats.get('time_sec', 0):.2f} сек | Документов: {stats.get('documents_found', 0)}")
    print('=' * 70 + '\n')


def _print_results(result: dict):
    """Печать результатов базового поиска run()"""
    print(f"\nЗапрос: {result['query']}")

    classification = result.get('classification')
    if classification:
        print(f"\nКатегория: {classification['primary_category']}")
        print(f"   Уверенность: {classification['primary_confidence']:.2%}")
        if classification.get('expanded_categories'):
            print(f"   Расширенные категории: {', '.join(classification['expanded_categories'])}")

    metrics = result.get('metrics', {})
    print(f"\nВремя выполнения: {metrics.get('time_sec', 0):.2f} сек")
    print(f"Найдено кандидатов: {metrics.get('candidates_found', 0)}")

    results_list = result.get('results', [])
    print(f"\nРезультаты (топ-{len(results_list)}):")

    for r in results_list:
        print(f"\n{r['rank']}. Документ: {r['doc_id']}")
        print(f"   Категория: {r['category']}")
        print(f"   Ранжировочный балл: {r.get('rerank_score', r.get('faiss_score', 0)):.4f}")
        print(f"   Текст: {r['text']}")

    print("\n" + "=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Система семантического поиска судебных решений')
    parser.add_argument('--mode', choices=['index', 'search'], required=True,
                        help='Режим работы: index - индексация документов, search - поиск')
    parser.add_argument('--config', type=str, default='config/indexing.yaml',
                        help='Путь к конфигурационному файлу (для режима index)')
    parser.add_argument('--query', type=str,
                        help='Поисковый запрос (для режима search)')
    parser.add_argument('--max_docs', type=int, default=None,
                        help='Максимальное количество документов для индексации')
    parser.add_argument('--interactive', action='store_true',
                        help='Интерактивный режим поиска')
    parser.add_argument('--no-llm', action='store_true',
                        help='Отключить LLM аналитику (использовать базовый поиск)')

    args = parser.parse_args()

    # Режим индексации
    if args.mode == 'index':
        logger.info("Запуск индексации...")

        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        indexing_pipeline = Indexing(config)
        indexing_pipeline.run(max_docs=args.max_docs)

        logger.info("Индексация завершена")

    # Режим поиска
    elif args.mode == 'search':
        with open('config/search.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        searcher = Search(config)

        # Статистика индекса
        stats = searcher.get_stats()
        logger.info(f"Индекс загружен: {stats['total_vectors']} векторов, {stats['total_chunks']} чанков")

        # Интерактивный режим
        if args.interactive:
            if args.no_llm:
                print("Поиск судебных решений (базовый режим).")
                print("Введите 'exit' для выхода, 'stats' для статистики\n")
                while True:
                    query = input("Запрос: ").strip()
                    if query.lower() == 'exit':
                        break
                    if query.lower() == 'stats':
                        stats = searcher.get_stats()
                        print(f"\nСтатистика индекса:")
                        print(f"   Всего векторов: {stats['total_vectors']}")
                        print(f"   Всего чанков: {stats['total_chunks']}")
                        print(f"   Категории: {stats['categories']}")
                        print()
                        continue
                    if not query:
                        continue
                    result = searcher.run(query)
                    _print_results(result)
            else:
                print("Поиск судебных решений (аналитический режим).")
                print("Введите 'exit' для выхода, 'stats' для статистики\n")
                while True:
                    query = input("Запрос: ").strip()
                    if query.lower() == 'exit':
                        break
                    if query.lower() == 'stats':
                        stats = searcher.get_stats()
                        print(f"\nСтатистика индекса:")
                        print(f"   Всего векторов: {stats['total_vectors']}")
                        print(f"   Всего чанков: {stats['total_chunks']}")
                        print(f"   Категории: {stats['categories']}")
                        print()
                        continue
                    if not query:
                        continue
                    result = searcher.search_with_analysis(query)
                    _print_analysis_report(result)

        # Однократный запрос
        elif args.query:
            if args.no_llm:
                result = searcher.run(args.query)
                _print_results(result)
            else:
                result = searcher.search_with_analysis(args.query)
                _print_analysis_report(result)

        else:
            print("Укажите --query или --interactive для поиска")


if __name__ == '__main__':
    main()