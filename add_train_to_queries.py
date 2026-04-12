"""
Добавляет train выборку из датасета в queries_categories.csv
Запуск: python add_train_to_queries.py
"""

import pandas as pd

# Загружаем существующий queries_categories.csv с правильными параметрами
try:
    queries = pd.read_csv('data/queries_categories.csv', encoding='utf-8')
    print(f"Существующих запросов: {len(queries)}")
except Exception as e:
    print(f"Ошибка при чтении CSV: {e}")
    print("Пробуем с другими параметрами...")
    queries = pd.read_csv(
        'data/queries_categories.csv',
        encoding='utf-8',
        on_bad_lines='skip',  # пропускаем проблемные строки
        engine='python'       # используем Python парсер вместо C
    )
    print(f"Загружено (с пропуском ошибок): {len(queries)}")

# Загружаем train из датасета
train_data = pd.read_csv('data/train_from_dataset.csv', encoding='utf-8')
print(f"Train из датасета: {len(train_data)}")

# Добавляем
combined = pd.concat([queries, train_data], ignore_index=True)

# Удаляем дубликаты по query_text
before = len(combined)
combined = combined.drop_duplicates(subset=['query_text'], keep='first')
print(f"Удалено дубликатов: {before - len(combined)}")

# Сохраняем с правильными параметрами (без лишних проблем)
combined.to_csv('data/queries_categories.csv', index=False, encoding='utf-8')
print(f"Сохранён queries_categories.csv: {len(combined)} записей")