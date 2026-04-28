"""
Подготовка train/test выборок из датасета sud-resh-benchmark.
Колонка query_text = text + "\n" + correct_answer
Запуск: python prepare_test_split.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_processing.builder import Builder

# 10 юридических классов
CLASSES = [
    'sud_resh_trud_pravo', 'sud_resh_sem_pravo', 'sud_resh_admin_pravo',
    'sud_resh_ugol_pravo', 'sud_resh_grazhd_pravo', 'sud_resh_eco_pravo',
    'sud_resh_const_pravo', 'sud_resh_soc_pravo', 'sud_resh_zhil_pravo',
    'sud_resh_fin_pravo'
]


def main():
    # 1. Загрузка датасета
    print("Загрузка датасета lawful-good-project/sud-resh-benchmark...")
    builder = Builder(
        dataset_name="lawful-good-project/sud-resh-benchmark",
        split="train"
    )
    builder.build_from_dataset(
        text_field="source",
        category_field="category"
    )

    df = builder.target_df

    # Фильтруем только нужные категории
    df = df[df['category'].isin(CLASSES)]

    # Объединяем text и correct_answer в одну колонку
    # Если correct_answer нет, используем только text
    if 'correct_answer' in df.columns:
        df['query_text'] = df['text'].fillna('') + "\n\n" + df['correct_answer'].fillna('')
    else:
        df['query_text'] = df['text']

    # Оставляем нужные колонки
    df = df[['query_text', 'category']].copy()

    print(f"Всего документов в датасете: {len(df)}")

    # 2. Разделение на train/test (85/15)
    train_df, test_df = train_test_split(
        df, test_size=0.15, random_state=42, stratify=df['category']
    )

    print(f"Train выборка: {len(train_df)} (85%)")
    print(f"Test выборка: {len(test_df)} (15%)")

    # 3. Сохраняем train выборку (будет добавлена в queries_categories)
    train_df.to_csv('data/train_from_dataset.csv', index=False)
    print("Сохранено: data/train_from_dataset.csv")

    # 4. Сохраняем test выборку (для финального тестирования)
    test_df.to_csv('data/test_from_dataset.csv', index=False)
    print("Сохранено: data/test_from_dataset.csv")

    # Статистика по классам в test выборке
    print("\nРаспределение классов в test выборке:")
    print(test_df['category'].value_counts())


if __name__ == "__main__":
    main()