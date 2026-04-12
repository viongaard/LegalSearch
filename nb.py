from datasets import load_dataset
import pandas as pd

ds = load_dataset("lawful-good-project/sud-resh-benchmark")
df = ds['train'].to_pandas()

print(df.info())
print(df.describe())
print(df['category'].unique())