#!/home/user-ml-srv/venv_mlops_ml_srv/bin/python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys

# Генерация случайного датасета
def generate_random_dataset():
    np.random.seed(42)
    n_samples = 200  # Количество наблюдений
    data = {
        'Feature1': np.random.rand(n_samples),
        'Feature2': np.random.choice(['A', 'B', 'C'], size=n_samples),
        'Target': np.random.randint(0, 2, size=n_samples)
    }
    df = pd.DataFrame(data)
    return df

dataset = generate_random_dataset()
dataset.to_csv('../../data/dataset.csv', index=False)

train_df, test_df = train_test_split(dataset, test_size=0.2, random_state=42)
train_df.to_csv('../../data/raw/train.csv', index=False)
test_df.to_csv('../../data/raw/test.csv', index=False)

print("Сгенерированный датасет сохранен в dataset.csv")
print("Тренировочная и тестовая выборки сохранены в train.csv и test.csv")
