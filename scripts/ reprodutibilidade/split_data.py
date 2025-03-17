import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Definir a seed para garantir reprodutibilidade
np.random.seed(42)

# Carregar o dataset real
df = pd.read_csv("rumos_bank/data/lending_data.csv")

# Definir a variável target (a coluna que queremos prever)
TARGET_COL = "default"  # <-- Alterar isto conforme o dataset

# Criar datasets de treino e teste (80% treino, 20% teste)
train, test = train_test_split(df, test_size=0.2, stratify=df[TARGET_COL])

# Salvar os datasets
train.to_csv("rumos_bank/data/train_data.csv", index=False)
test.to_csv("rumos_bank/data/test_data.csv", index=False)

print(f"Dataset original: {df.shape}")
print(f"Train set: {train.shape}")
print(f"Test set: {test.shape}")
