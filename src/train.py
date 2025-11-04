"""
train.py
---------
Script para treinar um modelo de regressão linear a partir do dataset de casas.

Descrição:
- Carrega dados usando utils.load_data
- Valida os dados com utils.validate_data
- Separa features/target, divide treino/teste
- Treina um sklearn.linear_model.LinearRegression
- Calcula R² no conjunto de teste
- Persiste o modelo (joblib) e as métricas (JSON)

Entradas esperadas:
- data/houses.csv (caminho relativo)

Saídas geradas:
- model.joblib   -> modelo treinado
- metrics.json   -> {"r2": <float>}

Observações:
- Este script pode ser executado diretamente: python src/train.py
"""
from typing import Tuple
from utils import load_data, validate_data
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import joblib
import json

DATA_PATH = "data/houses.csv"
MODEL_PATH = "model.joblib"
METRICS_PATH = "metrics.json"
# Parâmetros de configuração padrão usados pelo script.
# TEST_SIZE: fração do dataset reservada para o conjunto de teste.
#            Deve estar entre 0.0 e 1.0. Ex.: 0.2 => 20% dos dados para teste.
# RANDOM_STATE: semente (int) para garantir reprodutibilidade do
#               train_test_split. Use None para comportamento aleatório.
TEST_SIZE = 0.2
RANDOM_STATE = 42  # garante reprodutibilidade do train_test_split

def train_and_persist(data_path: str = DATA_PATH,
                      model_path: str = MODEL_PATH,
                      metrics_path: str = METRICS_PATH,
                      test_size: float = TEST_SIZE,
                      random_state: int = RANDOM_STATE) -> Tuple[LinearRegression, float]:
    """
    Executa o fluxo de treinamento e persiste artefatos.

    Retorna:
    - model: modelo treinado (LinearRegression)
    - r2: pontuação R² no conjunto de teste
    """
    print("🚀 Iniciando treinamento do modelo...")

    # Carrega e valida os dados
    df: pd.DataFrame = load_data(data_path)
    validate_data(df)

    # Separa features/target
    X = df.drop("price", axis=1)
    y = df["price"]

    # Divide treino/teste com semente para reprodutibilidade
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Instancia e treina o modelo
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predição e avaliação
    preds = model.predict(X_test)
    score = float(r2_score(y_test, preds))

    # Persiste o modelo e as métricas
    joblib.dump(model, model_path)
    with open(metrics_path, "w") as f:
        json.dump({"r2": score}, f, indent=2)

    print(f"✅ Treinamento concluído. R² = {score:.4f}")
    return model, score


if __name__ == "__main__":
    train_and_persist()
