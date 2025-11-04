"""
evaluate.py
-----------
Script para avaliar o modelo de regressão linear treinado usando o dataset completo.

Descrição:
- Carrega o modelo persistido (model.joblib)
- Carrega os dados originais (data/houses.csv)
- Calcula o R² (coeficiente de determinação) das predições
- Persiste as métricas em metrics.json

Entradas esperadas:
- model.joblib       -> modelo treinado
- data/houses.csv    -> dataset completo

Saídas geradas:
- metrics.json       -> {"r2": <float>}

Interpretação do R² Score:
--------------------------
O R² (coeficiente de determinação) mede a proporção da variância da variável
dependente que é explicada pelo modelo. Varia de -∞ a 1.

Valores e interpretação:
- R² = 1.0    : Perfeito - modelo explica 100% da variabilidade (raramente ocorre)
- R² >= 0.9   : Excelente - modelo muito preciso
- R² >= 0.7   : Bom - modelo captura a maioria dos padrões
- R² >= 0.5   : Moderado - modelo tem algum poder preditivo
- R² < 0.5    : Fraco - modelo explica menos da metade da variabilidade
- R² ≈ 0      : Ruim - modelo não é melhor que usar a média dos valores
- R² < 0      : Muito ruim - modelo é pior que simplesmente usar a média

Observações:
- Este script pode ser executado diretamente: python src/evaluate.py
- Avalia no dataset completo (não apenas no conjunto de teste)
"""

import joblib
import pandas as pd
import json
from sklearn.metrics import r2_score

print("📊 Iniciando avaliação do modelo...")

# Carrega os dados completos
df = pd.read_csv("data/houses.csv")
X = df[["size", "bedrooms"]]
y = df["price"]

# Carrega o modelo treinado
model = joblib.load("model.joblib")

# Realiza predições no dataset completo
preds = model.predict(X)

# Calcula o R² score
score = r2_score(y, preds)

print(f"✅ Avaliação concluída. R² = {score:.4f}")

# Persiste as métricas
with open("metrics.json", "w") as f:
    json.dump({"r2": score}, f, indent=2)
