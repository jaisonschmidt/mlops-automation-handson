import joblib, pandas as pd, json
from sklearn.metrics import r2_score

print("📊 Iniciando avaliação do modelo...")

df = pd.read_csv("data/houses.csv")
X = df[["size", "bedrooms"]]
y = df["price"]

model = joblib.load("model.joblib")
preds = model.predict(X)

score = r2_score(y, preds)

print(f"✅ Avaliação concluída. R² = {score:.4f}")

with open("metrics.json", "w") as f:
    json.dump({"r2": score}, f)
