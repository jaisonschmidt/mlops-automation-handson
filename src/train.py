from utils import load_data, validate_data
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd, joblib, json

print("🚀 Iniciando treinamento do modelo...")

df = load_data("data/houses.csv")
validate_data(df)

X = df.drop("price", axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

preds = model.predict(X_test)
score = r2_score(y_test, preds)

joblib.dump(model, "model.joblib")
json.dump({"r2": score}, open("metrics.json", "w"))

print(f"✅ Treinamento concluído. R² = {score:.4f}")
