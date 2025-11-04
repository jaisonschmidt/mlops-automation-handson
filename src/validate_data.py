import pandas as pd
import sys

try:
    print("🔍 Iniciando validação de dados...")

    df = pd.read_csv("data/houses.csv")

    required_columns = ["size", "bedrooms", "price"]
    for col in required_columns:
        assert col in df.columns, f"❌ Coluna obrigatória ausente: {col}"

    assert not df.isnull().values.any(), "❌ Existem valores nulos no dataset"
    assert (df["size"] > 0).all(), "❌ Há valores inválidos em 'size'"
    assert (df["bedrooms"] > 0).all(), "❌ Há valores inválidos em 'bedrooms'"
    assert (df["price"] > 0).all(), "❌ Há valores inválidos em 'price'"

    print("✅ Validação de dados concluída com sucesso!")

except AssertionError as e:
    print(str(e))
    sys.exit(1)
except Exception as e:
    print(f"⚠️ Erro inesperado: {e}")
    sys.exit(1)
