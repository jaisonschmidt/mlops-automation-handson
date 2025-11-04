import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df

def validate_data(df):
    assert not df.isnull().values.any(), "Dataset contém valores nulos."
    assert "price" in df.columns, "Coluna 'price' ausente."
