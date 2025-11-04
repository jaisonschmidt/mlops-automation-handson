from src.utils import load_data, validate_data
import pandas as pd

def test_load_data():
    df = load_data("data/houses.csv")
    assert isinstance(df, pd.DataFrame)

def test_validate_data():
    df = pd.DataFrame({"price": [100, 200], "size": [50, 70]})
    validate_data(df)
