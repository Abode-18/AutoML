import pandas as pd

def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)