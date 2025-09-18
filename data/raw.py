import pandas as pd

MEDIA_PATH = ""
SERIES_PATH = ""

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

