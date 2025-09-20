import pandas as pd

DATE_COL = ["begins_at"]
PRICE_COLS = ["open_price", "close_price", "high_price", "low_price"] + DATE_COL
SENTIMENT_COLS = ["articles"] + DATE_COL

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def get_price_df(path: str) -> pd.DataFrame:
    df = load_data(path)
    price_df = df[PRICE_COLS].copy()
    return price_df

def get_sentiment_df(path: str) -> pd.DataFrame:
    df = load_data(path)
    sentiment_df = df[SENTIMENT_COLS].copy()
    return sentiment_df
