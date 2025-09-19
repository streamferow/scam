import pandas as pd

PRICE_COLS = ["open_price", "close_price", "high_price", "low_price"]
SENTIMENT_COLS = ["articles"]

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def get_price_df(df: pd.DataFrame) -> pd.DataFrame:
    price_df = df[PRICE_COLS].copy()
    return price_df

def get_sentiment_df(df: pd.DataFrame) -> pd.DataFrame:
    sentiment_df = df[SENTIMENT_COLS]
    return sentiment_df