import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone


COIN_ID = "the-open-network"
VS_CURRENCY = "usd"
OUT_CSV = "ton_hourly.csv"
OUT_PARQUET = "ton_hourly.parquet"

CHUNK_DAYS = 90
CHUNK_DELTA = timedelta(days=CHUNK_DAYS)

BASE_URL = "https://api.coingecko.com/api/v3/coins/{id}/market_chart/range"
API_KEY = 'CG-SE8XA9n9YebxBYx5gR4X7eh9'
HEADERS = {
    "x-cg-demo-api-key": API_KEY,
    "accept": "application/json"
}

SLEEP_BETWEEN_CHUNKS = 1.2
REQUEST_TIMEOUT = 30
MAX_RETRIES = 6







