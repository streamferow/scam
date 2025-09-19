import pandas as pd
from data.raw import load_data
from sklearn.preprocessing import minmax_scale


class DataPreprocessor:
    def __init__(self):
        self.df = load_data("/Users/ivan/PycharmProjects/scam/BTC.csv")
        self.scalar = minmax_scale()

