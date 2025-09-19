import pandas as pd
from data.raw import load_data
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

TEXT_COLUMN = "articles"

class MoodAnalyser:
    def __init__(self, model_name: str = "cfist/multilingual-finbert"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.pipe = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)
        self.df = load_data("/Users/ivan/PycharmProjects/scam/BTC.csv")

    def analyze(self):
        self.df[TEXT_COLUMN] = self.df[TEXT_COLUMN].fillna("")

        moods = []
        for text in self.df[TEXT_COLUMN]:
            if text.strip() == "":
                moods.append(0.5)



