import pandas as pd
from typing import TypedDict
from data.raw import get_sentiment_df
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

TEXT_COLUMN = "articles"

class SentimentAnalyser:
    def __init__(self, model_name: str = "ez3nx/ru-tpulse-finance-sentiment"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.pipe = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)
        self.df = get_sentiment_df("/Users/ivan/PycharmProjects/scam/BTC.csv")

    @staticmethod
    def label_to_score(result: TypedDict) -> float:
        label = result["label"]
        score = result["score"]
        if "LABEL_1" in label:
            return 0.5 + 0.5 * score
        elif "LABEL_0" in label:
            return 0.0 + 0.5 * score
        else:
            return 0.4 + 0.2 * score


    def analyze_text(self) -> pd.DataFrame:
        self.df[TEXT_COLUMN] = self.df[TEXT_COLUMN].fillna("")
        sentiment_scores = []

        for texts in self.df[TEXT_COLUMN]:
            scores = []
            for text in texts:
                if text.strip() == "":
                    scores.append(0.5)
                else:
                    result = self.pipe(text)[0]
                    scores.append(SentimentAnalyser.label_to_score(result))
            sentiment_scores.append(sum(scores) / len(scores))
        self.df["sentiment_score"] = sentiment_scores
        return self.df
