import pandas as pd
from tqdm import tqdm
from typing import List
from data.raw import get_sentiment_df
from sklearn.preprocessing import  MinMaxScaler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

TEXT_COLUMNS = "articles"
DATE_COLUMN = "begins_at"

class SentimentPreprocessor:
    def __init__(self, model_name: str = "ez3nx/ru-tpulse-finance-sentiment"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.pipe = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer, return_all_scores=True)
        self.df = get_sentiment_df("/Users/ivan/PycharmProjects/scam/BTC.csv")
        self.scalar = MinMaxScaler()

    @staticmethod
    def probabilities_to_score(probs: List[dict]) -> float:
        negative_prob = next(x["score"] for x in probs if "LABEL_-1" in x["label"])
        positive_prob = next(x["score"] for x in probs if "LABEL_1" in x["label"])
        return positive_prob - negative_prob

    def analyze_text(self) -> pd.DataFrame:
        self.df[TEXT_COLUMNS] = self.df[TEXT_COLUMNS].fillna("")
        sentiment_scores = []
        batch_size = 16

        for texts in tqdm(self.df[TEXT_COLUMNS], desc="News analyzing", total=len(self.df)):
            batch_scores = []

            if isinstance(texts, list):
                for i in range(0, len(texts), batch_size):
                    batch = [t for t in texts[i:i + batch_size]]
                    results = self.pipe(batch)

                    for probs in results:
                        batch_scores.append(SentimentPreprocessor.probabilities_to_score(probs))

            else:
                probs = self.pipe([texts])[0]
                batch_scores.append(SentimentPreprocessor.probabilities_to_score(probs))

            sentiment_scores.append(sum(batch_scores) / len(batch_scores))
        self.df["sentiment_score"] = sentiment_scores
        return self.df

    def scale_features(self):
        self.df[TEXT_COLUMNS] = self.scalar.fit_transform(self.df[TEXT_COLUMNS])
        return self.df

    def to_time_series(self, frequency: str = "D"):
        self.df[DATE_COLUMN] = pd.to_datetime(self.df[DATE_COLUMN])
        time_series = self.df.groupby(pd.Grouper(key=DATE_COLUMN, freq=frequency))["sentiment_score"].mean()
        time_series = time_series.fillna(0)
        return time_series

    def run_sentiment_preprocessing(self):
        self.analyze_text()
        self.scale_features()
        return self.to_time_series()

