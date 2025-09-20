import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data.raw import get_price_df, get_sentiment_df
from sklearn.preprocessing import MinMaxScaler
from data.sentiment import SentimentAnalyser


class DataPreprocessor:
    def __init__(self):
        self.sentiment_df = SentimentAnalyser().analyze_text()
        self.price_df = get_price_df("/Users/ivan/PycharmProjects/scam/BTC.csv")
        self.df = pd.merge(self.sentiment_df, self.price_df, on="begins_at", how="left")

        self.scalar = MinMaxScaler()
        self.target_col = "close_price"
        self.numeric_cols = self.df.select_dtypes(include="number").columns


    def fill_missing(self, interpolation: str = "linear") -> pd.DataFrame:
        self.df[self.numeric_cols] = self.df[self.numeric_cols].interpolate(method=interpolation).ffill().bfill()
        return self.df

    def compute_percentage_change(self) -> pd.DataFrame:
        for col in self.numeric_cols:
            self.df[f"{col}_pct_change"] = self.df[col].pct_change() * 100

        pct_cols = [f"{col}_pct_change" for col in self.numeric_cols]
        self.df[pct_cols] = self.df[pct_cols].fillna(0)
        return self.df

    def scale_features(self) -> pd.DataFrame:
        self.df[self.numeric_cols] = self.scalar.fit_transform(self.df[self.numeric_cols])
        return self.df

    def plot_correlation_matrix(self, method: str = "pearson") -> None:
        corr = self.df[self.numeric_cols].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f")
        plt.title(f"Correlation Matrix ({method.capitalize()})")
        plt.show()

    def select_correlated_features(self, correlation_threshold: float = 0.5) -> pd.DataFrame:
        corr = self.df[self.numeric_cols].corr()
        target_corr = corr[self.target_col].abs()
        selected_features = target_corr[target_corr > correlation_threshold].index.tolist()
        return self.df[selected_features]


if __name__ == "__main__":
    prep = DataPreprocessor()
    prep.fill_missing()
    prep.compute_percentage_change()
    prep.scale_features()
    prep.plot_correlation_matrix()
    df_selected = prep.select_correlated_features()
