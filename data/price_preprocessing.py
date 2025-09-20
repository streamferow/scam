import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data.raw import get_price_df
from sklearn.preprocessing import MinMaxScaler

DATE_COLUMN = "begins_at"

class PricePreprocessor:
    def __init__(self):
        self.df = get_price_df("/Users/ivan/PycharmProjects/scam/BTC.csv")
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

    def to_time_series(self, frequency: str = "D"):
        self.df[DATE_COLUMN] = pd.to_datetime(self.df[DATE_COLUMN])
        time_series = self.df.groupby(pd.Grouper(key=DATE_COLUMN, freq=frequency)).mean()
        time_series = time_series.fillna(0)
        return time_series

    def run_price_preprocessing(self):
        self.fill_missing()
        self.compute_percentage_change()
        self.scale_features()
        self.plot_correlation_matrix()
        self.select_correlated_features()
        return self.to_time_series()

if __name__ == "__main__":
    prep = PricePreprocessor()
    ts = prep.run_price_preprocessing()
    print(ts)
