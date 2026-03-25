import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """Adds practical engineered features for transaction behaviour."""

    def __init__(self, time_column: str = "Time", amount_column: str = "Amount"):
        self.time_column = time_column
        self.amount_column = amount_column
        self.amount_mean_: float = 1.0

    def fit(self, X: pd.DataFrame, y=None):
        data = pd.DataFrame(X).copy()
        self.amount_mean_ = float(data[self.amount_column].mean()) if self.amount_column in data else 1.0
        if self.amount_mean_ == 0:
            self.amount_mean_ = 1.0
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        data = pd.DataFrame(X).copy()

        if self.time_column in data.columns:
            data["Hour"] = ((data[self.time_column] / 3600.0) % 24).astype(float)
        else:
            data["Hour"] = 0.0

        if self.amount_column in data.columns:
            data["LogAmount"] = np.log1p(data[self.amount_column].clip(lower=0))
            data["AmountToMeanRatio"] = data[self.amount_column] / self.amount_mean_
        else:
            data["LogAmount"] = 0.0
            data["AmountToMeanRatio"] = 0.0

        return data

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None
        return list(input_features) + ["Hour", "LogAmount", "AmountToMeanRatio"]
