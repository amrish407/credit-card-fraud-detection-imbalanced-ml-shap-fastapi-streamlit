from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src import config
from src.exception import FraudDetectionException
from src.utils import load_json, load_model


@dataclass
class PredictionResult:
    prediction: int
    fraud_probability: float
    decision_threshold: float
    model_name: str


class FraudPredictor:
    def __init__(
        self,
        model_path: Path = config.BEST_MODEL_FILE,
        metadata_path: Path = config.MODEL_METADATA_FILE,
    ):
        if not model_path.exists():
            raise FraudDetectionException(
                "Trained model artifact not found.",
                context=f"Expected file: {model_path}",
            )

        self.model = load_model(model_path)
        self.metadata = load_json(metadata_path) if metadata_path.exists() else {}
        self.threshold = float(self.metadata.get("decision_threshold", 0.5))
        self.model_name = self.metadata.get("best_model_name", "unknown_model")
        self.expected_features = self.metadata.get("input_features", config.BASE_FEATURES)

    def _coerce_dataframe(self, data: Any) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            dataframe = data.copy()
        elif isinstance(data, dict):
            dataframe = pd.DataFrame([data])
        elif isinstance(data, list):
            dataframe = pd.DataFrame(data)
        else:
            raise FraudDetectionException("Unsupported input format for prediction.")

        missing_features = [feature for feature in self.expected_features if feature not in dataframe.columns]
        if missing_features:
            raise FraudDetectionException(
                "Missing required features for prediction.",
                context=f"Missing: {missing_features}",
            )

        return dataframe[self.expected_features]

    def predict_batch(self, data: Any) -> pd.DataFrame:
        dataframe = self._coerce_dataframe(data)
        probabilities = self.model.predict_proba(dataframe)[:, 1]
        predictions = (probabilities >= self.threshold).astype(int)

        output = dataframe.copy()
        output["fraud_probability"] = probabilities
        output["prediction"] = predictions
        output["decision_threshold"] = self.threshold
        output["model_name"] = self.model_name
        return output

    def predict_single(self, transaction: dict[str, Any]) -> PredictionResult:
        result_df = self.predict_batch(transaction)
        row = result_df.iloc[0]
        return PredictionResult(
            prediction=int(row["prediction"]),
            fraud_probability=float(row["fraud_probability"]),
            decision_threshold=float(row["decision_threshold"]),
            model_name=str(row["model_name"]),
        )
