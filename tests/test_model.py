import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src import config
from src.feature_engineering import FeatureEngineeringTransformer
from src.prediction_pipeline import FraudPredictor


def _train_tiny_model(model_path: Path, metadata_path: Path) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        rng.normal(size=(200, len(config.BASE_FEATURES))),
        columns=config.BASE_FEATURES,
    )
    X["Amount"] = np.abs(X["Amount"] * 150)
    X["Time"] = np.abs(X["Time"] * 1000)

    signal = X["V14"] + X["V10"] - X["V4"]
    y = (signal > signal.quantile(0.92)).astype(int)

    model = ImbPipeline(
        steps=[
            ("feature_engineering", FeatureEngineeringTransformer()),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=500)),
        ]
    )
    model.fit(X, y)

    joblib.dump(model, model_path)
    metadata = {
        "decision_threshold": 0.5,
        "best_model_name": "unit_test_model",
        "input_features": config.BASE_FEATURES,
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    return X.head(3)


def test_prediction_pipeline_output_schema(tmp_path: Path):
    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "metadata.json"

    sample_df = _train_tiny_model(model_path, metadata_path)

    predictor = FraudPredictor(model_path=model_path, metadata_path=metadata_path)
    output_df = predictor.predict_batch(sample_df)

    assert len(output_df) == 3
    assert "fraud_probability" in output_df.columns
    assert "prediction" in output_df.columns
    assert output_df["fraud_probability"].between(0, 1).all()
    assert set(output_df["prediction"].unique()).issubset({0, 1})
