from fastapi.testclient import TestClient

import backend.fastapi_app as api_module
from src.prediction_pipeline import PredictionResult


class FakePredictor:
    model_name = "fake_model"
    threshold = 0.42

    def predict_single(self, transaction):
        return PredictionResult(
            prediction=1,
            fraud_probability=0.91,
            decision_threshold=self.threshold,
            model_name=self.model_name,
        )

    def predict_batch(self, records):
        import pandas as pd

        rows = []
        for _ in records:
            rows.append(
                {
                    "fraud_probability": 0.81,
                    "prediction": 1,
                    "decision_threshold": self.threshold,
                    "model_name": self.model_name,
                }
            )
        return pd.DataFrame(rows)


def _sample_payload() -> dict:
    payload = {"Time": 0.0, "Amount": 100.0}
    payload.update({f"V{i}": 0.0 for i in range(1, 29)})
    return payload


def test_api_predict_endpoints(monkeypatch):
    monkeypatch.setattr(api_module, "get_predictor", lambda: FakePredictor())
    client = TestClient(api_module.app)

    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json()["status"] == "ok"

    predict_resp = client.post("/predict", json=_sample_payload())
    assert predict_resp.status_code == 200
    assert "fraud_probability" in predict_resp.json()

    batch_resp = client.post(
        "/predict_batch",
        json={"records": [_sample_payload(), _sample_payload()]},
    )
    assert batch_resp.status_code == 200
    body = batch_resp.json()
    assert body["total_records"] == 2
    assert len(body["predictions"]) == 2
