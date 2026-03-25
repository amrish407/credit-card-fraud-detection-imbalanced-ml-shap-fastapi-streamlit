from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.exception import FraudDetectionException
from src.prediction_pipeline import FraudPredictor

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="FastAPI service for fraud probability prediction on imbalanced credit card transactions.",
    version="1.0.0",
)

_predictor_cache: FraudPredictor | None = None


class TransactionRecord(BaseModel):
    Time: float = Field(..., description="Seconds elapsed between this transaction and first transaction")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(..., ge=0, description="Transaction amount")

    model_config = {
        "json_schema_extra": {
            "example": {
                "Time": 0.0,
                "V1": -1.3598,
                "V2": -0.0728,
                "V3": 2.5363,
                "V4": 1.3782,
                "V5": -0.3383,
                "V6": 0.4624,
                "V7": 0.2396,
                "V8": 0.0987,
                "V9": 0.3638,
                "V10": 0.0908,
                "V11": -0.5516,
                "V12": -0.6178,
                "V13": -0.9914,
                "V14": -0.3112,
                "V15": 1.4682,
                "V16": -0.4704,
                "V17": 0.2080,
                "V18": 0.0258,
                "V19": 0.4040,
                "V20": 0.2514,
                "V21": -0.0183,
                "V22": 0.2778,
                "V23": -0.1105,
                "V24": 0.0669,
                "V25": 0.1285,
                "V26": -0.1891,
                "V27": 0.1336,
                "V28": -0.0211,
                "Amount": 149.62,
            }
        }
    }


class BatchRequest(BaseModel):
    records: List[TransactionRecord]


def get_predictor() -> FraudPredictor:
    global _predictor_cache
    if _predictor_cache is None:
        try:
            _predictor_cache = FraudPredictor()
        except FraudDetectionException as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
    return _predictor_cache


@app.get("/")
def root():
    return {
        "message": "Credit Card Fraud Detection API",
        "endpoints": ["/", "/health", "/predict", "/predict_batch"],
    }


@app.get("/health")
def health():
    try:
        predictor = get_predictor()
        return {"status": "ok", "model": predictor.model_name, "threshold": predictor.threshold}
    except HTTPException as error:
        return {"status": "degraded", "detail": error.detail}


@app.post("/predict")
def predict(record: TransactionRecord):
    predictor = get_predictor()
    result = predictor.predict_single(record.model_dump())
    return {
        "prediction": result.prediction,
        "fraud_probability": result.fraud_probability,
        "decision_threshold": result.decision_threshold,
        "model_name": result.model_name,
    }


@app.post("/predict_batch")
def predict_batch(request: BatchRequest):
    predictor = get_predictor()
    batch_df = predictor.predict_batch([record.model_dump() for record in request.records])

    return {
        "total_records": len(batch_df),
        "predictions": batch_df[
            ["fraud_probability", "prediction", "decision_threshold", "model_name"]
        ].to_dict(orient="records"),
    }
