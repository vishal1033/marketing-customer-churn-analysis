"""
app/main.py
------------
Stage 6: deployment. Serves the calibrated churn model as a REST API.

Input: RFM summary values for a customer (Recency, Frequency, Monetary,
etc.) — not raw transactions. In a real deployment, these would typically
already be maintained as a customer analytics table (e.g. refreshed
nightly), so this API takes them directly rather than re-aggregating
transaction history per request. This mirrors a realistic integration
point: a CRM or marketing platform computing RFM values and calling this
endpoint to get a churn-risk score.

Run locally:
    uvicorn app.main:app --reload --port 8001
Then open http://localhost:8001/docs
(Using port 8001 here so it doesn't collide with Project 1's app, if
both are ever run side by side.)
"""

from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_DIR = Path(__file__).resolve().parent / "model"

FEATURES = [
    "Recency", "Frequency", "Monetary", "TenureDays",
    "AvgOrderValue", "UniqueProducts", "CancellationRate",
]

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts probability that a customer will NOT make a "
                "repeat purchase in the next 90 days, based on RFM "
                "behavior. Trained on real UK e-commerce transaction "
                "data (UCI Online Retail).",
    version="1.0.0",
)

_artifacts = {}


@app.on_event("startup")
def load_artifacts():
    try:
        _artifacts["model"] = joblib.load(MODEL_DIR / "model.joblib")
        _artifacts["feature_names"] = joblib.load(MODEL_DIR / "feature_names.joblib")
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Missing model artifact: {e}. Run src/train.py first."
        )


class CustomerRFM(BaseModel):
    recency: int = Field(..., ge=0, description="Days since last purchase (as of the analysis cutoff)")
    frequency: int = Field(..., ge=1, description="Number of historical orders")
    monetary: float = Field(..., ge=0, description="Total historical spend (GBP)")
    tenure_days: int = Field(..., ge=0, description="Days between first and last purchase")
    avg_order_value: float = Field(..., ge=0, description="Average spend per order (GBP)")
    unique_products: int = Field(..., ge=1, description="Number of distinct products purchased")
    cancellation_rate: float = Field(..., ge=0, le=1, description="Fraction of transactions that were cancellations")

    class Config:
        json_schema_extra = {
            "example": {
                "recency": 45,
                "frequency": 4,
                "monetary": 850.0,
                "tenure_days": 180,
                "avg_order_value": 212.5,
                "unique_products": 30,
                "cancellation_rate": 0.02,
            }
        }


class PredictionResponse(BaseModel):
    churn_probability: float
    risk_tier: str


def risk_tier(probability: float) -> str:
    if probability < 0.25:
        return "Low"
    elif probability < 0.50:
        return "Moderate"
    elif probability < 0.75:
        return "Elevated"
    return "High"


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in _artifacts}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerRFM):
    if not _artifacts:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded.")

    X = np.array([[
        customer.recency,
        customer.frequency,
        customer.monetary,
        customer.tenure_days,
        customer.avg_order_value,
        customer.unique_products,
        customer.cancellation_rate,
    ]])

    proba = _artifacts["model"].predict_proba(X)[0, 1]

    return PredictionResponse(
        churn_probability=round(float(proba), 4),
        risk_tier=risk_tier(proba),
    )
