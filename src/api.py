"""Customer Churn Prediction — FastAPI service.

Production REST API that serves the same XGBoost + SHAP inference logic
used by the Streamlit app, plus best-effort BigQuery logging.

Endpoints:
    GET  /health    Liveness probe (Cloud Run pings this).
    POST /predict   Single-customer prediction with SHAP top-5 features.

Run locally:
    cd src
    uvicorn api:app --reload --port 8000

Then visit http://localhost:8000/docs for the interactive Swagger UI.
"""
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from bq_logger import BQLogger
from inference import load_artifacts, predict


# ---------------------------------------------------------------------------
# Pydantic schemas — the API contract. Any incoming JSON that does not match
# CustomerProfile is rejected with a 422 before predict() ever runs.
# ---------------------------------------------------------------------------

class CustomerProfile(BaseModel):
    """Single customer's profile, mirroring the training feature columns."""

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal["No", "Yes"]
    Partner: Literal["No", "Yes"]
    Dependents: Literal["No", "Yes"]
    tenure: int = Field(..., ge=0, le=72, description="Months on the network")
    PhoneService: Literal["No", "Yes"]
    MultipleLines: Literal["No", "Yes", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["No", "Yes", "No internet service"]
    OnlineBackup: Literal["No", "Yes", "No internet service"]
    DeviceProtection: Literal["No", "Yes", "No internet service"]
    TechSupport: Literal["No", "Yes", "No internet service"]
    StreamingTV: Literal["No", "Yes", "No internet service"]
    StreamingMovies: Literal["No", "Yes", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["No", "Yes"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(..., ge=0, le=200)
    TotalCharges: float = Field(..., ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female",
                "SeniorCitizen": "Yes",
                "Partner": "No",
                "Dependents": "No",
                "tenure": 2,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 85.0,
                "TotalCharges": 170.0,
            }
        }
    }


class ShapFeature(BaseModel):
    feature: str
    shap_value: float


class PredictionResponse(BaseModel):
    churn_probability: float = Field(..., description="Probability of churn (0-1).")
    churn_prediction: int = Field(..., description="Binary class: 0 = retain, 1 = churn.")
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    shap_top_features: list[ShapFeature] = Field(
        ..., description="Top-5 features by absolute SHAP impact, descending."
    )


class HealthResponse(BaseModel):
    status: Literal["ok"]


# ---------------------------------------------------------------------------
# Lifespan — load artefacts ONCE at startup. Loading per-request would add
# ~2s of latency and defeat the point of a hot service.
# ---------------------------------------------------------------------------

ARTIFACTS: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model, scaler, feature_names, explainer = load_artifacts()
    ARTIFACTS["model"] = model
    ARTIFACTS["scaler"] = scaler
    ARTIFACTS["feature_names"] = feature_names
    ARTIFACTS["explainer"] = explainer
    ARTIFACTS["bq_logger"] = BQLogger()
    yield
    ARTIFACTS.clear()


app = FastAPI(
    title="Customer Churn Prediction API",
    description="XGBoost-powered telecom churn predictions with SHAP explanations.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health():
    """Liveness probe — used by Cloud Run to know the container is up."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(profile: CustomerProfile):
    """Predict churn for a single customer profile."""
    inputs = profile.model_dump()
    result = predict(
        inputs,
        ARTIFACTS["model"],
        ARTIFACTS["scaler"],
        ARTIFACTS["feature_names"],
        ARTIFACTS["explainer"],
    )
    ARTIFACTS["bq_logger"].log_prediction(inputs, result)
    return result
