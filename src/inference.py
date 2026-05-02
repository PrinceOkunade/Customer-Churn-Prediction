"""Customer Churn Prediction — inference core.

Pure functions shared by every frontend (Streamlit, FastAPI, tests). No UI
dependencies live here. This is the only place ML logic should exist.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
TOP_N_SHAP = 5


def load_artifacts():
    """Load model, scaler, feature_names, and a TreeExplainer from MODELS_DIR."""
    model = joblib.load(MODELS_DIR / "churn_model.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    feature_names = joblib.load(MODELS_DIR / "feature_names.pkl")
    explainer = shap.TreeExplainer(model)
    return model, scaler, feature_names, explainer


def build_feature_vector(inputs: dict, scaler, feature_names) -> pd.DataFrame:
    """Map raw sidebar/API inputs to the one-hot encoded, scaled training vector."""
    row = {
        "gender": 1 if inputs["gender"] == "Male" else 0,
        "SeniorCitizen": 1 if inputs["SeniorCitizen"] == "Yes" else 0,
        "Partner": 1 if inputs["Partner"] == "Yes" else 0,
        "Dependents": 1 if inputs["Dependents"] == "Yes" else 0,
        "tenure": inputs["tenure"],
        "PhoneService": 1 if inputs["PhoneService"] == "Yes" else 0,
        "PaperlessBilling": 1 if inputs["PaperlessBilling"] == "Yes" else 0,
        "MonthlyCharges": inputs["MonthlyCharges"],
        "TotalCharges": inputs["TotalCharges"],
    }

    one_hot_fields = {
        "MultipleLines": inputs["MultipleLines"],
        "InternetService": inputs["InternetService"],
        "OnlineSecurity": inputs["OnlineSecurity"],
        "OnlineBackup": inputs["OnlineBackup"],
        "DeviceProtection": inputs["DeviceProtection"],
        "TechSupport": inputs["TechSupport"],
        "StreamingTV": inputs["StreamingTV"],
        "StreamingMovies": inputs["StreamingMovies"],
        "Contract": inputs["Contract"],
        "PaymentMethod": inputs["PaymentMethod"],
    }
    for col, val in one_hot_fields.items():
        row[f"{col}_{val}"] = 1

    input_df = pd.DataFrame([row])
    input_df = input_df.reindex(columns=feature_names, fill_value=0)
    input_df[NUM_COLS] = scaler.transform(input_df[NUM_COLS])
    return input_df


def _risk_level(proba: float) -> str:
    if proba < 0.30:
        return "LOW"
    if proba < 0.60:
        return "MEDIUM"
    return "HIGH"


def predict(inputs: dict, model, scaler, feature_names, explainer) -> dict:
    """Run a single-customer prediction and return the API-shaped response dict.

    Returns
    -------
    dict with keys:
        churn_probability  (float, 0-1, rounded to 4dp)
        churn_prediction   (int, 0 or 1)
        risk_level         (str, "LOW" | "MEDIUM" | "HIGH")
        shap_top_features  (list[dict], top-5 features by |SHAP value| desc)
    """
    X_input = build_feature_vector(inputs, scaler, feature_names)
    proba = float(model.predict_proba(X_input)[0, 1])
    pred = int(proba >= 0.5)

    shap_vals = explainer.shap_values(X_input)[0]
    top_idx = np.argsort(np.abs(shap_vals))[-TOP_N_SHAP:][::-1]
    shap_top = [
        {
            "feature": str(feature_names[i]),
            "shap_value": round(float(shap_vals[i]), 4),
        }
        for i in top_idx
    ]

    return {
        "churn_probability": round(proba, 4),
        "churn_prediction": pred,
        "risk_level": _risk_level(proba),
        "shap_top_features": shap_top,
    }


def build_shap_explanation(X_input: pd.DataFrame, explainer) -> shap.Explanation:
    """Build a shap.Explanation for a single-row input — used by waterfall plots."""
    shap_vals = explainer.shap_values(X_input)
    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        base = float(np.array(base).flatten()[0])
    return shap.Explanation(
        values=shap_vals[0],
        base_values=base,
        data=X_input.iloc[0].values,
        feature_names=list(X_input.columns),
    )
