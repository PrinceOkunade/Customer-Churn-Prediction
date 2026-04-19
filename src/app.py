"""Customer Churn Prediction — Streamlit App."""
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="wide")

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODELS_DIR / "churn_model.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    feature_names = joblib.load(MODELS_DIR / "feature_names.pkl")
    explainer = shap.TreeExplainer(model)
    return model, scaler, feature_names, explainer


def build_feature_vector(inputs: dict, scaler, feature_names):
    """Map sidebar inputs -> one-hot encoded, scaled feature vector aligned to training columns."""
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

    # One-hot (drop_first=True) — dummy columns named <col>_<value>
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


def main():
    st.title("Customer Churn Prediction")
    st.markdown(
        "Predict telecom customer churn using a tuned **XGBoost** model, "
        "with **SHAP** explanations for every prediction."
    )

    try:
        model, scaler, feature_names, explainer = load_artifacts()
    except FileNotFoundError:
        st.error(
            "Model files not found. Please run `notebooks/churn_prediction.ipynb` "
            "first to generate `churn_model.pkl`, `scaler.pkl`, and `feature_names.pkl`."
        )
        return

    # ---------- Sidebar ----------
    st.sidebar.header("Customer Profile")

    st.sidebar.subheader("Demographics")
    gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    senior = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.sidebar.selectbox("Partner", ["No", "Yes"])
    dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])

    st.sidebar.subheader("Account")
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    monthly = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 70.0, step=0.5)
    total = st.sidebar.number_input(
        "Total Charges ($)",
        min_value=0.0, value=float(tenure * monthly), step=10.0
    )

    st.sidebar.subheader("Services")
    phone = st.sidebar.selectbox("Phone Service", ["No", "Yes"])
    multiple = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_sec = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_bkp = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_prot = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_supp = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    stream_tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    stream_mv = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    st.sidebar.subheader("Contract & Billing")
    contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
    payment = st.sidebar.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )

    predict = st.sidebar.button("Predict Churn", type="primary", use_container_width=True)

    inputs = {
        "gender": gender, "SeniorCitizen": senior, "Partner": partner,
        "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
        "MultipleLines": multiple, "InternetService": internet,
        "OnlineSecurity": online_sec, "OnlineBackup": online_bkp,
        "DeviceProtection": device_prot, "TechSupport": tech_supp,
        "StreamingTV": stream_tv, "StreamingMovies": stream_mv,
        "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": total,
    }

    if not predict:
        st.info("Enter a customer profile in the sidebar and click **Predict Churn**.")
        return

    X_input = build_feature_vector(inputs, scaler, feature_names)
    proba = float(model.predict_proba(X_input)[0, 1])
    pred = int(proba >= 0.5)

    if proba < 0.30:
        risk = "LOW"
    elif proba < 0.60:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    c1, c2, c3 = st.columns(3)
    c1.metric("Churn Prediction", "Yes" if pred == 1 else "No")
    c2.metric("Churn Probability", f"{proba*100:.1f}%")
    c3.metric("Risk Level", risk)

    st.divider()

    st.subheader("Why this prediction?")
    shap_vals = explainer.shap_values(X_input)
    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        base = float(np.array(base).flatten()[0])

    explanation = shap.Explanation(
        values=shap_vals[0],
        base_values=base,
        data=X_input.iloc[0].values,
        feature_names=list(X_input.columns),
    )

    fig = plt.figure()
    shap.plots.waterfall(explanation, max_display=10, show=False)
    st.pyplot(fig, clear_figure=True)

    st.markdown(
        "**Red bars** push the prediction toward **churn**; "
        "**blue bars** push it toward **retention**."
    )

    st.divider()

    st.subheader("Customer Profile Summary")
    summary = pd.DataFrame({"Field": list(inputs.keys()), "Value": list(inputs.values())})
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.caption("Built with XGBoost · SHAP · Streamlit")


if __name__ == "__main__":
    main()
