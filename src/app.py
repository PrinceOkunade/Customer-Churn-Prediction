"""Customer Churn Prediction — Streamlit App (thin client).

Calls the FastAPI service over HTTP — does not load the model itself.
Default API URL is http://localhost:8000; override with the API_URL env var
to point at a deployed Cloud Run service.

Run:
    # 1. start the API in another terminal
    cd src && uvicorn api:app --reload --port 8000

    # 2. start Streamlit
    streamlit run src/app.py

    # or, against Cloud Run:
    set API_URL=https://churn-api-xxxx.a.run.app
    streamlit run src/app.py
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
TIMEOUT_SECONDS = 10

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="wide")


def call_api(inputs: dict) -> dict:
    """POST a customer profile to the /predict endpoint and return the JSON response."""
    response = requests.post(
        f"{API_URL}/predict",
        json=inputs,
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def render_shap_bars(shap_top: list[dict]) -> None:
    """Horizontal bar chart of top-5 SHAP features (red=increases churn, blue=decreases)."""
    features = [f["feature"] for f in shap_top]
    values = [f["shap_value"] for f in shap_top]
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in values]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.barh(features[::-1], values[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("SHAP value (impact on churn probability)")
    ax.set_title("Top 5 features driving this prediction")
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)


def main():
    st.title("Customer Churn Prediction")
    st.markdown(
        f"Predict telecom customer churn via the deployed **XGBoost + SHAP** API. "
        f"_API endpoint: `{API_URL}`_"
    )

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

    predict_btn = st.sidebar.button("Predict Churn", type="primary", use_container_width=True)

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

    if not predict_btn:
        st.info("Enter a customer profile in the sidebar and click **Predict Churn**.")
        return

    try:
        result = call_api(inputs)
    except requests.exceptions.ConnectionError:
        st.error(
            f"Could not reach the API at `{API_URL}`. "
            "Start the FastAPI service first (`cd src && uvicorn api:app --reload`) "
            "or set the `API_URL` env var to a deployed endpoint."
        )
        return
    except requests.exceptions.HTTPError as exc:
        st.error(f"API returned {exc.response.status_code}: {exc.response.text}")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Churn Prediction", "Yes" if result["churn_prediction"] == 1 else "No")
    c2.metric("Churn Probability", f"{result['churn_probability']*100:.1f}%")
    c3.metric("Risk Level", result["risk_level"])

    st.divider()

    st.subheader("Why this prediction?")
    render_shap_bars(result["shap_top_features"])
    st.markdown(
        "**Red bars** push the prediction toward **churn**; "
        "**blue bars** push it toward **retention**."
    )

    st.divider()

    st.subheader("Customer Profile Summary")
    summary = pd.DataFrame({"Field": list(inputs.keys()), "Value": list(inputs.values())})
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.caption(f"Powered by FastAPI on Cloud Run · XGBoost · SHAP · API: {API_URL}")


if __name__ == "__main__":
    main()
