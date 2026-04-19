# Customer Churn Prediction

End-to-end ML pipeline predicting telecom customer churn with **XGBoost + SHAP explanations**, wrapped in a **Streamlit** web app.
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://customer-churn-prediction-pta8ejjyrq8ahuxaeuwzzb.streamlit.app/)

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![ML](https://img.shields.io/badge/ML-XGBoost-green)

## Problem

Customer churn costs subscription businesses far more than acquisition — retaining an existing customer is 5–7x cheaper than acquiring a new one. This project predicts, per customer, the probability of churn and explains each prediction with SHAP so that retention teams can take targeted action.

## Dataset

**IBM Telco Customer Churn** — 7,043 customers, 21 features.

Source: [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Pipeline

1. **Data Cleaning** — fixed `TotalCharges` whitespace values; converted to numeric
2. **EDA** — 5 visualisations uncovering churn drivers (contract type, tenure, charges, services)
3. **Feature Engineering** — binary + one-hot encoding (`drop_first=True`), 30 final features
4. **Preprocessing** — stratified 80/20 split, `StandardScaler` (fit on train only), SMOTE on training set only
5. **Model Comparison** — Logistic Regression, Random Forest, XGBoost (5-fold stratified CV)
6. **Hyperparameter Tuning** — `GridSearchCV` over 108 XGBoost configurations
7. **Explainability** — SHAP feature importance, beeswarm, and per-customer waterfall plots
8. **Deployment** — interactive Streamlit app with live SHAP explanations

## Results

| Metric | Tuned XGBoost |
|--------|---------------|
| ROC-AUC | 0.8080 |
| F1-Score | 0.5914 |
| Accuracy | 0.7637 |

### Key EDA Insights

- **27%** of customers churned — class imbalance addressed with SMOTE
- **Month-to-month** contracts churn at ~43% vs under 3% for two-year contracts
- Customers **without** TechSupport, OnlineSecurity, or OnlineBackup churn at significantly higher rates
- Most churners leave within the **first few months** of tenure
- Higher **MonthlyCharges** correlate with higher churn, especially on fibre-optic internet

### Sample Visualisations

| Churn Distribution | ROC Curves | SHAP Feature Importance |
|---|---|---|
| ![](images/churn_distribution.png) | ![](images/roc_curves.png) | ![](images/shap_feature_importance.png) |

## Folder Structure

```
customer-churn-prediction/
├── notebooks/
│   └── churn_prediction.ipynb      # Full ML pipeline (11 cells)
├── src/
│   └── app.py                      # Streamlit web app
├── data/
│   └── WA_Fn-UseC_-Telco-...csv   # Telco dataset
├── models/
│   ├── churn_model.pkl             # Tuned XGBoost model
│   ├── scaler.pkl                  # Fitted StandardScaler
│   └── feature_names.pkl           # Column order for inference
├── images/                         # 11 EDA + evaluation PNGs
├── requirements.txt
└── README.md
```

## Run Locally

```bash
# Clone the repo
git clone https://github.com/<your-username>/customer-churn-prediction.git
cd customer-churn-prediction

# Install dependencies
pip install -r requirements.txt

# Launch the app
python -m streamlit run src/app.py
```

The app opens at [http://localhost:8501](http://localhost:8501).

## App Features

- **Sidebar inputs** — enter any customer profile (demographics, account, services)
- **Churn probability** — model outputs a percentage and a risk band (LOW / MEDIUM / HIGH)
- **SHAP waterfall plot** — explains exactly why the model made that prediction
- **Profile summary** — review all entered values at a glance

## Known Limitations

- Probabilities are not calibrated (Platt scaling or isotonic regression would improve this)
- No temporal validation — production deployment should use a time-based split
- Uses gender and senior-citizen features; a fairness audit is recommended before real-world use
- Static model — production would require monitoring for concept drift

## Tech Stack

Python · pandas · NumPy · scikit-learn · XGBoost · imbalanced-learn · SHAP · Matplotlib · Seaborn · Plotly · Streamlit · joblib

## Author

Prince Okunade
