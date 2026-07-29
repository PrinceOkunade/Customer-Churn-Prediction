"""Ranking / lift evaluation for the churn model.

Answers the only question a retention manager cares about:
"If I call the top N customers, how many are actually going to churn?"

Reproduces the notebook's exact held-out test split (random_state=42,
stratify=y, test_size=0.2), scores it with the saved model, and reports
precision / capture / lift by risk decile. Also writes a cumulative-gains
chart to images/lift_curve.png.

Run:
    python evaluation/rank_metrics.py
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def load_test_set():
    """Rebuild the exact test fold the model was evaluated on in the notebook."""
    df = pd.read_csv(ROOT / "data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    data = df.drop(columns=["customerID"]).copy()
    data["Churn"] = (data["Churn"] == "Yes").astype(int)
    for c in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        data[c] = (data[c] == "Yes").astype(int)
    data["gender"] = (data["gender"] == "Male").astype(int)
    ohe = ["MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
           "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
           "Contract", "PaymentMethod"]
    data = pd.get_dummies(data, columns=ohe, drop_first=True)
    for c in data.select_dtypes(include="bool").columns:
        data[c] = data[c].astype(int)

    X = data.drop(columns=["Churn"])
    y = data["Churn"]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_test, y_test


def main():
    X_test, y_test = load_test_set()
    model = joblib.load(ROOT / "models/churn_model.pkl")
    scaler = joblib.load(ROOT / "models/scaler.pkl")
    feats = joblib.load(ROOT / "models/feature_names.pkl")

    X_test = X_test.reindex(columns=feats, fill_value=0)
    X_test[NUM_COLS] = scaler.transform(X_test[NUM_COLS])

    proba = model.predict_proba(X_test)[:, 1]
    yt = y_test.values
    n, base = len(yt), y_test.mean()

    print(f"Test set: {n} customers | base churn rate: {base*100:.1f}% "
          f"| ROC-AUC: {roc_auc_score(yt, proba):.4f}\n")

    order = np.argsort(proba)[::-1]
    yt_sorted = yt[order]

    print(f"{'Contacted':>10} {'Customers':>10} {'Precision':>10} "
          f"{'Churn caught':>13} {'Lift':>6}")
    for frac in (0.05, 0.10, 0.20, 0.30, 0.50):
        k = int(round(n * frac))
        top = yt_sorted[:k]
        prec = top.mean()
        recall = top.sum() / yt.sum()
        print(f"{'top '+str(int(frac*100))+'%':>10} {k:>10} "
              f"{prec*100:>9.1f}% {recall*100:>12.1f}% {prec/base:>5.2f}x")

    # Cumulative-gains chart
    cum_caught = np.cumsum(yt_sorted) / yt.sum()
    x = np.arange(1, n + 1) / n
    plt.figure(figsize=(7, 6))
    plt.plot(x * 100, cum_caught * 100, color="#3498db", lw=2, label="Model")
    plt.plot([0, 100], [0, 100], "--", color="grey", label="Random")
    plt.xlabel("% of customer base contacted (highest risk first)")
    plt.ylabel("% of actual churners captured")
    plt.title("Cumulative Gains — churn model vs random outreach")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = ROOT / "images/lift_curve.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"\nSaved cumulative-gains chart -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
