# CV Bullets — Customer Churn Prediction (Production GCP)

Drop these into the existing churn project on `PRINCE_OKUNADE_CV_DS.docx`. Pick 3 — they're ordered most-to-least keyword-loaded for the Trustpilot JD.

## Replace existing churn bullets with these:

**Customer Churn Prediction | Production ML on GCP** _2026_

- Deployed a tuned XGBoost churn classifier as a containerised **FastAPI** service on **Google Cloud Run**, with prediction logging to **BigQuery** and CI/CD via **GitHub Actions** — converting a research notebook into a reproducible production pipeline that auto-deploys on every push to main.

- Designed a Pydantic-validated REST API exposing per-prediction **SHAP top-5 features** at the response level, enabling any downstream consumer (Streamlit dashboard, retention CRM, batch scoring jobs) to surface explainable reason codes without re-implementing inference logic.

- Authored **BigQuery SQL** queries for prediction volume, risk-band distribution, and training-vs-production churn-rate drift detection — providing the monitoring foundation needed before automated retraining pipelines.

- Tuned the model via 108-configuration `GridSearchCV` with 5-fold stratified cross-validation to a ROC-AUC of 0.81, with thresholds tuned to the business cost of retention (5–7× cheaper than acquisition) so the output is a prioritised retention call list rather than untargeted mass outreach.

## Optional extra (if the section needs a fifth bullet):

- Refactored the inference core into a pure-function module shared by FastAPI, Streamlit, and the pytest suite — single source of truth for prediction logic, eliminating the drift-between-frontends class of production ML bug.

## What this maps to in the Trustpilot JD

| JD requirement | Bullet that hits it |
|---|---|
| "Building and deploying production-ready ML models" | #1 |
| "Cloud technologies — we use GCP" | #1 (Cloud Run, BigQuery, Artifact Registry) |
| "Knowledge of data pipelining and prior experience with cloud-based ML model deployments" | #1, #3 |
| "Python and SQL for data manipulation, modelling, and scripting" | #3 (BigQuery SQL) |
| "Engage with both technical and non-technical stakeholders" | #2 (SHAP explanations are explicitly for non-technical retention staff) |
| "Churn, Upgrade, Upsell" | #4 |

## What's still missing on the CV after this update

The Trustpilot JD also names **attribution, segmentation, pricing optimisation, LTV**. Step 9 of your portfolio is to ship one more project covering those (Project B in the original gap analysis: **Customer Lifetime Value with BG/NBD on Online Retail II**). That's the next thing to build after this one ships.
