I am building an end-to-end Customer Churn Prediction machine learning project 
for my master's placement portfolio.

IMPORTANT — Before starting, make sure:
1. The dataset WA_Fn-UseC_-Telco-Customer-Churn.csv has been downloaded from
   https://www.kaggle.com/datasets/blastchar/telco-customer-churn
   and placed inside the data/ folder.
2. All dependencies are installed by running: pip install -r requirements.txt

My project folder structure is:
customer-churn-prediction/
├── notebooks/
│   └── churn_prediction.ipynb
├── src/
│   └── app.py
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── models/
├── images/
└── requirements.txt

requirements.txt must contain:
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
xgboost>=1.7.0
imbalanced-learn>=0.10.0
shap>=0.42.0
matplotlib>=3.6.0
seaborn>=0.12.0
plotly>=5.11.0
streamlit>=1.25.0
joblib>=1.2.0

========================================================
STEP 1 — RUN THE NOTEBOOK (churn_prediction.ipynb)
========================================================

Execute every cell in order. Do not skip any step.
Use random_state=42 everywhere for reproducibility.

CELL 1 — IMPORTS:
Load pandas, numpy, matplotlib, seaborn, plotly, warnings.
From sklearn: train_test_split, cross_val_score, GridSearchCV,
StratifiedKFold, StandardScaler, LogisticRegression,
RandomForestClassifier, classification_report, confusion_matrix,
roc_auc_score, roc_curve, precision_recall_curve, f1_score, accuracy_score.
From xgboost: XGBClassifier.
From imblearn.over_sampling: SMOTE.
Import shap and joblib.
Print "All libraries loaded successfully."

CELL 2 — DATA LOADING:
Load ../data/WA_Fn-UseC_-Telco-Customer-Churn.csv.
Print shape, head(5), dtypes, and isnull().sum().
Fix TotalCharges: it contains whitespace — convert with
pd.to_numeric(df['TotalCharges'], errors='coerce') then fill NaN with 0.
Print confirmation of fix.

CELL 3 — EDA (save all plots to ../images/):
Plot 1 — churn_distribution.png:
  Side-by-side bar chart and pie chart of Churn Yes/No with count and
  percentage labels on each bar.
Plot 2 — churn_by_contract.png:
  Grouped bar chart of churn rate by Contract type (Month-to-month,
  One year, Two year). Add percentage labels on each bar.
Plot 3 — tenure_distribution.png:
  Overlapping histogram of tenure (bins=30) for churned vs retained.
Plot 4 — monthly_charges_kde.png:
  KDE density plot of MonthlyCharges for churned vs retained customers.
Plot 5 — churn_by_services.png:
  3x3 subplot grid. For each of these 9 columns — PhoneService,
  MultipleLines, InternetService, OnlineSecurity, OnlineBackup,
  DeviceProtection, TechSupport, StreamingTV, StreamingMovies —
  show a horizontal bar chart of churn rate per category value.
Print one business insight as a print() statement below each plot.

CELL 4 — FEATURE ENGINEERING:
- Drop customerID column
- Encode target: Churn Yes=1, No=0
- Binary encode (Yes=1, No=0): Partner, Dependents, PhoneService, PaperlessBilling
- Binary encode: gender (Male=1, Female=0)
- One-hot encode with drop_first=True: MultipleLines, InternetService,
  OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
  StreamingTV, StreamingMovies, Contract, PaymentMethod
- Print final shape and list of all feature names

CELL 5 — PREPROCESSING:
- Separate X and y. Stratified train_test_split 80/20 random_state=42.
- StandardScaler on tenure, MonthlyCharges, TotalCharges.
  Fit on train only. Transform both train and test.
- Apply SMOTE(random_state=42) to training set ONLY.
  Print class counts before and after SMOTE.

CELL 6 — MODEL TRAINING:
Train all 3 models on SMOTE-balanced training data.
Evaluate on the original (non-SMOTE) test set.
Models:
  - Logistic Regression (max_iter=1000, random_state=42)
  - Random Forest (n_estimators=200, random_state=42, n_jobs=-1)
  - XGBoost (n_estimators=200, max_depth=5, learning_rate=0.1,
             random_state=42, eval_metric='logloss')
For each model print:
  5-fold stratified CV ROC-AUC (mean ± std),
  Test ROC-AUC, F1-Score, Accuracy, full classification_report.
Save to ../images/:
  - model_comparison.png: grouped bar chart of CV AUC, Test AUC,
    F1 Score, Accuracy for all 3 models
  - roc_curves.png: ROC curves for all 3 models on one plot with AUC
    in legend, plus random classifier diagonal

CELL 7 — HYPERPARAMETER TUNING:
GridSearchCV on XGBoost with:
  param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
  }
scoring='roc_auc', cv=StratifiedKFold(n_splits=5).
Print best_params_ and best_score_.
Evaluate best_estimator_ on test set. Print ROC-AUC, F1, Accuracy,
full classification_report.

CELL 8 — EVALUATION:
Save to ../images/evaluation_detailed.png:
  Left plot: confusion matrix heatmap (Blues colormap) with labels
             Retained / Churned on both axes.
  Right plot: Precision-Recall curve with area filled under the curve.

CELL 9 — SHAP EXPLAINABILITY:
Use shap.TreeExplainer(best_model). Compute shap_values on X_test.
Save to ../images/:
  - shap_feature_importance.png: bar summary plot max_display=15
  - shap_beeswarm.png: dot summary plot max_display=15
  - shap_waterfall.png: waterfall plot for the single highest-risk
    customer (y_pred_proba.argmax()), max_display=10.
    Print that customer's churn probability and actual label.

CELL 10 — SAVE ARTIFACTS:
joblib.dump(best_model,          '../models/churn_model.pkl')
joblib.dump(scaler,              '../models/scaler.pkl')
joblib.dump(list(X.columns),     '../models/feature_names.pkl')
Print all 3 saved paths as confirmation.

CELL 11 — NOTEBOOK SUMMARY:
Print a formatted summary block:
  - Best model name
  - Test ROC-AUC score
  - Test F1 score
  - Top 5 most important features (by mean absolute SHAP value)
  - Number of images saved
  - Number of model artifacts saved

========================================================
STEP 2 — BUILD THE STREAMLIT APP (src/app.py)
========================================================

Build a fully working Streamlit app that:

PAGE CONFIG:
  st.set_page_config(page_title="Customer Churn Predictor",
                     page_icon="📊", layout="wide")

LOAD ARTIFACTS:
  Use @st.cache_resource to load churn_model.pkl, scaler.pkl,
  feature_names.pkl from ../models/. Also create shap.TreeExplainer.

HEADER:
  Title: "Customer Churn Prediction"
  Subtitle explaining the app uses XGBoost + SHAP explanations.

SIDEBAR INPUTS — collect all these from the user:
  Demographics: Gender, Senior Citizen, Partner, Dependents
  Account: Tenure (slider 0-72), Monthly Charges (slider 18-120),
           Total Charges (number input defaulting to tenure*monthly)
  Services: Phone Service, Multiple Lines, Internet Service,
            Online Security, Online Backup, Device Protection,
            Tech Support, Streaming TV, Streaming Movies
  Account: Contract, Paperless Billing, Payment Method
  A primary "Predict Churn" button at the bottom of the sidebar.

FEATURE VECTOR:
  Build a function build_feature_vector() that maps all sidebar inputs
  to the exact same one-hot encoded columns used during training.
  Use input_df.reindex(columns=feature_names, fill_value=0) to ensure
  column alignment. Scale tenure, MonthlyCharges, TotalCharges with
  the loaded scaler.

PREDICTION OUTPUT (shown when button is clicked):
  Row of 3 metrics: Churn Prediction (Yes/No), Churn Probability (%),
  Risk Level (LOW <30% / MEDIUM 30-60% / HIGH >60%).
  
  st.divider()
  
  SHAP explanation section titled "Why this prediction?":
  Compute shap_values for the single input row.
  Display shap.waterfall_plot as st.pyplot() with max_display=10.
  Add a markdown explanation: red bars push toward churn,
  blue bars push toward retention.

  st.divider()

  Customer profile summary table showing all input values back to
  the user in a clean st.dataframe().

ERROR HANDLING:
  If model files not found, show st.error() telling user to run
  the notebook first.

FOOTER:
  st.caption("Built with XGBoost · SHAP · Streamlit")

========================================================
STEP 3 — LOCAL DEPLOYMENT & VERIFICATION
========================================================

After building the app, do the following:

1. Verify the models/ folder contains:
   churn_model.pkl, scaler.pkl, feature_names.pkl
   If any are missing, re-run the relevant notebook cells.

2. Launch the Streamlit app:
   Run: streamlit run src/app.py
   Confirm it starts without errors on http://localhost:8501

3. Smoke test the app:
   - Test with a HIGH RISK customer profile:
     gender=Female, SeniorCitizen=Yes, Partner=No, Dependents=No,
     tenure=2, MonthlyCharges=85, Contract=Month-to-month,
     InternetService=Fiber optic, TechSupport=No, OnlineSecurity=No,
     PaymentMethod=Electronic check, PaperlessBilling=Yes
   - Test with a LOW RISK customer profile:
     gender=Male, SeniorCitizen=No, Partner=Yes, Dependents=Yes,
     tenure=60, MonthlyCharges=45, Contract=Two year,
     InternetService=DSL, TechSupport=Yes, OnlineSecurity=Yes,
     PaymentMethod=Bank transfer (automatic), PaperlessBilling=No
   - Confirm predictions differ between profiles.
   - Confirm SHAP waterfall plot renders without errors.

4. Print final confirmation:
   "Deployment verified. App running at http://localhost:8501
    Project ready for GitHub upload."

========================================================
STEP 4 — FINAL FOLDER VERIFICATION
========================================================

Run a final check and print the complete folder tree showing all files:
customer-churn-prediction/
├── notebooks/churn_prediction.ipynb  ✓
├── src/app.py                        ✓
├── data/WA_Fn-UseC_-Telco-...csv    ✓
├── models/churn_model.pkl            ✓
├── models/scaler.pkl                 ✓
├── models/feature_names.pkl          ✓
├── images/  (list all .png files)    ✓
└── requirements.txt                  ✓

If any file is missing, rebuild it before finishing.
Do not stop until every file above is confirmed present.
