# Runbook — manual steps you need to run

Everything else is code. This file lists the actions only you can do (gcloud auth, GitHub secrets, Docker push). Run them in order. Estimated time: ~45 min the first time.

Fill in the values in the box first; every command below uses them.

```
PROJECT_ID  = churn-production
REGION      = europe-west2
SERVICE     = churn-api
REPOSITORY  = churn
IMAGE       = api
DATASET     = churn_data
TABLE       = predictions
GITHUB_USER = <your github username>
GITHUB_REPO = customer-churn-prediction
```

---

## 1. Verify the API runs locally (5 min)

```cmd
pip install -r requirements.txt
cd C:\Users\Prince\Desktop\Customer-Churn-Prediction
pytest tests/ -v
```

All five tests should pass. Then:

```cmd
cd src
uvicorn api:app --reload --port 8000
```

Hit http://localhost:8000/docs, fire one prediction, confirm `risk_level` matches the existing Streamlit behaviour.

---

## 2. Build and run the Docker image locally (10 min)

```cmd
cd C:\Users\Prince\Desktop\Customer-Churn-Prediction
docker build -t churn-api .
docker run -p 8080:8080 -e PORT=8080 churn-api
```

Hit http://localhost:8080/health → `{"status":"ok"}`. Stop with Ctrl+C.

If `docker` is not installed, get Docker Desktop for Windows: https://www.docker.com/products/docker-desktop/

---

## 3. Create the BigQuery dataset and predictions table (5 min)

```bash
# Create dataset
bq --location=europe-west2 mk --dataset churn-production:churn_data

# Create the predictions table from the schema file
bq mk --table \
  --schema=bigquery/predictions_schema.json \
  --time_partitioning_field=timestamp \
  churn-production:churn_data.predictions
```

Verify in the GCP console → BigQuery → `churn-production` → `churn_data` → `predictions`.

(Optional) Load the training CSV as `telco_raw`:
```bash
bq load --source_format=CSV --autodetect \
  churn-production:churn_data.telco_raw \
  data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

---

## 4. Create the Artifact Registry repository (2 min)

```bash
gcloud artifacts repositories create churn \
  --repository-format=docker \
  --location=europe-west2 \
  --description="Churn API container images"
```

---

## 5. Create a service account for CI/CD (5 min)

Windows cmd — paste each command one at a time:

```cmd
gcloud iam service-accounts create churn-deployer --display-name="Churn API CI/CD"

gcloud projects add-iam-policy-binding churn-production --member="serviceAccount:churn-deployer@churn-production.iam.gserviceaccount.com" --role="roles/run.admin"

gcloud projects add-iam-policy-binding churn-production --member="serviceAccount:churn-deployer@churn-production.iam.gserviceaccount.com" --role="roles/artifactregistry.writer"

gcloud projects add-iam-policy-binding churn-production --member="serviceAccount:churn-deployer@churn-production.iam.gserviceaccount.com" --role="roles/iam.serviceAccountUser"

gcloud projects add-iam-policy-binding churn-production --member="serviceAccount:churn-deployer@churn-production.iam.gserviceaccount.com" --role="roles/bigquery.dataEditor"

gcloud iam service-accounts keys create churn-deployer-key.json --iam-account=churn-deployer@churn-production.iam.gserviceaccount.com
```

The key file `churn-deployer-key.json` lands in your current directory. Open it in Notepad, copy the entire JSON content. **Then delete the file from your laptop** — it's a credential.

---

## 6. Add GitHub secrets (3 min)

In your GitHub repo → Settings → Secrets and variables → Actions → New repository secret:

| Name              | Value |
|-------------------|-------|
| `GCP_PROJECT_ID`  | `churn-production` |
| `GCP_SA_KEY`      | the entire JSON content from step 5 |

---

## 7. First manual deploy (10 min)

CI/CD will handle every future deploy, but you need one manual deploy first so the service exists:

```bash
# Authenticate Docker to Artifact Registry
gcloud auth configure-docker europe-west2-docker.pkg.dev --quiet

# Build and tag
docker build -t europe-west2-docker.pkg.dev/churn-production/churn/api:v1 .

# Push
docker push europe-west2-docker.pkg.dev/churn-production/churn/api:v1

# Deploy
gcloud run deploy churn-api \
  --image=europe-west2-docker.pkg.dev/churn-production/churn/api:v1 \
  --region=europe-west2 \
  --allow-unauthenticated \
  --memory=1Gi --cpu=1 \
  --min-instances=0 --max-instances=3 \
  --set-env-vars="GCP_PROJECT_ID=churn-production,BQ_LOGGING_ENABLED=true,MODEL_VERSION=v1"
```

The output prints a URL like `https://churn-api-xxxxxxxxxx-ew.a.run.app`. **Save that URL.**

Smoke test:
```bash
curl https://<URL>/health
curl -X POST https://<URL>/predict \
  -H "Content-Type: application/json" \
  -d @- <<'JSON'
{"gender":"Female","SeniorCitizen":"Yes","Partner":"No","Dependents":"No","tenure":2,"PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No","StreamingTV":"No","StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":85.0,"TotalCharges":170.0}
JSON
```

Then in the BigQuery console, run:
```sql
SELECT * FROM `churn-production.churn_data.predictions` ORDER BY timestamp DESC LIMIT 5;
```

You should see your test row(s).

---

## 8. Push to GitHub and let CI/CD take over (5 min)

If you haven't already:
```bash
cd C:\Users\Prince\Desktop\Customer-Churn-Prediction
git add .
git commit -m "Add FastAPI service, Dockerfile, BigQuery logging, CI/CD"
git push origin main
```

Watch the run at https://github.com/<user>/<repo>/actions. From now on, every push to `main` rebuilds and redeploys.

---

## 9. Wire Streamlit to the deployed API (1 min)

```cmd
set API_URL=https://churn-api-xxxxxxxxxx-ew.a.run.app
streamlit run src/app.py
```

Predict via the UI → confirm a new row appears in the BigQuery `predictions` table.

---

## 10. Update your CV

Open `CV_BULLETS.md`, copy the bullets into `PRINCE_OKUNADE_CV_DS.docx` replacing the existing churn entries.

---

## Cost monitoring

Check the Billing dashboard weekly at first. With `min-instances=0` the service scales to zero when idle, so the steady-state cost is near nothing. If you're seeing more than $5/month, something is misconfigured — likely `min-instances` is not 0, or the BigQuery streaming insert volume is high. Check the Cloud Run service config first.

---

## If something breaks

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: inference` in Cloud Run logs | Wrong `WORKDIR` in Dockerfile | Confirm Dockerfile has `WORKDIR /app/src` before CMD |
| 422 on every prediction | Pydantic field name mismatch | Field names in CustomerProfile must match exactly: `gender`, `SeniorCitizen`, etc. |
| BigQuery rows not appearing | Service account missing `bigquery.dataEditor` role | Re-run the IAM binding from step 5 |
| GitHub Actions fails on `docker push` | `gcloud auth configure-docker` missed | Already in the workflow — check the secret `GCP_SA_KEY` is the full JSON, not the file path |
| Cloud Run cold start > 10s | shap is heavy, image is fat | Set `--cpu=2` temporarily, or move SHAP behind a lazy-load |
