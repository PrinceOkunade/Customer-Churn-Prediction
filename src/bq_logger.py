"""BigQuery logger for prediction requests.

Writes each prediction to `<project>.<dataset>.predictions` via streaming
inserts. Designed to be **fail-soft**: if BigQuery is unreachable, creds
are missing, or logging is disabled, the API still returns a successful
prediction. We never fail a user request because of a logging side effect.

Configuration is via env vars so the same image runs locally (logging off)
and on Cloud Run (logging on):

    BQ_LOGGING_ENABLED   "true" to turn on (default: off)
    GCP_PROJECT_ID       e.g. "churn-production"
    BQ_DATASET           default "churn_data"
    BQ_PREDICTIONS_TABLE default "predictions"
    MODEL_VERSION        any string, written into each row (default "v1")
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class BQLogger:
    def __init__(self) -> None:
        self.enabled = os.getenv("BQ_LOGGING_ENABLED", "false").lower() == "true"
        self.project_id = os.getenv("GCP_PROJECT_ID")
        self.dataset = os.getenv("BQ_DATASET", "churn_data")
        self.table = os.getenv("BQ_PREDICTIONS_TABLE", "predictions")
        self.model_version = os.getenv("MODEL_VERSION", "v1")

        self._client = None
        self._table_ref = None

        if not self.enabled:
            logger.info("BQ logging disabled (set BQ_LOGGING_ENABLED=true to enable).")
            return

        if not self.project_id:
            logger.warning("BQ logging enabled but GCP_PROJECT_ID is unset; disabling.")
            self.enabled = False
            return

        try:
            from google.cloud import bigquery

            self._client = bigquery.Client(project=self.project_id)
            self._table_ref = f"{self.project_id}.{self.dataset}.{self.table}"
            logger.info("BQ logging enabled -> %s", self._table_ref)
        except Exception as exc:
            logger.warning("Failed to init BigQuery client (%s); disabling logging.", exc)
            self.enabled = False

    def log_prediction(self, inputs: dict, result: dict) -> None:
        """Best-effort: log one row. Swallows all exceptions."""
        if not self.enabled or self._client is None:
            return

        row = {
            "request_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "inputs": json.dumps(inputs),
            "churn_probability": float(result["churn_probability"]),
            "churn_prediction": int(result["churn_prediction"]),
            "risk_level": str(result["risk_level"]),
            "model_version": self.model_version,
        }
        try:
            errors = self._client.insert_rows_json(self._table_ref, [row])
            if errors:
                logger.warning("BQ insert returned errors: %s", errors)
        except Exception as exc:
            logger.warning("BQ insert failed: %s", exc)
