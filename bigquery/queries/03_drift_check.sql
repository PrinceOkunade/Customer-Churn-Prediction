-- Drift sanity check: predicted churn rate in production vs the training
-- set rate (~27%, hard-coded below). A sustained large gap = retrain signal.
WITH daily AS (
  SELECT
    DATE(timestamp) AS prediction_date,
    AVG(churn_prediction) AS predicted_churn_rate,
    COUNT(*)              AS daily_volume
  FROM `churn-production.churn_data.predictions`
  WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
  GROUP BY prediction_date
)
SELECT
  prediction_date,
  daily_volume,
  ROUND(predicted_churn_rate, 4)        AS predicted_churn_rate,
  0.27                                  AS training_churn_rate,
  ROUND(predicted_churn_rate - 0.27, 4) AS drift_delta
FROM daily
ORDER BY prediction_date DESC;
