-- Daily prediction volume — sanity check that the API is being hit.
SELECT
  DATE(timestamp) AS prediction_date,
  COUNT(*)        AS prediction_count
FROM `churn-production.churn_data.predictions`
WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
GROUP BY prediction_date
ORDER BY prediction_date DESC;
