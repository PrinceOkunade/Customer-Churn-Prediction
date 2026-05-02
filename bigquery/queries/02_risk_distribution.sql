-- Risk-level distribution over the last 7 days.
-- Compare these proportions to the training-set churn rate (~27%) to
-- catch upstream changes — e.g. a frontend bug skewing inputs HIGH.
SELECT
  risk_level,
  COUNT(*)                            AS request_count,
  ROUND(AVG(churn_probability), 4)    AS avg_probability,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct_of_total
FROM `churn-production.churn_data.predictions`
WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
GROUP BY risk_level
ORDER BY
  CASE risk_level WHEN 'HIGH' THEN 1 WHEN 'MEDIUM' THEN 2 ELSE 3 END;
