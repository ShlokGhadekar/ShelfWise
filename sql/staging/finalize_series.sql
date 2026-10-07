-- Per-series facts used by features and error analysis.

-- First day on sale = first day of the first week that has a price.
WITH first_wk AS (
    SELECT series_id, MIN(wm_yr_wk) AS wk FROM staging.prices GROUP BY series_id
), wk_start AS (
    SELECT wm_yr_wk, MIN(d_num) AS d FROM staging.calendar GROUP BY wm_yr_wk
)
UPDATE staging.series s
SET first_listed_d = ws.d
FROM first_wk fw
JOIN wk_start ws ON ws.wm_yr_wk = fw.wk
WHERE s.series_id = fw.series_id;

-- First day with a sale. Incremental: only scan the newly loaded days and keep the
-- earlier value if one already exists.
UPDATE staging.series s
SET first_sale_d = x.d
FROM (
    SELECT series_id, MIN(d_num) AS d
    FROM staging.sales
    WHERE sales > 0 AND date BETWEEN DATE '{from_date}' AND DATE '{to_date}'
    GROUP BY series_id
) x
WHERE s.series_id = x.series_id
  AND (s.first_sale_d IS NULL OR x.d < s.first_sale_d);
