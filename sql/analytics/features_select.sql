-- =====================================================================================
-- FEATURE ENGINEERING (one row per series per target day)
-- =====================================================================================
-- This SELECT runs on Postgres in the pipeline AND on DuckDB in the unit tests, so it
-- sticks to portable SQL (CAST(...) instead of ::, no Postgres-only functions).
--
-- THE LEAKAGE RULE
-- We forecast 28 days ahead. Standing at the forecast origin T, the latest sales we
-- know are for day T. For a target day t = T + h (h = 1..28) a feature may use sales
-- up to day T at most - and because h can be as large as 28, every sales-based
-- feature for day t must end at day t - 28 or earlier. That is why:
--   * the shortest lag is lag_28 (never lag_1 / lag_7)
--   * every rolling window ends "28 PRECEDING", not at the current row
-- With this rule ONE global model can serve all 28 horizons without feeding its own
-- predictions back in (no recursive forecasting, no error compounding).
--
-- Beginner mistake this avoids: computing rolling_mean_7 over the last 7 days
-- including yesterday, training on it (great validation score!) and then discovering
-- at forecast time that "yesterday" for a day 20 days out does not exist yet.
--
-- Things that ARE allowed to look at day t itself or beyond:
--   * calendar fields and events - published in advance
--   * prices - the retailer sets its own prices ahead of time (assumption; M5 also
--     provides prices for the forecast period)
--
-- OTHER DESIGN NOTES
--   * Before an item is on sale (no price) its sales are 0 in the file. Those zeros
--     are not "zero demand", so we turn them into NULL; AVG/STDDEV skip NULLs and
--     LightGBM treats NULL as missing.
--   * ROWS frames count rows, not days. They equal "N days" only if every series
--     has exactly one row per day - the data-quality step checks this (dense series).
--   * Window aggregates are computed on the INTEGER sales column on purpose: Postgres
--     has "inverse transition" functions for integer AVG/STDDEV, so a sliding window
--     costs O(1) per row. For float columns it re-aggregates the whole frame per row.
--   * Windows run on observed + future rows in one pass. Future rows (sales unknown)
--     still get valid lags because lag_28 of the last future day is the last observed day.
--
-- PARAMETERS (rendered by shelfwise.sqlrender - ints and dates only)
--   {lo_day}/{lo_date}               first day read (target day - look-back buffer)
--   {obs_end_day}/{obs_end_date}     last observed day ("today")
--   {future_end_day}                 last future day to create rows for (today + horizon)
--   {from_day}                       first target day to output
--   {sid_lo}..{sid_hi}               series_id batch
--   {dsls_cap}                       cap for days-since-last-sale (keeps incremental == full)
-- =====================================================================================
WITH spine AS (
    -- Observed days.
    SELECT s.series_id,
           s.d_num,
           CAST(s.sell_price AS DOUBLE PRECISION) AS sell_price,
           CASE WHEN s.sell_price IS NULL THEN NULL ELSE CAST(s.sales AS INTEGER) END AS sales
    FROM staging.sales s
    WHERE s.date BETWEEN DATE '{lo_date}' AND DATE '{obs_end_date}'   -- partition pruning
      AND s.d_num BETWEEN {lo_day} AND {obs_end_day}
      AND s.series_id BETWEEN {sid_lo} AND {sid_hi}
    UNION ALL
    -- Future days we need to forecast: sales unknown (NULL), price known in advance.
    SELECT sr.series_id,
           c.d_num,
           CAST(p.sell_price AS DOUBLE PRECISION) AS sell_price,
           CAST(NULL AS INTEGER) AS sales
    FROM staging.series sr
    CROSS JOIN staging.calendar c
    LEFT JOIN staging.prices p ON p.series_id = sr.series_id AND p.wm_yr_wk = c.wm_yr_wk
    WHERE c.d_num BETWEEN {obs_end_day} + 1 AND {future_end_day}
      AND sr.series_id BETWEEN {sid_lo} AND {sid_hi}
),
windowed AS (
    SELECT
        series_id, d_num, sell_price, sales,
        -- Same weekday 4, 5, 6, 7, 8 weeks back (weekly seasonality is the strongest pattern in retail).
        LAG(sales, 28) OVER (PARTITION BY series_id ORDER BY d_num) AS lag_28,
        LAG(sales, 35) OVER (PARTITION BY series_id ORDER BY d_num) AS lag_35,
        LAG(sales, 42) OVER (PARTITION BY series_id ORDER BY d_num) AS lag_42,
        LAG(sales, 49) OVER (PARTITION BY series_id ORDER BY d_num) AS lag_49,
        LAG(sales, 56) OVER (PARTITION BY series_id ORDER BY d_num) AS lag_56,
        -- Rolling level / volatility, all windows END 28 days before the target day.
        AVG(sales) OVER (PARTITION BY series_id ORDER BY d_num ROWS BETWEEN 34 PRECEDING AND 28 PRECEDING) AS rmean_7,
        AVG(sales) OVER (PARTITION BY series_id ORDER BY d_num ROWS BETWEEN 41 PRECEDING AND 28 PRECEDING) AS rmean_14,
        AVG(sales) OVER (PARTITION BY series_id ORDER BY d_num ROWS BETWEEN 55 PRECEDING AND 28 PRECEDING) AS rmean_28,
        AVG(sales) OVER (PARTITION BY series_id ORDER BY d_num ROWS BETWEEN 83 PRECEDING AND 28 PRECEDING) AS rmean_56,
        STDDEV_SAMP(sales) OVER (PARTITION BY series_id ORDER BY d_num ROWS BETWEEN 34 PRECEDING AND 28 PRECEDING) AS rstd_7,
        STDDEV_SAMP(sales) OVER (PARTITION BY series_id ORDER BY d_num ROWS BETWEEN 55 PRECEDING AND 28 PRECEDING) AS rstd_28,
        -- Share of zero-sale days: a direct measure of intermittency.
        AVG(CASE WHEN sales = 0 THEN 1 WHEN sales > 0 THEN 0 END)
            OVER (PARTITION BY series_id ORDER BY d_num ROWS BETWEEN 55 PRECEDING AND 28 PRECEDING) AS rzero_28,
        -- Last day with a sale, as known at the forecast origin (t - 28).
        -- UNBOUNDED PRECEDING keeps this O(n): the frame start never moves.
        MAX(CASE WHEN sales > 0 THEN d_num END)
            OVER (PARTITION BY series_id ORDER BY d_num ROWS BETWEEN UNBOUNDED PRECEDING AND 28 PRECEDING) AS last_sale_d
    FROM spine
),
price_feats AS (
    -- Price features at the weekly grain (prices change weekly), then joined to days.
    SELECT series_id, wm_yr_wk,
           price / NULLIF(LAG(price, 1) OVER (PARTITION BY series_id ORDER BY wm_yr_wk), 0) - 1 AS price_change_1w,
           -- "Regular price" = highest price in the last 8 weeks. Being well below it is our
           -- promotion proxy: M5 has no promotion flag, so discounts must be inferred.
           price / NULLIF(MAX(price) OVER (PARTITION BY series_id ORDER BY wm_yr_wk
                                           ROWS BETWEEN 7 PRECEDING AND CURRENT ROW), 0) AS price_rel_max_8w,
           price / NULLIF(AVG(price) OVER (PARTITION BY series_id ORDER BY wm_yr_wk
                                           ROWS BETWEEN 51 PRECEDING AND CURRENT ROW), 0) AS price_rel_mean_52w
    FROM (SELECT series_id, wm_yr_wk, CAST(sell_price AS DOUBLE PRECISION) AS price
          FROM staging.prices
          WHERE series_id BETWEEN {sid_lo} AND {sid_hi}) p
)
SELECT
    w.series_id,
    c.date,
    w.d_num,
    sr.item_code, sr.dept_code, sr.cat_code, sr.store_code, sr.state_code,
    c.wday, c.month, c.day_of_month, c.is_weekend, c.event_type_code, c.is_event, c.days_to_next_event,
    CASE sr.state_id WHEN 'CA' THEN c.snap_ca WHEN 'TX' THEN c.snap_tx ELSE c.snap_wi END AS snap,
    w.sell_price,
    pf.price_change_1w,
    pf.price_rel_max_8w,
    pf.price_rel_mean_52w,
    CASE WHEN pf.price_rel_max_8w < 0.95 THEN 1 ELSE 0 END AS is_discount,
    w.d_num - sr.first_listed_d AS days_since_listed,
    w.lag_28, w.lag_35, w.lag_42, w.lag_49, w.lag_56,
    (w.lag_28 + w.lag_35 + w.lag_42 + w.lag_49) / 4.0 AS rmean_dow_4w,
    w.rmean_7, w.rmean_14, w.rmean_28, w.rmean_56,
    w.rstd_7, w.rstd_28, w.rzero_28,
    -- Capped so that a run that only re-reads the last `lookback_buffer_days` days gives
    -- exactly the same value as a full rebuild (requires buffer >= cap + 28; checked in config.py).
    LEAST(COALESCE((w.d_num - 28) - w.last_sale_d, {dsls_cap}), {dsls_cap}) AS dsls,
    w.sales AS target
FROM windowed w
JOIN staging.series sr ON sr.series_id = w.series_id
JOIN staging.calendar c ON c.d_num = w.d_num
LEFT JOIN price_feats pf ON pf.series_id = w.series_id AND pf.wm_yr_wk = c.wm_yr_wk
WHERE w.d_num >= {from_day}
  AND w.sell_price IS NOT NULL      -- only days where the item is actually on sale
