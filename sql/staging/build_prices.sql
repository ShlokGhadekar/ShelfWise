-- Weekly prices for the active subset, keyed by series_id.
-- Cleaning rules (counts are reported by the data-quality step, never silently hidden):
--   * NULL / non-positive prices are dropped -> that week is treated as "not on sale"
--   * duplicate (store, item, week) rows: keep one deterministically (highest price)
TRUNCATE staging.prices;

INSERT INTO staging.prices (series_id, wm_yr_wk, sell_price)
SELECT DISTINCT ON (s.series_id, p.wm_yr_wk)
       s.series_id, p.wm_yr_wk, p.sell_price
FROM raw.sell_prices p
JOIN staging.series s ON s.item_id = p.item_id AND s.store_id = p.store_id
WHERE p.sell_price IS NOT NULL AND p.sell_price > 0
ORDER BY s.series_id, p.wm_yr_wk, p.sell_price DESC;

ANALYZE staging.prices;
