-- Build the series dimension for the active subset.
-- Codes and series_id are computed over the FULL raw file (before the subset
-- filter) so the same item/store always gets the same number, whatever subset
-- you run. ON CONFLICT DO NOTHING keeps existing ids stable on incremental runs.
INSERT INTO staging.series (series_id, id, item_id, dept_id, cat_id, store_id, state_id,
                            item_code, dept_code, cat_code, store_code, state_code)
SELECT series_id, id, item_id, dept_id, cat_id, store_id, state_id,
       item_code, dept_code, cat_code, store_code, state_code
FROM (
    SELECT r.id, r.item_id, r.dept_id, r.cat_id, r.store_id, r.state_id,
           CAST(ROW_NUMBER() OVER (ORDER BY r.id) AS integer)             AS series_id,
           CAST(DENSE_RANK() OVER (ORDER BY r.item_id) - 1 AS integer)    AS item_code,
           CAST(DENSE_RANK() OVER (ORDER BY r.dept_id) - 1 AS smallint)   AS dept_code,
           CAST(DENSE_RANK() OVER (ORDER BY r.cat_id) - 1 AS smallint)    AS cat_code,
           CAST(DENSE_RANK() OVER (ORDER BY r.store_id) - 1 AS smallint)  AS store_code,
           CAST(DENSE_RANK() OVER (ORDER BY r.state_id) - 1 AS smallint)  AS state_code
    FROM raw.sales_wide r
) w
WHERE {subset_filter}
ON CONFLICT (series_id) DO NOTHING;
