-- Reshape wide -> long for days {from_day}..{to_day} and append them to staging.sales.
-- Called repeatedly in blocks of days, so a full load inserts the table in date
-- order and an incremental load only touches the new days.
--
-- unnest(array_slice) WITH ORDINALITY turns {5,0,3} into rows (5,1), (0,2), (3,3);
-- the ordinality tells us which day each value belongs to. This is the SQL way to
-- "melt" a wide table without a Python loop.
INSERT INTO staging.sales (series_id, d_num, date, sales, sell_price)
SELECT s.series_id,
       c.d_num,
       c.date,
       -- Sales can't be negative. Rather than guess a correction we mark the value
       -- as missing (NULL); the data-quality report counts how often this happens.
       CAST(CASE WHEN u.sales >= 0 THEN u.sales END AS smallint),
       CAST(p.sell_price AS real)
FROM raw.sales_wide w
JOIN staging.series s ON s.id = w.id
CROSS JOIN LATERAL unnest(w.sales[{from_day}:{to_day}]) WITH ORDINALITY AS u(sales, ord)
JOIN staging.calendar c ON c.d_num = {from_day} + u.ord - 1
LEFT JOIN staging.prices p ON p.series_id = s.series_id AND p.wm_yr_wk = c.wm_yr_wk;
