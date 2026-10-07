-- Full refresh: drop indexes before the bulk load, rebuild them once at the end.
-- Maintaining a B-tree row-by-row during a 59M-row insert is several times slower
-- than one sorted index build afterwards.
ALTER TABLE staging.sales DROP CONSTRAINT IF EXISTS sales_pkey;
DROP INDEX IF EXISTS staging.sales_date_brin;
TRUNCATE staging.sales;
