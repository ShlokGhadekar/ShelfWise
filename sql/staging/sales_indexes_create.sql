-- B-tree primary key (series_id, date):
--   serves "history of one item in one store" (API, dashboard, exports) and guarantees
--   no duplicate days. On a partitioned table the key must include the partition key
--   (date), which it does.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint
                   WHERE conname = 'sales_pkey' AND conrelid = 'staging.sales'::regclass) THEN
        ALTER TABLE staging.sales ADD CONSTRAINT sales_pkey PRIMARY KEY (series_id, date);
    END IF;
END $$;

-- BRIN index on date: a few KB in size (vs ~1GB for a B-tree on 59M rows). It stores
-- the min/max date per block of pages, which works because we insert rows in date
-- order. Helps "all series, last N days" queries inside a partition.
CREATE INDEX IF NOT EXISTS sales_date_brin ON staging.sales USING brin (date) WITH (pages_per_range = 32);

ANALYZE staging.sales;
