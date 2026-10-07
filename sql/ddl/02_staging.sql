-- STAGING layer: cleaned, typed, conformed. Long format for sales.

-- One row per item-store "series". series_id is a compact integer surrogate key:
-- a 4-byte int instead of a 30-character text id on 59M rows saves gigabytes.
-- The *_code columns are fixed integer encodings of the categorical ids, so
-- LightGBM sees identical codes in every fold and at prediction time.
-- (Beginner mistake: letting pandas build categories separately on the train and
-- test frames -> the same code means different stores in each.)
CREATE TABLE IF NOT EXISTS staging.series (
    series_id      integer PRIMARY KEY,
    id             text UNIQUE NOT NULL,
    item_id        text NOT NULL,
    dept_id        text NOT NULL,
    cat_id         text NOT NULL,
    store_id       text NOT NULL,
    state_id       text NOT NULL,
    item_code      integer NOT NULL,
    dept_code      smallint NOT NULL,
    cat_code       smallint NOT NULL,
    store_code     smallint NOT NULL,
    state_code     smallint NOT NULL,
    first_listed_d smallint,     -- first day the item has a price in this store (= on sale)
    first_sale_d   smallint      -- first day with sales > 0
);

CREATE TABLE IF NOT EXISTS staging.calendar (
    d_num               smallint PRIMARY KEY,   -- 1 for d_1
    date                date UNIQUE NOT NULL,
    wm_yr_wk            integer NOT NULL,
    weekday             text NOT NULL,
    wday                smallint NOT NULL,      -- M5 convention: 1 = Saturday ... 7 = Friday
    month               smallint NOT NULL,
    year                smallint NOT NULL,
    day_of_month        smallint NOT NULL,
    is_weekend          smallint NOT NULL,
    event_name_1        text,
    event_type_1        text,
    event_name_2        text,
    event_type_2        text,
    event_type_code     smallint NOT NULL,      -- 0 none, 1 Cultural, 2 National, 3 Religious, 4 Sporting
    is_event            smallint NOT NULL,
    days_to_next_event  smallint NOT NULL,      -- capped at 30
    snap_ca             smallint NOT NULL,
    snap_tx             smallint NOT NULL,
    snap_wi             smallint NOT NULL
);

CREATE TABLE IF NOT EXISTS staging.prices (
    series_id   integer NOT NULL,
    wm_yr_wk    integer NOT NULL,
    sell_price  numeric(10, 2) NOT NULL,
    PRIMARY KEY (series_id, wm_yr_wk)
);

-- The big table (~59M rows for ALL). Range-partitioned by date, one partition per year.
-- Why partition?
--   * queries filtered on a date range only touch the relevant partitions ("pruning")
--   * old years can be dropped/archived instantly (DROP TABLE on a partition)
--   * index builds and VACUUM work per partition (smaller, parallelisable)
-- Beginner gotcha: pruning only happens when the WHERE clause filters on `date`
-- itself. Filtering on d_num alone scans every partition - which is why the SQL in
-- this repo always adds a date predicate.
CREATE TABLE IF NOT EXISTS staging.sales (
    series_id   integer  NOT NULL,
    d_num       smallint NOT NULL,
    date        date     NOT NULL,
    sales       smallint,          -- NULL if the raw value was invalid (negative)
    sell_price  real               -- NULL = item not on sale yet in this store
) PARTITION BY RANGE (date);

CREATE TABLE IF NOT EXISTS staging.sales_2011 PARTITION OF staging.sales FOR VALUES FROM ('2011-01-01') TO ('2012-01-01');
CREATE TABLE IF NOT EXISTS staging.sales_2012 PARTITION OF staging.sales FOR VALUES FROM ('2012-01-01') TO ('2013-01-01');
CREATE TABLE IF NOT EXISTS staging.sales_2013 PARTITION OF staging.sales FOR VALUES FROM ('2013-01-01') TO ('2014-01-01');
CREATE TABLE IF NOT EXISTS staging.sales_2014 PARTITION OF staging.sales FOR VALUES FROM ('2014-01-01') TO ('2015-01-01');
CREATE TABLE IF NOT EXISTS staging.sales_2015 PARTITION OF staging.sales FOR VALUES FROM ('2015-01-01') TO ('2016-01-01');
CREATE TABLE IF NOT EXISTS staging.sales_2016 PARTITION OF staging.sales FOR VALUES FROM ('2016-01-01') TO ('2017-01-01');
CREATE TABLE IF NOT EXISTS staging.sales_default PARTITION OF staging.sales DEFAULT;

-- Indexes are created by sql/staging/create_sales_indexes.sql AFTER bulk loads
-- (building an index once at the end is much faster than updating it per row).
