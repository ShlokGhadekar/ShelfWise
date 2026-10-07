-- RAW layer: mirrors the Kaggle files. No cleaning here on purpose: if a cleaning
-- rule turns out to be wrong we can fix it and rebuild staging without re-reading CSVs.

-- The sales file is "wide": one row per item-store with 1,941 day columns.
-- Postgres allows at most 1,600 columns per table, so we keep the shape (one row
-- per series) but store the daily values in an integer ARRAY. The reshape to long
-- format happens in SQL (unnest ... WITH ORDINALITY) in the staging layer.
CREATE TABLE IF NOT EXISTS raw.sales_wide (
    id        text PRIMARY KEY,          -- e.g. HOBBIES_1_001_CA_1_evaluation
    item_id   text NOT NULL,
    dept_id   text NOT NULL,
    cat_id    text NOT NULL,
    store_id  text NOT NULL,
    state_id  text NOT NULL,
    n_days    integer NOT NULL,
    sales     integer[] NOT NULL         -- sales[1] = d_1, sales[2] = d_2, ...
);

-- Column order matches calendar.csv exactly so we can COPY the file straight in.
CREATE TABLE IF NOT EXISTS raw.calendar (
    date          date,
    wm_yr_wk      integer,
    weekday       text,
    wday          smallint,
    month         smallint,
    year          smallint,
    d             text,
    event_name_1  text,
    event_type_1  text,
    event_name_2  text,
    event_type_2  text,
    snap_ca       smallint,
    snap_tx       smallint,
    snap_wi       smallint
);

-- Weekly prices, matches sell_prices.csv column order.
CREATE TABLE IF NOT EXISTS raw.sell_prices (
    store_id    text,
    item_id     text,
    wm_yr_wk    integer,
    sell_price  numeric(10, 2)
);
