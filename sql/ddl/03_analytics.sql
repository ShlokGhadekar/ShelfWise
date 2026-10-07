-- ANALYTICS layer: model-ready features and model outputs.

-- One row per (series, target day) for days where the item is on sale.
-- Every sales-derived column only uses data from 28+ days before `date`
-- (see sql/analytics/features_select.sql for the full leakage argument).
CREATE TABLE IF NOT EXISTS analytics.features (
    series_id           integer  NOT NULL,
    date                date     NOT NULL,
    d_num               smallint NOT NULL,
    -- identifiers (categorical for LightGBM)
    item_code           integer,
    dept_code           smallint,
    cat_code            smallint,
    store_code          smallint,
    state_code          smallint,
    -- calendar (known in advance)
    wday                smallint,
    month               smallint,
    day_of_month        smallint,
    is_weekend          smallint,
    event_type_code     smallint,
    is_event            smallint,
    days_to_next_event  smallint,
    snap                smallint,
    -- price (known in advance: the retailer sets its own prices)
    sell_price          real,
    price_change_1w     real,
    price_rel_max_8w    real,
    price_rel_mean_52w  real,
    is_discount         smallint,
    days_since_listed   smallint,
    -- sales history, all ending at least 28 days before `date`
    lag_28              real,
    lag_35              real,
    lag_42              real,
    lag_49              real,
    lag_56              real,
    rmean_dow_4w        real,
    rmean_7             real,
    rmean_14            real,
    rmean_28            real,
    rmean_56            real,
    rstd_7              real,
    rstd_28             real,
    rzero_28            real,
    dsls                smallint,
    -- target (NULL for future days we have to forecast)
    target              real,
    PRIMARY KEY (series_id, date)
) PARTITION BY RANGE (date);

CREATE TABLE IF NOT EXISTS analytics.features_2013 PARTITION OF analytics.features FOR VALUES FROM ('2013-01-01') TO ('2014-01-01');
CREATE TABLE IF NOT EXISTS analytics.features_2014 PARTITION OF analytics.features FOR VALUES FROM ('2014-01-01') TO ('2015-01-01');
CREATE TABLE IF NOT EXISTS analytics.features_2015 PARTITION OF analytics.features FOR VALUES FROM ('2015-01-01') TO ('2016-01-01');
CREATE TABLE IF NOT EXISTS analytics.features_2016 PARTITION OF analytics.features FOR VALUES FROM ('2016-01-01') TO ('2017-01-01');
CREATE TABLE IF NOT EXISTS analytics.features_default PARTITION OF analytics.features DEFAULT;

-- Live 28-day forecast + ordering recommendation, served by the API and dashboard.
CREATE TABLE IF NOT EXISTS analytics.forecasts (
    series_id          integer  NOT NULL,
    date               date     NOT NULL,
    d_num              smallint NOT NULL,
    horizon            smallint NOT NULL,     -- 1..28 days after the last observed day
    point              real,                  -- mean forecast (tweedie model)
    p10                real,
    p50                real,
    p90                real,
    baseline_snaive7   real,
    sell_price         real,
    order_qty          integer,               -- newsvendor order / order-up-to level
    critical_ratio     real,
    model_version      text,
    created_at         timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (series_id, date)
);

-- Tidy (long) results tables used by the dashboard/API.
CREATE TABLE IF NOT EXISTS analytics.model_metrics (
    run_at   timestamptz NOT NULL DEFAULT now(),
    model    text NOT NULL,
    fold     text NOT NULL,          -- '1', '2', ... or 'all'
    metric   text NOT NULL,
    value    double precision
);

CREATE TABLE IF NOT EXISTS analytics.decision_results (
    run_at          timestamptz NOT NULL DEFAULT now(),
    policy          text NOT NULL,
    fold            text NOT NULL,
    critical_ratio  double precision,
    metric          text NOT NULL,
    value           double precision
);

-- Pre-aggregated store x day totals for the dashboard (and the benchmark).
-- Created empty; filled by REFRESH after staging loads.
CREATE MATERIALIZED VIEW IF NOT EXISTS analytics.store_daily_sales AS
SELECT sr.store_id,
       s.date,
       SUM(s.sales)::bigint                     AS units,
       SUM(s.sales * s.sell_price)::numeric(14,2) AS revenue
FROM staging.sales s
JOIN staging.series sr USING (series_id)
GROUP BY sr.store_id, s.date
WITH NO DATA;

CREATE UNIQUE INDEX IF NOT EXISTS store_daily_sales_pk ON analytics.store_daily_sales (store_id, date);
