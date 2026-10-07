-- Layered warehouse:
--   raw       : files loaded as-is (only typed), so we can always re-derive everything
--   staging   : cleaned, conformed, long format (one row per series per day)
--   analytics : model-ready features, forecasts, evaluation results
--   meta      : pipeline bookkeeping (watermarks, file registry, data-quality results)
--   bench     : throw-away tables for the performance benchmark
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS meta;
CREATE SCHEMA IF NOT EXISTS bench;

CREATE TABLE IF NOT EXISTS meta.file_registry (
    file_name   text PRIMARY KEY,
    sha256      text NOT NULL,
    size_bytes  bigint NOT NULL,
    n_rows      bigint,
    loaded_at   timestamptz NOT NULL DEFAULT now()
);

-- key examples: 'staging_sales_max_day', 'features_obs_end_day'
CREATE TABLE IF NOT EXISTS meta.watermarks (
    key         text PRIMARY KEY,
    value       integer NOT NULL,
    subset      text NOT NULL,
    updated_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS meta.pipeline_runs (
    run_id      bigserial PRIMARY KEY,
    started_at  timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz,
    steps       text NOT NULL,
    subset      text NOT NULL,
    last_day    integer,
    status      text NOT NULL DEFAULT 'running',
    message     text
);

CREATE TABLE IF NOT EXISTS meta.dq_results (
    run_id      bigint,
    check_name  text NOT NULL,
    severity    text NOT NULL,      -- ERROR | WARN | INFO
    status      text NOT NULL,      -- PASS | FAIL
    value       double precision,
    threshold   double precision,
    description text,
    sample      jsonb,
    checked_at  timestamptz NOT NULL DEFAULT now()
);
