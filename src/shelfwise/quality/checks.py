"""Automated data-quality checks -> meta.dq_results + reports/data_quality/.

Each check is a SQL query returning ONE number (`value`), compared with a threshold:
PASS if value <= threshold. Severity decides what a failure means:
  ERROR -> the pipeline stops (later steps would silently produce wrong numbers)
  WARN  -> logged and reported; the data was cleaned or the issue is tolerable
  INFO  -> context worth knowing (e.g. how many rows are "not on sale yet")
Every check can also return sample rows so the report shows *which* records are affected.

Why bother? Window-function features assume one row per series per day. A single
missing day silently turns lag_28 into lag_29 for every later row of that series -
no error, just a worse model. Cheap checks catch that class of bug up front.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import logging

import pandas as pd

from shelfwise.config import Settings
from shelfwise.db import connect, fetch_df, scalar

log = logging.getLogger(__name__)


@dataclasses.dataclass
class Check:
    name: str
    severity: str           # ERROR | WARN | INFO
    description: str
    value_sql: str
    threshold: float = 0
    sample_sql: str | None = None


@dataclasses.dataclass
class CheckResult:
    name: str
    severity: str
    status: str
    value: float
    threshold: float
    description: str
    sample: list[dict]


def build_checks(settings: Settings) -> list[Check]:
    q = settings.quality
    jump, outlier = q.price_jump_ratio, q.price_outlier_ratio
    return [
        # ---------------------------------------------------------------- calendar
        Check("calendar_missing_dates", "ERROR",
              "Gaps in the calendar date sequence (every day must exist exactly once).",
              """SELECT COUNT(*) FROM (SELECT date, LEAD(date) OVER (ORDER BY date) AS nxt FROM staging.calendar) x
                 WHERE nxt IS NOT NULL AND nxt - date <> 1""",
              sample_sql="""SELECT date, nxt FROM (SELECT date, LEAD(date) OVER (ORDER BY date) AS nxt
                            FROM staging.calendar) x WHERE nxt IS NOT NULL AND nxt - date <> 1"""),
        Check("calendar_covers_sales", "ERROR",
              "Days in the sales file that have no calendar row (they would be dropped by the join).",
              """SELECT GREATEST(MAX(n_days) - (SELECT COUNT(*) FROM staging.calendar), 0) FROM raw.sales_wide"""),
        # ---------------------------------------------------------------- completeness
        Check("sales_missing_days", "ERROR",
              "Series whose staging rows are not one-per-day from day 1 to the watermark "
              "(ROWS-based window features would be shifted).",
              """SELECT COUNT(*) FROM (
                     SELECT series_id, COUNT(*) AS n, MIN(d_num) AS lo, MAX(d_num) AS hi
                     FROM staging.sales GROUP BY series_id) x
                 WHERE n <> hi - lo + 1 OR lo <> 1
                    OR hi <> (SELECT value FROM meta.watermarks WHERE key = 'staging_sales_max_day')""",
              sample_sql="""SELECT series_id, COUNT(*) AS n, MIN(d_num) AS lo, MAX(d_num) AS hi
                            FROM staging.sales GROUP BY series_id
                            HAVING COUNT(*) <> MAX(d_num) - MIN(d_num) + 1 OR MIN(d_num) <> 1"""),
        Check("series_without_sales_rows", "ERROR",
              "Series in staging.series with no rows in staging.sales.",
              """SELECT COUNT(*) FROM staging.series s
                 WHERE NOT EXISTS (SELECT 1 FROM staging.sales x WHERE x.series_id = s.series_id)"""),
        # ---------------------------------------------------------------- validity
        Check("negative_sales_raw", "WARN",
              "Negative daily sales in the raw file (set to NULL in staging).",
              """SELECT COUNT(*) FROM raw.sales_wide w, unnest(w.sales) AS v WHERE v < 0""",
              sample_sql="""SELECT w.id, u.ord AS d_num, u.v FROM raw.sales_wide w,
                            unnest(w.sales) WITH ORDINALITY AS u(v, ord) WHERE u.v < 0"""),
        Check("null_sales_staging", "WARN",
              "NULL sales values in staging (invalid raw values).",
              "SELECT COUNT(*) FROM staging.sales WHERE sales IS NULL"),
        # ---------------------------------------------------------------- uniqueness
        Check("duplicate_series_ids_raw", "ERROR",
              "Duplicate series ids in the raw sales file.",
              "SELECT COUNT(*) - COUNT(DISTINCT id) FROM raw.sales_wide"),
        Check("duplicate_keys_sales", "ERROR",
              "Duplicate (series_id, date) rows in staging.sales.",
              """SELECT COUNT(*) FROM (SELECT series_id, date FROM staging.sales
                 GROUP BY series_id, date HAVING COUNT(*) > 1) x"""),
        Check("duplicate_keys_prices_raw", "WARN",
              "Duplicate (store, item, week) rows in the raw price file (deduplicated in staging).",
              """SELECT COUNT(*) FROM (SELECT 1 FROM raw.sell_prices
                 GROUP BY store_id, item_id, wm_yr_wk HAVING COUNT(*) > 1) x"""),
        # ---------------------------------------------------------------- prices
        Check("price_nonpositive_or_null", "WARN",
              "Raw prices that are NULL or <= 0 (dropped -> week treated as not on sale).",
              "SELECT COUNT(*) FROM raw.sell_prices WHERE sell_price IS NULL OR sell_price <= 0"),
        Check("price_week_jumps", "WARN",
              f"Week-over-week price changes larger than {jump}x up or down (possible data errors "
              "or deep promotions). Kept, but worth eyeballing.",
              f"""SELECT COUNT(*) FROM (
                     SELECT sell_price / LAG(sell_price) OVER (PARTITION BY series_id ORDER BY wm_yr_wk) AS r
                     FROM staging.prices) x
                  WHERE r > {jump} OR r < {1 / jump}""",
              threshold=0,
              sample_sql=f"""SELECT s.id, x.wm_yr_wk, x.prev_price, x.sell_price FROM (
                     SELECT series_id, wm_yr_wk, sell_price,
                            LAG(sell_price) OVER (PARTITION BY series_id ORDER BY wm_yr_wk) AS prev_price
                     FROM staging.prices) x JOIN staging.series s USING (series_id)
                  WHERE x.sell_price / x.prev_price > {jump} OR x.sell_price / x.prev_price < {1 / jump}"""),
        Check("price_outliers_vs_item_median", "WARN",
              f"Prices more than {outlier}x away from the same item's median price (all stores/weeks).",
              f"""WITH med AS (SELECT item_id, percentile_cont(0.5) WITHIN GROUP (ORDER BY sell_price) AS m
                              FROM raw.sell_prices WHERE sell_price > 0 GROUP BY item_id)
                  SELECT COUNT(*) FROM raw.sell_prices p JOIN med USING (item_id)
                  WHERE p.sell_price > {outlier} * med.m OR p.sell_price < med.m / {outlier}""",
              sample_sql=f"""WITH med AS (SELECT item_id, percentile_cont(0.5) WITHIN GROUP (ORDER BY sell_price) AS m
                              FROM raw.sell_prices WHERE sell_price > 0 GROUP BY item_id)
                  SELECT p.store_id, p.item_id, p.wm_yr_wk, p.sell_price, ROUND(CAST(med.m AS numeric), 2) AS item_median
                  FROM raw.sell_prices p JOIN med USING (item_id)
                  WHERE p.sell_price > {outlier} * med.m OR p.sell_price < med.m / {outlier}"""),
        Check("price_week_gaps", "WARN",
              "Series whose price history has missing weeks after first listing "
              "(weekly price LAG/rolling features assume contiguous weeks).",
              """WITH wk AS (SELECT DISTINCT wm_yr_wk, DENSE_RANK() OVER (ORDER BY wm_yr_wk) AS k
                             FROM staging.calendar)
                 SELECT COUNT(*) FROM (
                     SELECT p.series_id FROM staging.prices p JOIN wk USING (wm_yr_wk)
                     GROUP BY p.series_id HAVING MAX(wk.k) - MIN(wk.k) + 1 <> COUNT(*)) x"""),
        # ---------------------------------------------------------------- not yet on sale
        Check("rows_not_yet_on_sale", "INFO",
              "Share of staging rows before the item was on sale (no price). These zeros are "
              "NOT zero demand: excluded from training and evaluation.",
              "SELECT AVG(CASE WHEN sell_price IS NULL THEN 1.0 ELSE 0.0 END) FROM staging.sales",
              threshold=1.0),
        Check("sales_without_price", "WARN",
              "Rows with sales > 0 but no price (item sold while supposedly not on sale).",
              "SELECT COUNT(*) FROM staging.sales WHERE sales > 0 AND sell_price IS NULL",
              sample_sql="""SELECT s.id, x.date, x.sales FROM staging.sales x JOIN staging.series s USING (series_id)
                            WHERE x.sales > 0 AND x.sell_price IS NULL"""),
        Check("series_never_sold", "INFO",
              "Series with no sale at all so far (cannot be scaled for RMSSE; excluded from that metric).",
              "SELECT COUNT(*) FROM staging.series WHERE first_sale_d IS NULL",
              threshold=float("inf")),
        # ---------------------------------------------------------------- plausibility
        Check("zero_sales_last_28d", "INFO",
              "On-sale series with zero sales in the last 28 days - could be dead items or "
              "stock-outs (sales data can't tell them apart: we observe sales, not demand).",
              """WITH mx AS (SELECT value AS d FROM meta.watermarks WHERE key = 'staging_sales_max_day')
                 SELECT COUNT(*) FROM (
                     SELECT series_id FROM staging.sales, mx
                     WHERE d_num > mx.d - 28 AND sell_price IS NOT NULL
                     GROUP BY series_id HAVING SUM(sales) = 0) x""",
              threshold=float("inf")),
        Check("extreme_daily_sales", "INFO",
              "Days in the last year where sales exceed 10 standard deviations above the series mean.",
              """WITH mx AS (SELECT value AS d FROM meta.watermarks WHERE key = 'staging_sales_max_day'),
                      stats AS (
                          SELECT series_id, AVG(sales) AS m, STDDEV_SAMP(sales) AS sd
                          FROM staging.sales, mx WHERE d_num > mx.d - 365 AND sell_price IS NOT NULL
                          GROUP BY series_id)
                 SELECT COUNT(*) FROM staging.sales s JOIN stats USING (series_id), mx
                 WHERE s.d_num > mx.d - 365 AND stats.sd > 0 AND s.sales > stats.m + 10 * stats.sd""",
              threshold=float("inf")),
    ]


def _json_safe(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records", date_format="iso", default_handler=str))


def run_checks(settings: Settings, run_id: int | None = None) -> list[CheckResult]:
    results: list[CheckResult] = []
    with connect() as conn:
        for chk in build_checks(settings):
            value = scalar(conn, chk.value_sql)
            value = float(value) if value is not None else 0.0
            status = "PASS" if value <= chk.threshold else "FAIL"
            sample: list[dict] = []
            if status == "FAIL" and chk.sample_sql:
                sample = _json_safe(fetch_df(conn, f"{chk.sample_sql} LIMIT {settings.quality.sample_rows}"))
            results.append(CheckResult(chk.name, chk.severity, status, value, chk.threshold, chk.description, sample))
            level = logging.ERROR if (status == "FAIL" and chk.severity == "ERROR") else \
                logging.WARNING if status == "FAIL" else logging.INFO
            log.log(level, "DQ %-30s %-5s %-4s value=%g", chk.name, chk.severity, status, value)

        with conn.cursor() as cur:
            for r in results:
                cur.execute(
                    """INSERT INTO meta.dq_results (run_id, check_name, severity, status, value, threshold,
                                                   description, sample)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                    (run_id, r.name, r.severity, r.status, r.value,
                     r.threshold if r.threshold != float("inf") else None, r.description, json.dumps(r.sample)),
                )
    _write_report(settings, results)

    errors = [r for r in results if r.status == "FAIL" and r.severity == "ERROR"]
    if errors and settings.quality.fail_on_error:
        raise RuntimeError("Data-quality ERROR checks failed: " + ", ".join(r.name for r in errors)
                           + " (see reports/data_quality/latest.md)")
    return results


def _write_report(settings: Settings, results: list[CheckResult]) -> None:
    out = settings.paths.reports_dir / "data_quality"
    out.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame([{
        "check": r.name, "severity": r.severity, "status": r.status,
        "value": r.value, "threshold": r.threshold, "description": r.description,
    } for r in results])
    lines = [
        "# Data quality report", "",
        f"- generated: {ts}",
        f"- subset: `{settings.subset.signature}`",
        f"- checks: {len(results)}, failed: {int((df.status == 'FAIL').sum())} "
        f"(ERROR: {int(((df.status == 'FAIL') & (df.severity == 'ERROR')).sum())})", "",
        df.to_markdown(index=False, floatfmt=".4g"), "",
    ]
    for r in results:
        if r.sample:
            lines += [f"## Sample rows: {r.name}", "", pd.DataFrame(r.sample).to_markdown(index=False), ""]
    (out / "latest.md").write_text("\n".join(lines))
    (out / "latest.json").write_text(json.dumps([dataclasses.asdict(r) for r in results], indent=2, default=str))
    log.info("DQ report written to %s", out / "latest.md")
