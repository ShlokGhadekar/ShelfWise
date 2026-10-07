"""staging -> analytics.features (SQL window functions), full or incremental.

Incremental logic
-----------------
`meta.watermarks['features_obs_end_day']` = the last *observed* day the feature table
was built with. Rows for target days after that day were "future" rows (target NULL).
When new days arrive we:
  1. delete feature rows with target day >= old_obs_end + 1 (their targets/lags change)
  2. recompute only those days, re-reading `lookback_buffer_days` of history so every
     window is complete
  3. add new future rows up to new_obs_end + horizon
A unit test (tests/test_features_sql.py) checks incremental output == full rebuild output.
"""

from __future__ import annotations

import dataclasses
import logging

from shelfwise.config import MIN_LAG, Settings
from shelfwise.db import clear_watermark, connect, execute, get_watermark, run_sql_file, scalar, set_watermark
from shelfwise.sqlrender import load
from shelfwise.transform.staging import WM_KEY as STAGING_WM_KEY

log = logging.getLogger(__name__)
WM_KEY = "features_obs_end_day"

# Column order of analytics.features (must match the SELECT list in features_select.sql)
FEATURE_TABLE_COLUMNS = [
    "series_id", "date", "d_num",
    "item_code", "dept_code", "cat_code", "store_code", "state_code",
    "wday", "month", "day_of_month", "is_weekend", "event_type_code", "is_event", "days_to_next_event", "snap",
    "sell_price", "price_change_1w", "price_rel_max_8w", "price_rel_mean_52w", "is_discount",
    "days_since_listed",
    "lag_28", "lag_35", "lag_42", "lag_49", "lag_56", "rmean_dow_4w",
    "rmean_7", "rmean_14", "rmean_28", "rmean_56", "rstd_7", "rstd_28", "rzero_28", "dsls",
    "target",
]


@dataclasses.dataclass(frozen=True)
class FeatureWindow:
    """All parameters needed to render features_select.sql."""

    from_day: int        # first target day to output
    obs_end_day: int     # last observed day
    future_end_day: int  # last day to create (future) rows for
    lo_day: int          # first day read from staging (from_day - buffer)
    dsls_cap: int
    sid_lo: int = 0
    sid_hi: int = 2_147_483_647

    def sql(self, date_of) -> str:
        """`date_of` maps a day number to its calendar date (used for partition pruning)."""
        return load(
            "analytics/features_select.sql",
            from_day=self.from_day, obs_end_day=self.obs_end_day, future_end_day=self.future_end_day,
            lo_day=self.lo_day, dsls_cap=self.dsls_cap, sid_lo=self.sid_lo, sid_hi=self.sid_hi,
            lo_date=date_of(max(self.lo_day, 1)), obs_end_date=date_of(self.obs_end_day),
        )


def plan_window(from_day: int, obs_end_day: int, horizon: int, buffer: int, dsls_cap: int,
                calendar_max_day: int) -> FeatureWindow:
    assert horizon <= MIN_LAG, "feature SQL only supports horizons <= 28"
    return FeatureWindow(
        from_day=from_day,
        obs_end_day=obs_end_day,
        future_end_day=min(obs_end_day + horizon, calendar_max_day),
        lo_day=from_day - buffer,
        dsls_cap=dsls_cap,
    )


def build_features(settings: Settings, full_refresh: bool = False) -> dict:
    fcfg = settings.features
    subset = settings.subset.signature
    with connect() as conn:
        run_sql_file(conn, "ddl/03_analytics.sql")
        obs_end, st_subset = get_watermark(conn, STAGING_WM_KEY)
        if obs_end is None:
            raise RuntimeError("staging is empty - run the 'staging' step first")
        if st_subset != subset:
            raise RuntimeError(f"staging holds subset {st_subset} but SUBSET={subset}: run the 'staging' step first")
        wm, wm_subset = get_watermark(conn, WM_KEY)
        cal_max = int(scalar(conn, "SELECT MAX(d_num) FROM staging.calendar"))

        full = full_refresh or wm is None or wm_subset != subset or wm > obs_end
        if full:
            from_day = fcfg.start_day
            log.info("features: FULL rebuild from day %d (observed up to %d)", from_day, obs_end)
            execute(conn, "TRUNCATE analytics.features")
            # Forget the old watermark now: if this rebuild crashes half-way, the next run
            # must start a full rebuild again instead of thinking the table is up to date.
            clear_watermark(conn, WM_KEY)
        elif wm == obs_end:
            log.info("features: up to date (observed up to day %d)", obs_end)
            return {"mode": "noop"}
        else:
            from_day = wm + 1
            log.info("features: INCREMENTAL rebuild of target days >= %d (observed up to %d)", from_day, obs_end)
            from_date = scalar(conn, "SELECT date FROM staging.calendar WHERE d_num = %s", (from_day,))
            with conn.cursor() as cur:
                cur.execute("DELETE FROM analytics.features WHERE date >= %s", (from_date,))

        window = plan_window(from_day, obs_end, settings.forecast.horizon, fcfg.lookback_buffer_days,
                             fcfg.dsls_cap_days, cal_max)
        dates = dict(conn.execute("SELECT d_num, date FROM staging.calendar").fetchall())
        series_ids = [r[0] for r in conn.execute("SELECT series_id FROM staging.series ORDER BY series_id")]
        cols = ", ".join(FEATURE_TABLE_COLUMNS)
        # Bigger sorts in RAM for the window functions (this session only).
        execute(conn, "SET work_mem = '512MB'")
        conn.commit()

        bs = fcfg.series_batch_size
        for i in range(0, len(series_ids), bs):
            batch = series_ids[i:i + bs]
            w = dataclasses.replace(window, sid_lo=batch[0], sid_hi=batch[-1])
            execute(conn, f"INSERT INTO analytics.features ({cols})\n{w.sql(dates.__getitem__)}")
            # Committing per batch is safe: if the run dies half-way the watermark is
            # not advanced, so the next run deletes/recomputes the same days again.
            conn.commit()
            log.info("features: series %d-%d done (%d/%d)", batch[0], batch[-1],
                     min(i + bs, len(series_ids)), len(series_ids))
        set_watermark(conn, WM_KEY, obs_end, subset)
        conn.commit()

    with connect(autocommit=True) as conn:
        execute(conn, "ANALYZE analytics.features")
        n = scalar(conn, "SELECT COUNT(*) FROM analytics.features")
    log.info("features: table now has %s rows", f"{n:,}")
    return {"mode": "full" if full else "incremental", "from_day": from_day, "obs_end": obs_end}
