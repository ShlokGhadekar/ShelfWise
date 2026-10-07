"""raw -> staging, with incremental loading of new days.

How incremental loading works
-----------------------------
`meta.watermarks['staging_sales_max_day']` stores the last day already in staging.sales.
On each run we compare it with the last day we *want* (LAST_DAY env / config, or the
last day in the file):

  * nothing loaded yet / subset changed / --full-refresh  -> full rebuild
  * want > watermark                                       -> append only the new days
  * want == watermark                                      -> nothing to do
  * want < watermark                                       -> refuse (needs --full-refresh)

Limitation (documented in the README): if the source re-states *old* days, an
incremental run will not notice. Real pipelines handle late-arriving corrections with a
re-processing window; here a full refresh does it.
"""

from __future__ import annotations

import logging

from shelfwise.config import Settings
from shelfwise.db import clear_watermark, connect, get_watermark, run_sql_file, scalar, set_watermark
from shelfwise.sqlrender import Trusted

log = logging.getLogger(__name__)
WM_KEY = "staging_sales_max_day"


def _date_of(conn, d_num: int):
    return scalar(conn, "SELECT date FROM staging.calendar WHERE d_num = %s", (d_num,))


def target_last_day(conn, settings: Settings) -> int:
    file_days = scalar(conn, "SELECT MAX(n_days) FROM raw.sales_wide")
    if file_days is None:
        raise RuntimeError("raw.sales_wide is empty - run the 'load' step first")
    want = settings.data.last_day or file_days
    if want > file_days:
        raise ValueError(f"LAST_DAY={want} but the sales file only has {file_days} days")
    return int(want)


def build_staging(settings: Settings, full_refresh: bool = False) -> dict:
    subset = settings.subset.signature
    with connect() as conn:
        run_sql_file(conn, "ddl/02_staging.sql")
        last_day = target_last_day(conn, settings)
        wm, wm_subset = get_watermark(conn, WM_KEY)

        full = full_refresh or wm is None or wm_subset != subset
        if not full and last_day < wm:
            raise RuntimeError(
                f"Staging already holds days up to {wm} but LAST_DAY={last_day}. "
                "Moving 'today' backwards needs a full rebuild: add --full-refresh."
            )
        if not full and last_day == wm:
            log.info("staging: already up to date (day %d, subset %s)", wm, subset)
            return {"mode": "noop", "from_day": None, "to_day": wm}

        if full:
            reason = "requested" if full_refresh else ("first load" if wm is None else f"subset {wm_subset} -> {subset}")
            log.info("staging: FULL rebuild (%s), days 1..%d, subset %s", reason, last_day, subset)
            with conn.cursor() as cur:
                cur.execute("TRUNCATE staging.series, staging.prices")
            run_sql_file(conn, "staging/sales_indexes_drop.sql")
            clear_watermark(conn, WM_KEY)   # a crash half-way must lead to a full rebuild next time
            from_day = 1
        else:
            from_day = wm + 1
            log.info("staging: INCREMENTAL load of days %d..%d (subset %s)", from_day, last_day, subset)
            # Idempotency: remove rows a previously crashed run may have left behind after
            # the watermark, so re-running never creates duplicate days.
            with conn.cursor() as cur:
                cur.execute("DELETE FROM staging.sales WHERE d_num >= %s AND date >= "
                            "(SELECT date FROM staging.calendar WHERE d_num = %s)", (from_day, from_day))

        # Dimensions are small and cheap: refresh them every run.
        run_sql_file(conn, "staging/build_calendar.sql")
        run_sql_file(conn, "staging/build_series.sql", subset_filter=Trusted(settings.subset.sql_filter("w")))
        run_sql_file(conn, "staging/build_prices.sql")
        n_series = scalar(conn, "SELECT COUNT(*) FROM staging.series")
        if not n_series:
            raise RuntimeError(f"Subset {subset} matched no series - check SUBSET")
        conn.commit()

        # Long-format sales, inserted in blocks of days (date order -> BRIN-friendly,
        # bounded transaction size, progress in the log).
        step = settings.data.staging_chunk_days
        for lo in range(from_day, last_day + 1, step):
            hi = min(lo + step - 1, last_day)
            run_sql_file(conn, "staging/insert_sales_range.sql", from_day=lo, to_day=hi)
            conn.commit()
            log.info("staging: days %d-%d inserted (%d series)", lo, hi, n_series)

        run_sql_file(conn, "staging/sales_indexes_create.sql")
        run_sql_file(conn, "staging/finalize_series.sql",
                     from_date=_date_of(conn, from_day), to_date=_date_of(conn, last_day))
        set_watermark(conn, WM_KEY, last_day, subset)
        conn.commit()

    # REFRESH MATERIALIZED VIEW must run outside the long transaction.
    with connect(autocommit=True) as conn:
        run_sql_file(conn, "ddl/03_analytics.sql")
        with conn.cursor() as cur:
            cur.execute("REFRESH MATERIALIZED VIEW analytics.store_daily_sales")
    return {"mode": "full" if full else "incremental", "from_day": from_day, "to_day": last_day}
