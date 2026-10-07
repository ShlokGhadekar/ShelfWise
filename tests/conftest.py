"""Shared fixtures.

The feature SQL is tested on DuckDB with a tiny synthetic warehouse - the exact same SQL
file the pipeline runs on Postgres - so the riskiest code (leakage!) is tested in
milliseconds without Docker.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

START = dt.date(2011, 1, 29)          # d_1 in M5
N_DAYS = 250
LAUNCH_DAY_S3 = 99                    # series 3 goes on sale on day 99 (first day of a week)


def date_of(d: int) -> dt.date:
    return START + dt.timedelta(days=int(d) - 1)


def week_of(d: int) -> int:
    return 11101 + (int(d) - 1) // 7


def make_tables(seed: int = 7) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    days = np.arange(1, N_DAYS + 1)
    cal = pd.DataFrame({
        "d_num": days,
        "date": [date_of(d) for d in days],
        "wm_yr_wk": [week_of(d) for d in days],
        "wday": (days - 1) % 7 + 1,
        "month": [date_of(d).month for d in days],
        "day_of_month": [date_of(d).day for d in days],
        "is_weekend": ((days - 1) % 7 < 2).astype(int),
        "event_type_code": np.where(days % 30 == 0, 1, 0),
        "is_event": np.where(days % 30 == 0, 1, 0),
        "days_to_next_event": 30 - days % 30,
        "snap_ca": np.where(days % 31 < 10, 1, 0),
        "snap_tx": 0,
        "snap_wi": 0,
    })
    series = pd.DataFrame({
        "series_id": [1, 2, 3],
        "item_code": [0, 1, 2], "dept_code": [0, 0, 1], "cat_code": [0, 0, 0],
        "store_code": [0, 0, 0], "state_code": [0, 0, 0], "state_id": ["CA", "CA", "CA"],
        "first_listed_d": [1, 1, LAUNCH_DAY_S3],
    })
    weeks = sorted(set(cal["wm_yr_wk"]))
    prices, sales = [], []
    for sid, first in zip(series["series_id"], series["first_listed_d"], strict=True):
        base = 2.0 + sid
        for w in weeks:
            if w >= week_of(first):
                discount = 0.8 if (w % 9 == 0) else 1.0
                prices.append((sid, w, round(base * discount, 2)))
        price_of = {w: p for s, w, p in prices if s == sid}
        lam = [0.4, 3.0, 1.5][sid - 1]          # intermittent, regular, new item
        for d in days:
            p = price_of.get(week_of(d))
            y = int(rng.poisson(lam)) if p is not None else 0
            sales.append((sid, d, date_of(d), y, p))
    return {
        "calendar": cal,
        "series": series,
        "prices": pd.DataFrame(prices, columns=["series_id", "wm_yr_wk", "sell_price"]),
        "sales": pd.DataFrame(sales, columns=["series_id", "d_num", "date", "sales", "sell_price"]),
    }


@pytest.fixture()
def duck():
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect(":memory:")
    con.execute("CREATE SCHEMA staging")
    for name, df in make_tables().items():
        con.register("tmp_df", df)
        con.execute(f"CREATE TABLE staging.{name} AS SELECT * FROM tmp_df")
        con.unregister("tmp_df")
    # pandas uses NaN for missing floats; make sure they are real SQL NULLs
    con.execute("UPDATE staging.sales SET sell_price = NULL WHERE isnan(sell_price)")
    con.execute("ALTER TABLE staging.prices ALTER sell_price TYPE DECIMAL(10, 2)")
    yield con
    con.close()
