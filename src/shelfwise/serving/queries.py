"""Read-only queries shared by the API and the dashboard.

All user input goes through bind parameters (:name) - never string formatting - so a
request like item_id="x'; DROP TABLE ..." is just a string that matches nothing.
The history query filters on (series_id, date), which is exactly the primary key of
staging.sales, so it is an index lookup in at most two yearly partitions.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
from sqlalchemy import text

from shelfwise.db import get_engine


def _df(sql: str, **params) -> pd.DataFrame:
    with get_engine().connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)


def find_series(item_id: str, store_id: str) -> dict | None:
    df = _df("""SELECT series_id, id, item_id, dept_id, cat_id, store_id, state_id, first_listed_d
                FROM staging.series WHERE item_id = :item AND store_id = :store""",
             item=item_id.upper(), store=store_id.upper())
    return None if df.empty else df.iloc[0].to_dict()


def stores() -> list[str]:
    return _df("SELECT DISTINCT store_id FROM staging.series ORDER BY 1")["store_id"].tolist()


def items(store_id: str, dept_id: str | None = None) -> pd.DataFrame:
    sql = "SELECT item_id, dept_id, cat_id FROM staging.series WHERE store_id = :store"
    if dept_id:
        sql += " AND dept_id = :dept"
    return _df(sql + " ORDER BY item_id", store=store_id, dept=dept_id)


def history(series_id: int, days: int) -> pd.DataFrame:
    last = _df("SELECT MAX(date) AS d FROM staging.sales WHERE series_id = :sid", sid=series_id)["d"].iloc[0]
    if last is None:
        return pd.DataFrame(columns=["date", "sales", "sell_price"])
    start = pd.Timestamp(last).date() - dt.timedelta(days=days - 1)
    return _df("""SELECT date, sales, sell_price FROM staging.sales
                  WHERE series_id = :sid AND date BETWEEN :start AND :end ORDER BY date""",
               sid=series_id, start=start, end=pd.Timestamp(last).date())


def forecast(series_id: int) -> pd.DataFrame:
    return _df("""SELECT date, horizon, point, p10, p50, p90, baseline_snaive7, sell_price, order_qty,
                         critical_ratio, model_version
                  FROM analytics.forecasts WHERE series_id = :sid ORDER BY date""", sid=series_id)


def model_metrics(fold: str = "all") -> pd.DataFrame:
    df = _df("SELECT model, metric, value FROM analytics.model_metrics WHERE fold = :fold", fold=fold)
    return df.pivot_table(index="model", columns="metric", values="value").reset_index() if not df.empty else df


def decision_results(fold: str = "all") -> pd.DataFrame:
    df = _df("SELECT policy, metric, value FROM analytics.decision_results WHERE fold = :fold", fold=fold)
    return df.pivot_table(index="policy", columns="metric", values="value").reset_index() if not df.empty else df


def latest_dq() -> pd.DataFrame:
    return _df("""SELECT check_name, severity, status, value, threshold, description, checked_at
                  FROM meta.dq_results
                  -- all checks of one run share checked_at (now() is fixed within a transaction)
                  WHERE checked_at = (SELECT MAX(checked_at) FROM meta.dq_results)
                  ORDER BY CASE severity WHEN 'ERROR' THEN 0 WHEN 'WARN' THEN 1 ELSE 2 END, check_name""")


def store_daily(store_id: str, days: int = 365) -> pd.DataFrame:
    return _df("""SELECT date, units, revenue FROM analytics.store_daily_sales
                  WHERE store_id = :store AND date > (SELECT MAX(date) FROM analytics.store_daily_sales) - CAST(:days AS integer)
                  ORDER BY date""", store=store_id, days=days)
