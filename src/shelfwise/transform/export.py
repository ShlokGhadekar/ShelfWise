"""Export model inputs from Postgres to columnar files in data/cache/.

Design choice: Postgres does the heavy relational work (reshaping, window functions,
joins); the ML code reads compact Parquet/NumPy files. Pulling 20M+ rows through a
database cursor on every training run would dominate the run time, while a columnar
file reads in seconds and supports column/row filtering.

Outputs
  features.parquet    analytics.features (float32 / small ints)
  sales_matrix.npz    dense [series x day] matrices of sales and prices (NaN = not on sale)
                      -> used by baselines, metric scaling (RMSSE) and the decision layer
  series.parquet      series metadata (ids, store, dept, first listed / first sale day)
  calendar.parquet    calendar
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from shelfwise.config import Settings
from shelfwise.db import connect, copy_query_to_buffer, fetch_df, get_watermark
from shelfwise.transform.features import FEATURE_TABLE_COLUMNS
from shelfwise.transform.staging import WM_KEY as STAGING_WM_KEY

log = logging.getLogger(__name__)

INT_COLS = {
    "series_id": pa.int32(), "d_num": pa.int16(), "item_code": pa.int32(), "dept_code": pa.int16(),
    "cat_code": pa.int16(), "store_code": pa.int16(), "state_code": pa.int16(), "wday": pa.int16(),
    "month": pa.int16(), "day_of_month": pa.int16(), "is_weekend": pa.int16(), "event_type_code": pa.int16(),
    "is_event": pa.int16(), "days_to_next_event": pa.int16(), "snap": pa.int16(), "is_discount": pa.int16(),
}


def _feature_schema() -> dict:
    types = {}
    for c in FEATURE_TABLE_COLUMNS:
        if c == "date":
            types[c] = pa.date32()
        elif c == "days_since_listed" or c == "dsls":
            types[c] = pa.float32()   # may be NULL -> keep as float (NaN) for LightGBM
        else:
            types[c] = INT_COLS.get(c, pa.float32())
    return types


def cache_paths(settings: Settings) -> dict:
    d = settings.paths.cache_dir
    return {
        "features": d / "features.parquet",
        "matrix": d / "sales_matrix.npz",
        "series": d / "series.parquet",
        "calendar": d / "calendar.parquet",
    }


def export_all(settings: Settings, batch_series: int = 2000) -> None:
    paths = cache_paths(settings)
    settings.paths.cache_dir.mkdir(parents=True, exist_ok=True)
    col_types = _feature_schema()
    cols = ", ".join(FEATURE_TABLE_COLUMNS)

    with connect() as conn:
        obs_end, _ = get_watermark(conn, STAGING_WM_KEY)
        series = fetch_df(conn, """SELECT series_id, id, item_id, dept_id, cat_id, store_id, state_id,
                                          item_code, dept_code, cat_code, store_code, state_code,
                                          first_listed_d, first_sale_d
                                   FROM staging.series ORDER BY series_id""")
        calendar = fetch_df(conn, "SELECT * FROM staging.calendar ORDER BY d_num")
        series.to_parquet(paths["series"], index=False)
        calendar.to_parquet(paths["calendar"], index=False)
        sids = series["series_id"].to_numpy()

        # ---- features.parquet, written batch by batch so memory stays flat
        tmp = paths["features"].with_suffix(".parquet.tmp")
        writer = None
        n_rows = 0
        for i in range(0, len(sids), batch_series):
            lo, hi = int(sids[i]), int(sids[min(i + batch_series, len(sids)) - 1])
            buf = copy_query_to_buffer(
                conn, f"SELECT {cols} FROM analytics.features WHERE series_id BETWEEN {lo} AND {hi} "
                      "ORDER BY series_id, d_num")
            table = pacsv.read_csv(buf, convert_options=pacsv.ConvertOptions(column_types=col_types))
            table = table.cast(pa.schema([(c, col_types[c]) for c in table.column_names]))
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
            writer.write_table(table)
            n_rows += table.num_rows
        if writer is None:
            raise RuntimeError("analytics.features is empty - run the 'features' step first")
        writer.close()
        tmp.replace(paths["features"])
        log.info("export: features.parquet %s rows", f"{n_rows:,}")

        # ---- dense sales/price matrices. string_agg + np.fromstring is far lighter than
        # fetching 59M Python objects through the cursor.
        n_days = int(obs_end)
        sales = np.full((len(sids), n_days), np.nan, dtype=np.float32)
        prices = np.full((len(sids), n_days), np.nan, dtype=np.float32)
        row_of = {int(s): k for k, s in enumerate(sids)}
        for i in range(0, len(sids), batch_series):
            lo, hi = int(sids[i]), int(sids[min(i + batch_series, len(sids)) - 1])
            rows = conn.execute(f"""
                SELECT series_id,
                       string_agg(CAST(COALESCE(CASE WHEN sell_price IS NOT NULL THEN sales END, -1) AS text),
                                  ',' ORDER BY d_num),
                       string_agg(CAST(COALESCE(sell_price, -1) AS text), ',' ORDER BY d_num),
                       MIN(d_num), COUNT(*)
                FROM staging.sales
                WHERE series_id BETWEEN {lo} AND {hi} AND d_num <= {n_days}
                GROUP BY series_id""").fetchall()
            for sid, s_txt, p_txt, dmin, cnt in rows:
                if dmin != 1 or cnt != n_days:
                    raise RuntimeError(f"series {sid} is not dense (min day {dmin}, {cnt} rows) - see DQ report")
                k = row_of[int(sid)]
                sales[k] = np.array(s_txt.split(","), dtype=np.float32)
                prices[k] = np.array(p_txt.split(","), dtype=np.float32)
        sales[sales < 0] = np.nan     # -1 sentinel = not on sale / invalid
        prices[prices < 0] = np.nan
        np.savez_compressed(paths["matrix"], sales=sales, prices=prices, series_id=sids.astype(np.int32),
                            last_day=np.int32(n_days))
        log.info("export: sales_matrix.npz %d series x %d days", *sales.shape)


# ----------------------------------------------------------------- loaders

def load_matrix(settings: Settings) -> dict:
    with np.load(cache_paths(settings)["matrix"]) as z:
        return {k: z[k] for k in z.files}


def load_series(settings: Settings) -> pd.DataFrame:
    return pd.read_parquet(cache_paths(settings)["series"])


def load_calendar(settings: Settings) -> pd.DataFrame:
    return pd.read_parquet(cache_paths(settings)["calendar"])
