"""Load the three M5 CSV files into the raw schema.

- calendar.csv and sell_prices.csv are streamed straight into Postgres with COPY
  (no pandas at all - the database parses the CSV, which is the fastest option).
- sales_train_evaluation.csv is wide (1,941 day columns > Postgres' 1,600-column
  limit), so each row's day values are packed into an integer array.

A file is only reloaded if its content hash changed (meta.file_registry). The raw
tables are small enough (30k rows / 7M rows) that a full reload is cheap; the expensive
layers (staging, features) are the ones built incrementally.
"""

from __future__ import annotations

import hashlib
import io
import logging
from pathlib import Path

import pandas as pd
import psycopg

from shelfwise.config import Settings
from shelfwise.db import connect, run_sql_file, scalar

log = logging.getLogger(__name__)

KAGGLE_HELP = """
Missing M5 data file: {path}

Download the data yourself (needs a free Kaggle account and accepting the competition rules):
    pip install kaggle            # and put your API token in ~/.kaggle/kaggle.json
    kaggle competitions download -c m5-forecasting-accuracy -p data/raw
    unzip data/raw/m5-forecasting-accuracy.zip -d data/raw
Required files in {raw_dir}: sales_train_evaluation.csv, calendar.csv, sell_prices.csv
"""

ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def check_files(settings: Settings) -> None:
    for path in settings.raw_files.values():
        if not path.exists():
            raise FileNotFoundError(KAGGLE_HELP.format(path=path, raw_dir=settings.paths.raw_dir))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _already_loaded(conn: psycopg.Connection, path: Path, digest: str) -> bool:
    return scalar(conn, "SELECT sha256 FROM meta.file_registry WHERE file_name = %s", (path.name,)) == digest


def _register(conn: psycopg.Connection, path: Path, digest: str, n_rows: int) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO meta.file_registry (file_name, sha256, size_bytes, n_rows, loaded_at)
               VALUES (%s, %s, %s, %s, now())
               ON CONFLICT (file_name) DO UPDATE
               SET sha256 = EXCLUDED.sha256, size_bytes = EXCLUDED.size_bytes,
                   n_rows = EXCLUDED.n_rows, loaded_at = now()""",
            (path.name, digest, path.stat().st_size, n_rows),
        )


def _copy_csv_file(conn: psycopg.Connection, path: Path, table: str) -> int:
    with conn.cursor() as cur:
        cur.execute(f"TRUNCATE {table}")
        with open(path, "rb") as fh, cur.copy(f"COPY {table} FROM STDIN WITH (FORMAT csv, HEADER true)") as cp:
            for block in iter(lambda: fh.read(1 << 20), b""):
                cp.write(block)
        return int(scalar(conn, f"SELECT COUNT(*) FROM {table}"))


def _load_sales_wide(conn: psycopg.Connection, path: Path, chunk_rows: int = 2000) -> int:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    day_cols = [c for c in header if c.startswith("d_")]
    expected = [f"d_{i}" for i in range(1, len(day_cols) + 1)]
    if header[: len(ID_COLS)] != ID_COLS or day_cols != expected:
        raise ValueError(f"Unexpected columns in {path.name}: day columns must be d_1..d_N in order")
    n_days = len(day_cols)
    log.info("%s: %d day columns", path.name, n_days)

    total = 0
    with conn.cursor() as cur:
        cur.execute("TRUNCATE raw.sales_wide")
        with cur.copy("COPY raw.sales_wide (id, item_id, dept_id, cat_id, store_id, state_id, n_days, sales) "
                      "FROM STDIN") as cp:
            # Reading in chunks keeps memory flat (the full file is ~120MB as CSV,
            # several GB as a pandas frame of int64).
            for chunk in pd.read_csv(path, chunksize=chunk_rows, dtype={c: "int32" for c in day_cols}):
                values = chunk[day_cols].to_numpy()
                buf = io.StringIO()
                for ids, row in zip(chunk[ID_COLS].itertuples(index=False), values, strict=True):
                    # Postgres array literal: {1,0,3,...}
                    buf.write("\t".join(ids) + f"\t{n_days}\t{{" + ",".join(map(str, row)) + "}\n")
                cp.write(buf.getvalue())
                total += len(chunk)
    return total


def load_raw(settings: Settings, force: bool = False) -> None:
    check_files(settings)
    files = settings.raw_files
    loaders = {
        "calendar": lambda c, p: _copy_csv_file(c, p, "raw.calendar"),
        "prices": lambda c, p: _copy_csv_file(c, p, "raw.sell_prices"),
        "sales": _load_sales_wide,
    }
    with connect() as conn:
        run_sql_file(conn, "ddl/01_raw.sql")
        for key, loader in loaders.items():
            path = files[key]
            digest = _sha256(path)
            if not force and _already_loaded(conn, path, digest):
                log.info("raw: %s unchanged since last load - skipping", path.name)
                continue
            log.info("raw: loading %s (%.0f MB)", path.name, path.stat().st_size / 1e6)
            n = loader(conn, path)
            _register(conn, path, digest, n)
            conn.commit()
            log.info("raw: %s -> %s rows", path.name, f"{n:,}")
        with conn.cursor() as cur:
            cur.execute("ANALYZE raw.sales_wide; ANALYZE raw.calendar; ANALYZE raw.sell_prices;")
