"""Database helpers.

Two kinds of connection:
- SQLAlchemy engine: convenient for pandas `read_sql` (dashboard, API, small queries).
- Raw psycopg connection: needed for COPY (the fastest way to move bulk data in/out
  of Postgres) and for executing whole multi-statement SQL files.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Iterator
from contextlib import contextmanager

import pandas as pd
import psycopg
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from shelfwise.config import get_settings
from shelfwise.sqlrender import load

log = logging.getLogger(__name__)
_ENGINE: Engine | None = None


def get_engine() -> Engine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = create_engine(get_settings().db.sqlalchemy_url, pool_pre_ping=True)
    return _ENGINE


@contextmanager
def connect(autocommit: bool = False) -> Iterator[psycopg.Connection]:
    """psycopg connection that commits on success and rolls back on error."""
    conn = psycopg.connect(get_settings().db.conninfo, autocommit=autocommit)
    try:
        yield conn
        if not autocommit:
            conn.commit()
    except Exception:
        if not autocommit:
            conn.rollback()
        raise
    finally:
        conn.close()


def execute(conn: psycopg.Connection, sql: str) -> None:
    """Run SQL text that may contain several statements (no bind params -> simple protocol)."""
    with conn.cursor() as cur:
        cur.execute(sql)


def run_sql_file(conn: psycopg.Connection, relative_path: str, **params: object) -> None:
    log.debug("Running sql/%s %s", relative_path, params or "")
    execute(conn, load(relative_path, **params))


def scalar(conn: psycopg.Connection, sql: str, params: tuple | None = None):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        row = cur.fetchone()
    return None if row is None else row[0]


def fetch_df(conn: psycopg.Connection, sql: str, params: tuple | None = None) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [d.name for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


def copy_query_to_buffer(conn: psycopg.Connection, select_sql: str) -> io.BytesIO:
    """Stream a SELECT out as CSV via COPY. ~10x faster than fetching rows through a cursor."""
    buf = io.BytesIO()
    with conn.cursor() as cur, cur.copy(f"COPY ({select_sql}) TO STDOUT WITH (FORMAT csv, HEADER true)") as cp:
        for chunk in cp:
            buf.write(chunk)
    buf.seek(0)
    return buf


def copy_df_to_table(conn: psycopg.Connection, df: pd.DataFrame, table: str) -> None:
    """Bulk insert a DataFrame with COPY (columns must match by name)."""
    cols = ", ".join(df.columns)
    data = df.to_csv(index=False, header=False, na_rep="")
    with conn.cursor() as cur, cur.copy(f"COPY {table} ({cols}) FROM STDIN WITH (FORMAT csv)") as cp:
        cp.write(data)


# ---------------------------------------------------------------- watermarks
# A watermark records "how far has this layer been built". It is what makes
# incremental runs possible: next time we only process days after it.

def get_watermark(conn: psycopg.Connection, key: str) -> tuple[int | None, str | None]:
    with conn.cursor() as cur:
        cur.execute("SELECT value, subset FROM meta.watermarks WHERE key = %s", (key,))
        row = cur.fetchone()
    return (row[0], row[1]) if row else (None, None)


def set_watermark(conn: psycopg.Connection, key: str, value: int, subset: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO meta.watermarks (key, value, subset, updated_at)
               VALUES (%s, %s, %s, now())
               ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, subset = EXCLUDED.subset,
                                               updated_at = now()""",
            (key, value, subset),
        )


def clear_watermark(conn: psycopg.Connection, key: str) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM meta.watermarks WHERE key = %s", (key,))
