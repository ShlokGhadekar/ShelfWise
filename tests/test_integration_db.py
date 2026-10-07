"""Post-run sanity checks against the real warehouse. Skipped by default.

Run after `make run` with:  pytest -m integration
They re-check the leakage guarantee on real data: every stored lag_28 must equal the
sales value 28 days earlier in staging.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def conn():
    from shelfwise.db import connect
    try:
        with connect() as c:
            yield c
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"database not reachable: {exc}")


def test_lag_28_matches_staging(conn):
    bad = conn.execute("""
        WITH sample AS (SELECT series_id FROM staging.series ORDER BY random() LIMIT 50)
        SELECT COUNT(*) FROM analytics.features f
        JOIN sample USING (series_id)
        JOIN staging.sales s ON s.series_id = f.series_id AND s.date = f.date - 28
        WHERE f.lag_28 IS DISTINCT FROM CAST(CASE WHEN s.sell_price IS NULL THEN NULL ELSE s.sales END AS real)
    """).fetchone()[0]
    assert bad == 0


def test_feature_rows_only_for_on_sale_days(conn):
    assert conn.execute("SELECT COUNT(*) FROM analytics.features WHERE sell_price IS NULL").fetchone()[0] == 0


def test_forecasts_are_ordered_and_non_negative(conn):
    bad = conn.execute("""SELECT COUNT(*) FROM analytics.forecasts
                          WHERE p10 > p50 OR p50 > p90 OR p10 < 0 OR point < 0 OR order_qty < 0""").fetchone()[0]
    assert bad == 0
