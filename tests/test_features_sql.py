"""Leakage and correctness tests for sql/analytics/features_select.sql (run on DuckDB).

The central guarantee: a feature row for target day t only depends on sales up to day
t - 28 (the forecast horizon). We test it by *perturbation*: change every sale from some
day P onwards and verify that features for all days t <= P + 27 did not move. If anyone
later adds e.g. a lag_7 or a window ending at the current row, this test fails.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import LAUNCH_DAY_S3, N_DAYS, date_of
from shelfwise.transform.features import FeatureWindow

BUFFER, CAP = 90, 60
FEATURE_COLS = [
    "lag_28", "lag_35", "lag_42", "lag_49", "lag_56", "rmean_dow_4w", "rmean_7", "rmean_14", "rmean_28",
    "rmean_56", "rstd_7", "rstd_28", "rzero_28", "dsls", "sell_price", "price_change_1w", "price_rel_max_8w",
    "price_rel_mean_52w", "is_discount", "days_since_listed", "wday", "snap", "is_event",
]


def run(con, from_day: int, obs_end: int = N_DAYS, future_end: int | None = None) -> pd.DataFrame:
    w = FeatureWindow(from_day=from_day, obs_end_day=obs_end, future_end_day=future_end or obs_end,
                      lo_day=from_day - BUFFER, dsls_cap=CAP)
    df = con.execute(w.sql(date_of)).df()
    return df.set_index(["series_id", "d_num"]).sort_index()


def sales_of(con, sid: int) -> pd.Series:
    df = con.execute(f"SELECT d_num, CASE WHEN sell_price IS NULL THEN NULL ELSE sales END AS s "
                     f"FROM staging.sales WHERE series_id = {sid} ORDER BY d_num").df()
    return df.set_index("d_num")["s"].astype(float)


def assert_frames_equal(a: pd.DataFrame, b: pd.DataFrame, cols: list[str]) -> None:
    pd.testing.assert_frame_equal(a[cols].astype(float), b[cols].astype(float), check_exact=False, rtol=1e-9)


def test_features_ignore_the_last_28_days(duck):
    """Core leakage test: perturbing sales from day P on must not change features for t <= P + 27."""
    before = run(duck, from_day=120)
    P = 170
    duck.execute(f"UPDATE staging.sales SET sales = sales + 50 WHERE d_num >= {P}")
    after = run(duck, from_day=120)

    safe_days = before.index.get_level_values("d_num") <= P + 27
    assert_frames_equal(before[safe_days], after[safe_days], FEATURE_COLS)
    # ...and the test is sensitive: from day P + 28 on, lag_28 sees the change.
    t = P + 28
    for sid in (1, 2):
        assert after.loc[(sid, t), "lag_28"] == before.loc[(sid, t), "lag_28"] + 50


def test_target_is_same_day_sales(duck):
    df = run(duck, from_day=120)
    s = sales_of(duck, 2)
    for t in (120, 150, 250):
        assert df.loc[(2, t), "target"] == s[t]


def test_lags_and_windows_match_python(duck):
    """Recompute a few features by hand and compare with the SQL."""
    df = run(duck, from_day=150)
    for sid in (1, 2):
        s = sales_of(duck, sid)
        for t in (150, 199, 250):
            row = df.loc[(sid, t)]
            assert row["lag_28"] == s[t - 28]
            assert row["lag_56"] == s[t - 56]
            win7 = s.loc[t - 34:t - 28]
            assert row["rmean_7"] == pytest.approx(win7.mean())
            win28 = s.loc[t - 55:t - 28]
            assert row["rstd_28"] == pytest.approx(win28.std(ddof=1))
            assert row["rzero_28"] == pytest.approx((win28 == 0).mean())
            assert row["rmean_dow_4w"] == pytest.approx(np.mean([s[t - k] for k in (28, 35, 42, 49)]))
            last_sale = s.loc[:t - 28][s.loc[:t - 28] > 0].index.max()
            assert row["dsls"] == min((t - 28) - last_sale, CAP)


def test_not_yet_on_sale_rows_are_excluded_and_not_treated_as_zero_demand(duck):
    df = run(duck, from_day=60)
    s3_days = df.loc[3].index
    assert s3_days.min() == LAUNCH_DAY_S3                      # no rows before launch
    t = LAUNCH_DAY_S3 + 28 + 5
    s = sales_of(duck, 3)
    # window [t-55, t-28] straddles the launch: only the on-sale days count
    expected = s.loc[t - 55:t - 28].dropna().mean()
    assert df.loc[(3, t), "rmean_28"] == pytest.approx(expected)
    assert df.loc[(3, LAUNCH_DAY_S3), "days_since_listed"] == 0


def test_incremental_build_equals_full_build(duck):
    """Recomputing only the newest days (with the look-back buffer) must give identical rows."""
    full = run(duck, from_day=120)
    inc = run(duck, from_day=223)
    cols = FEATURE_COLS + ["target"]
    assert_frames_equal(full.loc[full.index.get_level_values("d_num") >= 223], inc, cols)


def test_future_rows_have_null_target_and_valid_lags(duck):
    obs_end = 222
    df = run(duck, from_day=200, obs_end=obs_end, future_end=obs_end + 28)
    future = df[df.index.get_level_values("d_num") > obs_end]
    assert future["target"].isna().all()
    assert future.index.get_level_values("d_num").max() == obs_end + 28
    s = sales_of(duck, 2)
    assert df.loc[(2, obs_end + 28), "lag_28"] == s[obs_end]  # last future day uses the last observed day
    assert future["lag_28"].notna().all()
