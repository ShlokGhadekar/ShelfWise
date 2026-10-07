"""Forecast accuracy metrics - and why the obvious ones mislead here.

WHY NOT PLAIN RMSE / MAE / MAPE / "ACCURACY"?
  * Scale: RMSE in units is dominated by the few items that sell 100/day. A model can
    look great while being useless for the 25,000 slow movers (or vice versa).
    -> scale each series' error by how hard that series is to forecast.
  * Zeros: more than half of all item-days in M5 have zero sales. MAPE divides by the
    actual (undefined at 0). MAE rewards forecasting 0 everywhere, because the median
    of an intermittent series is 0 - a "model" that never orders anything wins on MAE.
    (We include a `zeros` baseline in the tables to make this visible.)
  * "Accuracy" (exact hits) is a classification metric; demand is a quantity, and being
    off by 1 unit vs 50 units matters.

WHAT WE USE
  * RMSSE (root mean squared scaled error, the M5 metric): the model's MSE divided by the
    MSE of a one-step naive forecast on that series' own history (from its first sale).
    RMSSE < 1 roughly means "better than the in-sample naive forecast". Unit-free, so
    series can be averaged. Squared error -> it targets the MEAN, which is what you want
    when forecasts are summed up for replenishment.
  * WRMSSE: RMSSE computed at 12 aggregation levels (total, state, store, ..., item-store)
    and weighted by each series' recent dollar sales - the official M5 score. Rewards
    being right where the money is.
  * MASE: same idea with absolute errors (scaled MAE).
  * Pinball (quantile) loss for P10/P50/P90: the proper scoring rule for quantiles; scaled
    pinball (SPL) divides by the series' mean absolute naive difference, like MASE.
  * Coverage: share of actuals inside [P10, P90]. Should be ~80%. Too low = overconfident
    intervals; too high = intervals too wide to be useful. Count data caveat: with many
    zeros, P10 = P90 = 0 and actual = 0 counts as "covered", which inflates coverage for
    slow movers - we also report one-sided coverage and per-segment coverage.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import sparse

# M5's 12 aggregation levels (columns of the series metadata to group by)
LEVELS: dict[str, list[str]] = {
    "1_total": [],
    "2_state": ["state_id"],
    "3_store": ["store_id"],
    "4_cat": ["cat_id"],
    "5_dept": ["dept_id"],
    "6_state_cat": ["state_id", "cat_id"],
    "7_state_dept": ["state_id", "dept_id"],
    "8_store_cat": ["store_id", "cat_id"],
    "9_store_dept": ["store_id", "dept_id"],
    "10_item": ["item_id"],
    "11_item_state": ["item_id", "state_id"],
    "12_item_store": ["item_id", "store_id"],
}


def _nanmean(x: np.ndarray, axis: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(x, axis=axis)


def naive_scale(history: np.ndarray, power: int = 2) -> np.ndarray:
    """Per-series in-sample one-step naive error, from the first non-zero sale onwards.

    power=2 -> mean squared difference (RMSSE denominator)
    power=1 -> mean absolute difference (MASE / scaled pinball denominator)
    Leading zeros before the first sale are excluded (M5 convention) - otherwise items
    launched late would look artificially easy. NaN (not on sale) is skipped.
    Returns NaN for series with no usable history (never sold / constant).
    """
    h = history.astype(np.float32, copy=False)   # float32 keeps the full matrix at ~0.5GB
    started = np.cumsum(np.nan_to_num(h) > 0, axis=1) > 0
    diffs = h[:, 1:] - h[:, :-1]
    valid = started[:, :-1] & ~np.isnan(diffs)
    d = np.where(valid, np.abs(np.nan_to_num(diffs)) ** power, 0.0)
    n = valid.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = d.sum(axis=1, dtype=np.float64) / n
    scale[(n == 0) | (scale == 0)] = np.nan
    return scale


def rmsse(actual: np.ndarray, forecast: np.ndarray, scale_sq: np.ndarray) -> np.ndarray:
    """Per-series RMSSE. actual/forecast: [n_series, horizon], NaN where not evaluated."""
    mse = _nanmean((actual - forecast) ** 2, axis=1)
    return np.sqrt(mse / scale_sq)


def mase(actual: np.ndarray, forecast: np.ndarray, scale_abs: np.ndarray) -> np.ndarray:
    return _nanmean(np.abs(actual - forecast), axis=1) / scale_abs


def pinball(y: np.ndarray, q_pred: np.ndarray, q: float) -> np.ndarray:
    """Element-wise pinball loss. Under-forecasting costs q per unit, over-forecasting (1-q)."""
    diff = y - q_pred
    return np.maximum(q * diff, (q - 1) * diff)


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    ok = ~np.isnan(y)
    return float(np.mean((y[ok] >= lo[ok]) & (y[ok] <= hi[ok])))


# ------------------------------------------------------------------ multi-level WRMSSE

def aggregation_matrix(meta: pd.DataFrame, cols: list[str]) -> tuple[sparse.csr_matrix, pd.Index]:
    """Sparse 0/1 matrix G [n_groups x n_series]: G @ bottom_level_series = group totals."""
    n = len(meta)
    if not cols:
        return sparse.csr_matrix(np.ones((1, n), dtype=np.float32)), pd.Index(["Total"])
    keys = meta[cols].astype(str).agg("|".join, axis=1)
    codes, uniques = pd.factorize(keys, sort=True)
    G = sparse.csr_matrix((np.ones(n, dtype=np.float32), (codes, np.arange(n))), shape=(len(uniques), n))
    return G, pd.Index(uniques)


def wrmsse_by_level(meta: pd.DataFrame, history: np.ndarray, actual: np.ndarray, forecasts: dict[str, np.ndarray],
                    dollar_sales: np.ndarray, levels: dict[str, list[str]] | None = None) -> pd.DataFrame:
    """M5-style weighted RMSSE at each aggregation level.

    meta         series metadata aligned with the rows of the matrices
    history      [n, T] sales up to the forecast origin (NaN = not on sale -> counted as 0)
    actual       [n, H] actuals in the test window (NaN -> 0: not on sale means no sales)
    forecasts    name -> [n, H] point forecasts (NaN -> 0)
    dollar_sales [n] sales x price over the last 28 days before the origin (the weights)
    Returns a long DataFrame: level, model, score.
    """
    levels = levels or LEVELS
    hist0 = np.nan_to_num(history)
    act0 = np.nan_to_num(actual)
    w0 = np.nan_to_num(dollar_sales)
    rows = []
    for level, cols in levels.items():
        G, _ = aggregation_matrix(meta, cols)
        scale = naive_scale(np.asarray(G @ hist0), power=2)
        w = np.asarray(G @ w0).ravel()
        ok = ~np.isnan(scale) & (w > 0)
        if not ok.any():
            continue
        w = w[ok] / w[ok].sum()
        a = np.asarray(G @ act0)[ok]
        for name, f in forecasts.items():
            p = np.asarray(G @ np.nan_to_num(f))[ok]
            r = np.sqrt(np.mean((a - p) ** 2, axis=1) / scale[ok])
            rows.append({"level": level, "model": name, "score": float(np.sum(w * r))})
    return pd.DataFrame(rows)
