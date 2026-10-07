"""Simple baselines. Always build these first.

A fancy model is only interesting if it beats simple rules by a margin that matters.
Baselines also catch bugs: if LightGBM loses to "same as last week", something is
probably wrong with the features, not with retail.

All functions take the sales matrix Y [n_series, n_days] (column j = day j+1, NaN =
not on sale) and a forecast origin T (last observed day, 1-based). They only read
columns < T, i.e. days 1..T - a baseline can't leak either (tested in test_baselines.py).
"""

from __future__ import annotations

import warnings

import numpy as np


def _history(Y: np.ndarray, origin: int) -> np.ndarray:
    return Y[:, :origin]


def naive(Y: np.ndarray, origin: int, horizon: int) -> np.ndarray:
    """Tomorrow (and every later day) = the last observed day."""
    last = np.nan_to_num(_history(Y, origin)[:, -1])
    return np.repeat(last[:, None], horizon, axis=1)


def seasonal_naive(Y: np.ndarray, origin: int, horizon: int, season: int) -> np.ndarray:
    """Repeat the last `season` days (7 = same weekday last week; 28 = last 4 weeks)."""
    last = np.nan_to_num(_history(Y, origin)[:, -season:])
    reps = int(np.ceil(horizon / season))
    return np.tile(last, reps)[:, :horizon]


def moving_average(Y: np.ndarray, origin: int, horizon: int, window: int = 28) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN rows -> NaN -> 0
        m = np.nanmean(_history(Y, origin)[:, -window:], axis=1)
    return np.repeat(np.nan_to_num(m)[:, None], horizon, axis=1)


def empirical_quantiles(Y: np.ndarray, origin: int, horizon: int, quantiles: list[float],
                        window: int = 56) -> dict[float, np.ndarray]:
    """Quantile baseline: the q-th quantile of the last `window` days, flat over the horizon.

    This is the 'fair' baseline for interval forecasts and for the ordering decision:
    it uses the same newsvendor logic as the model, just with a naive distribution.
    """
    hist = _history(Y, origin)[:, -window:]
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for q in quantiles:
            v = np.nan_to_num(np.nanquantile(hist, q, axis=1))
            out[q] = np.repeat(v[:, None], horizon, axis=1)
    return out


def all_point_baselines(Y: np.ndarray, origin: int, horizon: int) -> dict[str, np.ndarray]:
    return {
        "naive": naive(Y, origin, horizon),
        "snaive7": seasonal_naive(Y, origin, horizon, 7),
        "snaive28": seasonal_naive(Y, origin, horizon, 28),
        "ma28": moving_average(Y, origin, horizon, 28),
        # Not a real contender: included to show why MAE-style metrics mislead on
        # intermittent data (predicting all zeros scores surprisingly well on MAE).
        "zeros": np.zeros((Y.shape[0], horizon), dtype=np.float32),
    }
