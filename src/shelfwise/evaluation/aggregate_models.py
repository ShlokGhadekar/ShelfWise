"""Classical time-series models (ETS, SARIMA) at aggregate levels, for comparison.

ETS/ARIMA fit one model per series. That is hopeless for 30k sparse item-store series,
but at the total / state / store level the series are smooth and long, which is where
these models shine. The fair comparison: sum LightGBM's bottom-level MEAN forecasts up
to the same level ("bottom-up") and compare RMSSE there.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from shelfwise.evaluation.metrics import aggregation_matrix, naive_scale

log = logging.getLogger(__name__)


def _fit_forecast(kind: str, y: np.ndarray, horizon: int) -> np.ndarray:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if kind == "ets":
            # Additive damped trend + additive weekly seasonality.
            res = ExponentialSmoothing(y, trend="add", damped_trend=True, seasonal="add",
                                       seasonal_periods=7, initialization_method="estimated").fit()
            return np.asarray(res.forecast(horizon))
        if kind == "sarima":
            # Weekly seasonal differencing + ARMA(1,1) + seasonal MA(1) - a robust daily-retail default.
            res = SARIMAX(y, order=(1, 0, 1), seasonal_order=(0, 1, 1, 7), trend="c").fit(disp=False)
            return np.asarray(res.forecast(horizon))
    raise ValueError(kind)


def compare_aggregate(preds: pd.DataFrame, Y: np.ndarray, meta: pd.DataFrame, kinds: list[str],
                      train_days: int, horizon: int) -> pd.DataFrame:
    """RMSSE at total/state/store level for ETS, SARIMA, LightGBM bottom-up and seasonal naive."""
    row_of = pd.Series(np.arange(len(meta)), index=meta["series_id"].to_numpy())
    levels = {"total": [], "state": ["state_id"], "store": ["store_id"]}
    rows = []
    for fold, f in preds.groupby("fold"):
        origin = int(f["origin"].iloc[0])
        r = row_of.loc[f["series_id"].to_numpy()].to_numpy()
        c = (f["h"].to_numpy() - 1).astype(int)
        dense = {}
        for name in ("actual", "lgbm_point", "snaive7"):
            M = np.zeros((len(meta), horizon))
            M[r, c] = np.nan_to_num(f[name].to_numpy())
            dense[name] = M
        hist = np.nan_to_num(Y[:, :origin])
        for level, cols in levels.items():
            G, labels = aggregation_matrix(meta, cols)
            H_agg = np.asarray(G @ hist)
            scale = naive_scale(H_agg, power=2)
            agg = {k: np.asarray(G @ v) for k, v in dense.items()}
            for g, label in enumerate(labels):
                if len(labels) == 1 and level != "total":
                    continue   # e.g. 'state' when the subset is a single state = same as total
                y_train = H_agg[g, -train_days:]
                fc = {"lgbm_bottom_up": agg["lgbm_point"][g], "snaive7": agg["snaive7"][g]}
                for kind in kinds:
                    try:
                        fc[kind] = _fit_forecast(kind, y_train, horizon)
                    except Exception as exc:
                        log.warning("%s failed for %s/%s fold %s: %s", kind, level, label, fold, exc)
                for model, p in fc.items():
                    err = np.sqrt(np.mean((agg["actual"][g] - p) ** 2) / scale[g])
                    rows.append({"fold": fold, "level": level, "series": label, "model": model, "rmsse": float(err)})
        log.info("aggregate models: fold %s done", fold)
    return pd.DataFrame(rows)
