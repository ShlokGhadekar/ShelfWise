"""Production forecast: retrain on the latest data, forecast the next 28 days, recommend orders.

Same model recipe as the backtest (so the backtest numbers describe this model), with
the forecast origin = the last observed day. Results go to analytics.forecasts, which
the API and the dashboard read.
"""

from __future__ import annotations

import datetime as dt
import json
import logging

import numpy as np
import pandas as pd

from shelfwise.config import Settings
from shelfwise.db import connect, copy_df_to_table, execute
from shelfwise.decision.newsvendor import order_quantity
from shelfwise.forecasting.baselines import seasonal_naive
from shelfwise.forecasting.data import load_features
from shelfwise.forecasting.lgbm_model import fit_models, qname
from shelfwise.transform.export import load_calendar, load_matrix

log = logging.getLogger(__name__)


def run_forecast(settings: Settings) -> pd.DataFrame:
    cfg, H = settings.lgbm, settings.forecast.horizon
    m = load_matrix(settings)
    obs_end = int(m["last_day"])
    train_start = max(obs_end - cfg.train_window_days + 1, settings.features.start_day)
    log.info("forecast: training on days %d-%d, forecasting %d-%d", train_start, obs_end, obs_end + 1, obs_end + H)

    train = load_features(settings, train_start, obs_end)
    models = fit_models(train, val_start=obs_end - H + 1, cfg=cfg, seed=int(cfg.params.get("seed", 42)))
    del train

    future = load_features(settings, obs_end + 1, obs_end + H, observed_only=False)
    future = future[future["target"].isna()].reset_index(drop=True)
    if future.empty:
        raise RuntimeError("No future rows in features.parquet - the calendar may end at the last observed day")
    pred = models.predict(future)

    # Ordering recommendation from the quantile forecasts (see decision/newsvendor.py)
    qs = sorted(cfg.quantiles)
    cr = settings.decision.critical_ratio
    Q = pred[[f"lgbm_{qname(q)}" for q in qs]].to_numpy()
    order = order_quantity(qs, Q, cr)

    row_of = pd.Series(np.arange(len(m["series_id"])), index=m["series_id"])
    rows = row_of.loc[future["series_id"].to_numpy()].to_numpy()
    snaive = seasonal_naive(m["sales"], obs_end, H, 7)[rows, (future["d_num"] - obs_end - 1).to_numpy()]

    cal = load_calendar(settings).set_index("d_num")["date"]
    version = f"lgbm_{dt.datetime.now():%Y%m%d_%H%M}_d{obs_end}"
    out = pd.DataFrame({
        "series_id": future["series_id"].astype(int),
        "date": pd.to_datetime(cal.loc[future["d_num"]].to_numpy()).date,
        "d_num": future["d_num"].astype(int),
        "horizon": (future["d_num"] - obs_end).astype(int),
        "point": pred["lgbm_point"].round(4),
        "p10": pred.get("lgbm_q10", pd.Series(np.nan, index=pred.index)).round(4),
        "p50": pred.get("lgbm_q50", pd.Series(np.nan, index=pred.index)).round(4),
        "p90": pred.get("lgbm_q90", pd.Series(np.nan, index=pred.index)).round(4),
        "baseline_snaive7": snaive,
        "sell_price": future["sell_price"],
        "order_qty": order.astype(int),
        "critical_ratio": round(cr, 4),
        "model_version": version,
    })

    with connect() as conn:
        execute(conn, "TRUNCATE analytics.forecasts")
        copy_df_to_table(conn, out, "analytics.forecasts")
    models_dir = settings.paths.models_dir / "final"
    models.save(models_dir, "final")
    (models_dir / "metadata.json").write_text(json.dumps({
        "model_version": version, "obs_end_day": obs_end, "train_start_day": train_start,
        "best_iterations": models.best_iterations, "quantiles": qs, "critical_ratio": cr,
        "subset": settings.subset.signature,
    }, indent=2))
    log.info("forecast: %s rows written to analytics.forecasts (version %s)", f"{len(out):,}", version)
    return out
