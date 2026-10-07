"""Walk-forward backtest: baselines + global LightGBM on every fold.

Output: data/cache/backtest_predictions.parquet - one row per (fold, series, test day)
with the actual value, every model's forecast and the segment columns used by error
analysis. All metrics, tests and the decision simulation are computed from this file,
so they can be re-run without re-training (`make evaluate`).
"""

from __future__ import annotations

import gc
import json
import logging

import numpy as np
import pandas as pd

from shelfwise.config import Settings
from shelfwise.forecasting.baselines import all_point_baselines, empirical_quantiles
from shelfwise.forecasting.data import load_features
from shelfwise.forecasting.folds import Fold, make_folds
from shelfwise.forecasting.lgbm_model import FEATURES, feature_importance, fit_models, qname
from shelfwise.transform.export import load_matrix

log = logging.getLogger(__name__)

# Columns carried from the feature rows into the prediction file (for error analysis).
KEEP_COLS = ["series_id", "d_num", "target", "sell_price", "item_code", "dept_code", "cat_code", "store_code",
             "state_code", "wday", "is_event", "event_type_code", "snap", "is_discount", "days_since_listed"]


def predictions_path(settings: Settings):
    return settings.paths.cache_dir / "backtest_predictions.parquet"


def folds_for(settings: Settings, last_day: int) -> list[Fold]:
    return make_folds(
        last_day=last_day,
        horizon=settings.forecast.horizon,
        n_folds=settings.backtest.n_folds,
        step=settings.backtest.step_days,
        train_window=settings.lgbm.train_window_days,
        first_available_day=settings.features.start_day,
    )


def add_baselines(out: pd.DataFrame, Y: np.ndarray, row_of: pd.Series, origin: int, horizon: int,
                  quantiles: list[float], emp_window: int) -> pd.DataFrame:
    """Look up baseline forecasts (computed on the dense matrix) for each prediction row."""
    rows = row_of.loc[out["series_id"].to_numpy()].to_numpy()
    cols = (out["d_num"].to_numpy() - origin - 1).astype(int)
    for name, P in all_point_baselines(Y, origin, horizon).items():
        out[name] = P[rows, cols]
    for q, P in empirical_quantiles(Y, origin, horizon, quantiles, emp_window).items():
        out[f"emp_{qname(q)}"] = P[rows, cols]
    return out


def run_backtest(settings: Settings) -> pd.DataFrame:
    cfg = settings.lgbm
    H = settings.forecast.horizon
    m = load_matrix(settings)
    Y, last_day = m["sales"], int(m["last_day"])
    row_of = pd.Series(np.arange(len(m["series_id"])), index=m["series_id"])
    folds = folds_for(settings, last_day)
    models_dir = settings.paths.models_dir / "backtest"
    results_dir = settings.paths.reports_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_preds, importances, fold_meta = [], [], []
    for fold in folds:
        log.info("fold %d: train days %d-%d (inner val from %d), test %d-%d",
                 fold.k, fold.train_start, fold.train_end, fold.val_start, fold.test_start, fold.test_end)
        # Training rows: target day <= origin. The test rows are loaded separately and
        # never touch fit_models() - the only safe way to keep the two apart.
        train = load_features(settings, fold.train_start, fold.train_end)
        models = fit_models(train, fold.val_start, cfg, seed=int(cfg.params.get("seed", 42)))
        n_train = len(train)
        del train
        gc.collect()

        test = load_features(settings, fold.test_start, fold.test_end)
        out = test[KEEP_COLS].rename(columns={"target": "actual"}).copy()
        out.insert(0, "fold", fold.k)
        out["origin"] = fold.origin
        out["h"] = (out["d_num"] - fold.origin).astype("int16")
        out = out.join(models.predict(test))
        out = add_baselines(out, Y, row_of, fold.origin, H, cfg.quantiles, settings.baselines.empirical_window_days)
        all_preds.append(out)

        imp = feature_importance(models)
        imp["fold"] = fold.k
        importances.append(imp)
        fold_meta.append(fold.as_dict() | {"n_train_rows": n_train, "n_test_rows": len(test),
                                           "best_iterations": models.best_iterations})

        if fold.k == folds[-1].k:
            # Keep the last fold's model + a test sample for SHAP explanations.
            models.save(models_dir, f"fold{fold.k}")
            n = min(settings.evaluation.shap_sample_rows, len(test))
            test.sample(n=n, random_state=0)[["series_id", "d_num", *FEATURES]].to_parquet(
                settings.paths.cache_dir / "shap_sample.parquet", index=False)
        del test, models
        gc.collect()

    preds = pd.concat(all_preds, ignore_index=True)
    preds.to_parquet(predictions_path(settings), index=False)
    pd.concat(importances).to_csv(results_dir / "feature_importance_by_fold.csv", index=False)
    (results_dir / "backtest_folds.json").write_text(json.dumps(fold_meta, indent=2))
    log.info("backtest: %s prediction rows written to %s", f"{len(preds):,}", predictions_path(settings))
    return preds
