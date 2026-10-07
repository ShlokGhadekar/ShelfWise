"""Global LightGBM models: one model for ALL series (not one per item).

Why global? 30k series, many with only a handful of sales per month. A per-series model
has almost nothing to learn from; a global model learns shared patterns ("weekends
are busier", "a 20% discount roughly doubles sales") from everyone and applies them to
each item through its features. This is what won the M5 competition.

We train:
  * a POINT model (objective = tweedie) -> the expected value (mean). Tweedie is a good
    fit for retail counts: lots of exact zeros plus a long right tail. Means add up, so
    this is the forecast to sum when aggregating to store/total level.
  * one QUANTILE model per level q (objective = quantile / pinball loss) -> P10/P50/P90.
    These feed the prediction intervals and the ordering decision.

Beginner notes
  * Quantiles do NOT add up: the sum of 30k item P90s is far above the P90 of the
    store total. Use the point model for aggregation.
  * Separately trained quantile models can "cross" (P10 > P50 on some rows). We fix it
    by sorting the quantile predictions row-wise.
  * Sub-sampling ROWS for training (max_train_rows) is fine. What is never fine is a
    random *train/validation split* - validation is always the latest days.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from shelfwise.config import LgbmCfg

log = logging.getLogger(__name__)

CATEGORICAL = ["item_code", "dept_code", "cat_code", "store_code", "state_code", "event_type_code"]
FEATURES = CATEGORICAL + [
    "wday", "month", "day_of_month", "is_weekend", "is_event", "days_to_next_event", "snap",
    "sell_price", "price_change_1w", "price_rel_max_8w", "price_rel_mean_52w", "is_discount",
    "days_since_listed",
    "lag_28", "lag_35", "lag_42", "lag_49", "lag_56", "rmean_dow_4w",
    "rmean_7", "rmean_14", "rmean_28", "rmean_56", "rstd_7", "rstd_28", "rzero_28", "dsls",
]
TARGET = "target"


def qname(q: float) -> str:
    """0.1 -> 'q10' (used for column names everywhere)."""
    return f"q{round(q * 100):02d}"


@dataclasses.dataclass
class ModelSet:
    point: lgb.Booster
    quantiles: dict[float, lgb.Booster]
    best_iterations: dict[str, int]

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[FEATURES]
        out = pd.DataFrame(index=df.index)
        out["lgbm_point"] = np.clip(self.point.predict(X), 0, None)
        qs = sorted(self.quantiles)
        if qs:
            Q = np.column_stack([self.quantiles[q].predict(X) for q in qs])
            Q = np.sort(np.clip(Q, 0, None), axis=1)   # no negative demand, no quantile crossing
            for j, q in enumerate(qs):
                out[f"lgbm_{qname(q)}"] = Q[:, j]
        return out

    def save(self, directory: Path, prefix: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.point.save_model(directory / f"{prefix}_point.txt")
        for q, b in self.quantiles.items():
            b.save_model(directory / f"{prefix}_{qname(q)}.txt")


def _dataset_params(cfg: LgbmCfg) -> dict:
    # The same base params are used to build the Dataset and to train every model on it,
    # so LightGBM never has to re-bin the data. feature_pre_filter=False stops LightGBM
    # from dropping features at Dataset construction based on min_data_in_leaf.
    return {**cfg.params, "feature_pre_filter": False}


def _params(cfg: LgbmCfg, objective: str, alpha: float | None = None) -> dict:
    p = _dataset_params(cfg)
    p["objective"] = objective
    if objective == "tweedie":
        p["tweedie_variance_power"] = cfg.tweedie_variance_power
        p["metric"] = "tweedie"
    elif objective == "quantile":
        p["alpha"] = alpha
        p["metric"] = "quantile"
    return p


def _train_one(cfg: LgbmCfg, params: dict, d_all: lgb.Dataset, d_fit: lgb.Dataset, d_val: lgb.Dataset,
               name: str) -> tuple[lgb.Booster, int]:
    booster = lgb.train(
        params, d_fit, num_boost_round=cfg.num_boost_round, valid_sets=[d_val], valid_names=["val"],
        callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False), lgb.log_evaluation(0)],
    )
    best = booster.best_iteration or cfg.num_boost_round
    log.info("  %-10s best iteration %d (val %s)", name, best,
             {k: round(v, 5) for k, v in booster.best_score.get("val", {}).items()})
    if cfg.refit_full:
        # Early stopping told us HOW LONG to train. Now retrain on all training days,
        # including the most recent `horizon` days we held out - for forecasting, the
        # latest weeks are the most valuable data, so throwing them away would hurt.
        booster = lgb.train(params, d_all, num_boost_round=best)
    return booster, best


def fit_models(train: pd.DataFrame, val_start: int, cfg: LgbmCfg, seed: int = 42) -> ModelSet:
    """Fit point + quantile models. Rows with d_num >= val_start form the inner validation window."""
    train = train.reset_index(drop=True)
    is_val = train["d_num"].to_numpy() >= val_start
    fit_idx = np.flatnonzero(~is_val)
    val_idx = np.flatnonzero(is_val)
    all_idx = np.arange(len(train))
    if cfg.max_train_rows and len(fit_idx) > cfg.max_train_rows:
        rng = np.random.default_rng(seed)
        frac = cfg.max_train_rows / len(fit_idx)
        fit_idx = np.sort(rng.choice(fit_idx, cfg.max_train_rows, replace=False))
        all_idx = np.sort(np.concatenate([fit_idx, rng.choice(val_idx, int(len(val_idx) * frac), replace=False)]))
        log.info("  sub-sampled training rows to %s", f"{len(fit_idx):,}")
    log.info("  rows: fit=%s val=%s", f"{len(fit_idx):,}", f"{len(val_idx):,}")

    # One binned Dataset, sliced with .subset(): avoids copying a multi-GB frame per model.
    d_base = lgb.Dataset(train[FEATURES], label=train[TARGET], categorical_feature=CATEGORICAL,
                         params=_dataset_params(cfg),
                         free_raw_data=False)
    d_base.construct()
    d_fit, d_val = d_base.subset(fit_idx), d_base.subset(val_idx)
    d_all = d_base if len(all_idx) == len(train) else d_base.subset(all_idx)

    best: dict[str, int] = {}
    point, best["point"] = _train_one(cfg, _params(cfg, cfg.point_objective), d_all, d_fit, d_val, "point")
    quantiles = {}
    for q in cfg.quantiles:
        quantiles[q], best[qname(q)] = _train_one(cfg, _params(cfg, "quantile", q), d_all, d_fit, d_val, qname(q))
    return ModelSet(point=point, quantiles=quantiles, best_iterations=best)


def feature_importance(models: ModelSet) -> pd.DataFrame:
    return pd.DataFrame({
        "feature": FEATURES,
        "gain": models.point.feature_importance("gain"),
        "splits": models.point.feature_importance("split"),
    }).sort_values("gain", ascending=False)
