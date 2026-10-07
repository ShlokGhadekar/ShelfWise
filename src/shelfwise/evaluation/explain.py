"""SHAP explanations for the point model of the last backtest fold.

SHAP splits each prediction into additive contributions, one per feature: "this
forecast is 0.4 above average because of lag_28, -0.1 because of the price".
Averaging |SHAP| over many rows gives a global importance ranking that, unlike
LightGBM's split counts, is consistent and comparable across features.

Note: the point model uses a tweedie objective with a log link, so SHAP values are in
log space (they add up to log(prediction)), not in units. Rankings and directions are
still valid; magnitudes are multiplicative effects.
"""

from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np
import pandas as pd

from shelfwise.config import Settings
from shelfwise.evaluation import plots
from shelfwise.forecasting.lgbm_model import FEATURES

log = logging.getLogger(__name__)


def compute_shap(settings: Settings) -> pd.DataFrame | None:
    model_files = sorted((settings.paths.models_dir / "backtest").glob("fold*_point.txt"))
    sample_path = settings.paths.cache_dir / "shap_sample.parquet"
    if not model_files or not sample_path.exists():
        log.warning("SHAP: no saved backtest model/sample - run the backtest step first")
        return None
    booster = lgb.Booster(model_file=str(model_files[-1]))
    X = pd.read_parquet(sample_path)[FEATURES]

    try:
        import shap

        values = shap.TreeExplainer(booster).shap_values(X)
        source = "shap.TreeExplainer"
    except Exception as exc:  # e.g. version mismatch with LightGBM categorical splits
        log.warning("shap.TreeExplainer failed (%s); using LightGBM's built-in TreeSHAP (pred_contrib)", exc)
        values = booster.predict(X, pred_contrib=True)[:, :-1]   # last column = expected value
        source = "lightgbm pred_contrib (TreeSHAP)"
        shap = None

    imp = pd.DataFrame({"feature": FEATURES, "mean_abs_shap": np.abs(values).mean(axis=0)}) \
        .sort_values("mean_abs_shap", ascending=False)
    out = settings.paths.reports_dir
    imp.to_csv(out / "results" / "shap_importance.csv", index=False)
    plots.bar_metric(imp.head(20), "mean_abs_shap", "feature",
                     "Top 20 features by mean |SHAP| (point model, last fold)", "mean |SHAP| (log scale units)",
                     out / "figures" / "shap_importance.png", highlight="*", fmt="{:.3f}")
    if shap is not None:
        try:
            import matplotlib.pyplot as plt

            shap.summary_plot(values, X, max_display=20, show=False)
            plt.gcf().savefig(out / "figures" / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
            plt.close("all")
        except Exception as exc:
            log.warning("SHAP beeswarm plot failed: %s", exc)
    log.info("SHAP computed with %s on %d rows", source, len(X))
    return imp
