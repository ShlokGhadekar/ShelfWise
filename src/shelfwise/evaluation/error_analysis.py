"""Where does the model win, and where does it fail?

An overall average hides the interesting part. We slice the backtest errors by:
  * demand pattern (Syntetos-Boylan classes): smooth / erratic / intermittent / lumpy
  * item age: new items (< 90 days on sale) have little history to learn from
  * promotions: days with a price discount (our proxy - M5 has no promo flag)
  * holidays/events and SNAP days
  * forecast horizon (1-7, 8-14, ... days ahead) and product category

Metric per segment: pooled RMSSE = sqrt(mean over rows of squared error / series scale),
plus bias = (sum forecast - sum actual) / sum actual (positive = over-forecasting).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

ADI_CUT, CV2_CUT = 1.32, 0.49


def demand_class(history: np.ndarray, window: int = 365) -> np.ndarray:
    """Syntetos-Boylan classification from the last `window` days before the origin.

    ADI = average interval between non-zero sales (how often it sells)
    CV2 = squared coefficient of variation of the non-zero sizes (how much the size varies)
    """
    h = history[:, -window:]
    nz = np.nan_to_num(h) > 0
    n_obs = (~np.isnan(h)).sum(axis=1)
    n_nz = nz.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        adi = n_obs / n_nz
        sizes = np.where(nz, h, np.nan)
        cv2 = (np.nanstd(sizes, axis=1) / np.nanmean(sizes, axis=1)) ** 2
    cls = np.full(len(h), "no_sales", dtype=object)
    has = n_nz >= 2
    cls[has & (adi < ADI_CUT) & (cv2 < CV2_CUT)] = "smooth"
    cls[has & (adi < ADI_CUT) & (cv2 >= CV2_CUT)] = "erratic"
    cls[has & (adi >= ADI_CUT) & (cv2 < CV2_CUT)] = "intermittent"
    cls[has & (adi >= ADI_CUT) & (cv2 >= CV2_CUT)] = "lumpy"
    return cls


def add_segments(preds: pd.DataFrame, cat_names: dict[int, str]) -> pd.DataFrame:
    p = preds
    p["seg_item_age"] = pd.cut(p["days_since_listed"], [-np.inf, 89, 364, np.inf],
                               labels=["new (<90d)", "young (90-364d)", "established (1y+)"]).astype(str)
    p["seg_promo"] = np.where(p["is_discount"] == 1, "discount", "regular price")
    p["seg_event"] = np.where(p["is_event"] == 1, "event day", "normal day")
    p["seg_snap"] = np.where(p["snap"] == 1, "SNAP day", "non-SNAP day")
    p["seg_horizon"] = pd.cut(p["h"], [0, 7, 14, 21, 28], labels=["h 1-7", "h 8-14", "h 15-21", "h 22-28"]).astype(str)
    p["seg_category"] = p["cat_code"].map(cat_names).fillna("?")
    return p


SEGMENT_COLS = {
    "demand pattern": "seg_demand",
    "item age": "seg_item_age",
    "promotion (price-drop proxy)": "seg_promo",
    "event / holiday": "seg_event",
    "SNAP": "seg_snap",
    "horizon": "seg_horizon",
    "category": "seg_category",
}


def segment_table(preds: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Rows must have: actual, scale_sq, one column per model, and the seg_* columns."""
    rows = []
    ok = preds["scale_sq"].notna()
    p = preds[ok]
    for seg_type, col in SEGMENT_COLS.items():
        for seg, g in p.groupby(col, observed=True):
            for m in models:
                err = (g["actual"] - g[m]) ** 2 / g["scale_sq"]
                total = g["actual"].sum()
                rows.append({
                    "segment_type": seg_type, "segment": seg, "model": m, "n_rows": len(g),
                    "pooled_rmsse": float(np.sqrt(err.mean())),
                    "bias_pct": float(100 * (g[m].sum() - total) / total) if total > 0 else np.nan,
                })
    return pd.DataFrame(rows)


def interval_by_segment(preds: pd.DataFrame, lo: str, hi: str) -> pd.DataFrame:
    rows = []
    for seg_type, col in SEGMENT_COLS.items():
        for seg, g in preds.groupby(col, observed=True):
            inside = (g["actual"] >= g[lo]) & (g["actual"] <= g[hi])
            rows.append({"segment_type": seg_type, "segment": seg, "n_rows": len(g),
                         "coverage_p10_p90": float(inside.mean()),
                         "mean_width": float((g[hi] - g[lo]).mean())})
    return pd.DataFrame(rows)
