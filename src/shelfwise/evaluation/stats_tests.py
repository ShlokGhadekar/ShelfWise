"""Is the improvement over the baseline real, or noise?

A lower average error on one backtest could be luck. Two complementary tests:

1. Diebold-Mariano (DM) test - the standard test for comparing two forecasts over time.
   For each day we compute the average scaled loss of model A and model B across all
   series; d_t = loss_A - loss_B. If B is truly better, mean(d) > 0. Forecast errors on
   neighbouring days are correlated (same origin, shared shocks), so the variance uses a
   Newey-West (HAC) estimator, plus the Harvey-Leybourne-Newbold small-sample
   correction and a t-distribution. With 3 folds x 28 days we only have 84 time points,
   so this test is conservative - that is the honest situation.

2. Cluster bootstrap over series - per-series RMSSE difference, resampled in clusters
   (by default whole items: the same item in different stores and folds shares demand
   shocks, so treating those rows as independent would overstate certainty). Gives a
   95% confidence interval for the mean improvement.

A Wilcoxon signed-rank test is reported too, with the caveat that it assumes
independent series, which is not true here (it will look overly significant).
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, hac_lag: int = 7) -> dict:
    """DM test of H0: equal accuracy. Positive statistic -> model B has lower loss."""
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    d = d[~np.isnan(d)]
    T = len(d)
    if T < 3:
        return {"n": T, "mean_diff": float("nan"), "dm_stat": float("nan"), "p_two_sided": float("nan"),
                "p_b_better": float("nan")}
    dbar = d.mean()
    dc = d - dbar
    L = min(hac_lag, T - 1)
    gamma0 = np.dot(dc, dc) / T
    lrv = gamma0
    for k in range(1, L + 1):
        gk = np.dot(dc[k:], dc[:-k]) / T
        lrv += 2 * (1 - k / (L + 1)) * gk          # Bartlett weights keep the variance positive
    if lrv <= 0:
        lrv = gamma0
    if lrv <= 0:   # identical loss series: no difference to test
        return {"n": T, "mean_diff": float(dbar), "dm_stat": 0.0, "p_two_sided": 1.0, "p_b_better": 0.5}
    dm = dbar / np.sqrt(lrv / T)
    dm *= np.sqrt((T - 1) / T)                     # HLN correction for one-step-equivalent losses
    return {
        "n": T,
        "mean_diff": float(dbar),
        "dm_stat": float(dm),
        "p_two_sided": float(2 * stats.t.sf(abs(dm), df=T - 1)),
        "p_b_better": float(stats.t.sf(dm, df=T - 1)),
    }


def cluster_bootstrap_mean(values: np.ndarray, clusters: np.ndarray, n_boot: int = 2000,
                           seed: int = 0, ci: float = 0.95) -> dict:
    """Bootstrap CI for the mean of `values`, resampling whole clusters with replacement."""
    values = np.asarray(values, dtype=float)
    ok = ~np.isnan(values)
    values, clusters = values[ok], np.asarray(clusters)[ok]
    codes, _ = _factorize(clusters)
    G = codes.max() + 1
    sums = np.bincount(codes, weights=values, minlength=G)
    counts = np.bincount(codes, minlength=G).astype(float)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):       # loop keeps memory small even with many clusters
        idx = rng.integers(0, G, size=G)
        boot[b] = sums[idx].sum() / counts[idx].sum()
    alpha = (1 - ci) / 2
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(boot, alpha)),
        "ci_high": float(np.quantile(boot, 1 - alpha)),
        "p_mean_le_0": float(np.mean(boot <= 0)),   # one-sided: share of resamples with no improvement
        "n_clusters": int(G),
        "n_obs": int(len(values)),
    }


def cluster_bootstrap_ratio(numerator: np.ndarray, denominator: np.ndarray, clusters: np.ndarray,
                            n_boot: int = 2000, seed: int = 0, ci: float = 0.95) -> dict:
    """Bootstrap CI for sum(numerator) / sum(denominator), resampling whole clusters.

    Used for "cost saved / baseline cost" - a ratio of totals, not a mean of ratios
    (a mean of per-item percentages would let tiny items with huge % swings dominate).
    """
    codes, _ = _factorize(np.asarray(clusters))
    G = codes.max() + 1
    num = np.bincount(codes, weights=np.asarray(numerator, dtype=float), minlength=G)
    den = np.bincount(codes, weights=np.asarray(denominator, dtype=float), minlength=G)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, G, size=G)
        boot[b] = num[idx].sum() / den[idx].sum()
    alpha = (1 - ci) / 2
    return {"ratio": float(num.sum() / den.sum()), "ci_low": float(np.quantile(boot, alpha)),
            "ci_high": float(np.quantile(boot, 1 - alpha)), "n_clusters": int(G)}


def wilcoxon_signed_rank(diff: np.ndarray) -> dict:
    diff = np.asarray(diff, dtype=float)
    diff = diff[~np.isnan(diff) & (diff != 0)]
    if len(diff) < 10:
        return {"n": len(diff), "stat": float("nan"), "p_two_sided": float("nan")}
    res = stats.wilcoxon(diff)
    return {"n": int(len(diff)), "stat": float(res.statistic), "p_two_sided": float(res.pvalue)}


def _factorize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    uniques, codes = np.unique(x, return_inverse=True)
    return codes.ravel(), uniques
