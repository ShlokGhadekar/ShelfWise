from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shelfwise.evaluation.metrics import coverage, naive_scale, pinball, rmsse, wrmsse_by_level
from shelfwise.evaluation.stats_tests import cluster_bootstrap_mean, diebold_mariano
from shelfwise.forecasting.baselines import all_point_baselines, empirical_quantiles, seasonal_naive

# ------------------------------------------------------------------ metrics


def test_naive_scale_skips_leading_zeros_and_not_on_sale():
    h = np.array([[np.nan, 0, 0, 1, 2, 3], [0, 0, 0, 0, 0, 0]], dtype=float)
    s = naive_scale(h, power=2)
    assert s[0] == pytest.approx(1.0)   # diffs after first sale: (2-1), (3-2)
    assert np.isnan(s[1])               # never sold -> cannot be scaled


def test_rmsse_perfect_and_naive_like():
    a = np.array([[1.0, 2.0, 3.0]])
    assert rmsse(a, a, np.array([1.0]))[0] == 0
    assert rmsse(a, a + 1, np.array([1.0]))[0] == pytest.approx(1.0)
    assert rmsse(a, a + 1, np.array([4.0]))[0] == pytest.approx(0.5)


def test_pinball_and_coverage():
    y = np.array([10.0, 10.0])
    assert pinball(y, np.array([8.0, 12.0]), 0.9).tolist() == pytest.approx([1.8, 0.2])
    assert coverage(np.array([1.0, 5.0, 9.0]), np.array([0, 0, 0.0]), np.array([4, 6, 8.0])) == pytest.approx(2 / 3)


def test_wrmsse_zero_for_perfect_forecast():
    meta = pd.DataFrame({"state_id": ["CA", "CA"], "store_id": ["CA_1", "CA_1"], "cat_id": ["F", "H"],
                         "dept_id": ["F_1", "H_1"], "item_id": ["a", "b"]})
    rng = np.random.default_rng(1)
    hist = rng.poisson(3, size=(2, 100)).astype(float)
    act = rng.poisson(3, size=(2, 28)).astype(float)
    res = wrmsse_by_level(meta, hist, act, {"perfect": act, "off": act + 1}, np.array([10.0, 5.0]))
    assert (res[res["model"] == "perfect"]["score"] == 0).all()
    assert (res[res["model"] == "off"]["score"] > 0).all()

# ------------------------------------------------------------------ baselines


def test_baselines_only_use_data_up_to_origin():
    rng = np.random.default_rng(0)
    Y = rng.poisson(2, size=(5, 200)).astype(float)
    origin = 150
    before = all_point_baselines(Y, origin, 28)
    Y2 = Y.copy()
    Y2[:, origin:] = 999          # change the "future"
    after = all_point_baselines(Y2, origin, 28)
    for k in before:
        np.testing.assert_array_equal(before[k], after[k])
    eq1 = empirical_quantiles(Y, origin, 28, [0.5])[0.5]
    eq2 = empirical_quantiles(Y2, origin, 28, [0.5])[0.5]
    np.testing.assert_array_equal(eq1, eq2)


def test_seasonal_naive_alignment():
    Y = np.arange(1, 22, dtype=float)[None, :]      # day d has value d, days 1..21
    f = seasonal_naive(Y, origin=21, horizon=10, season=7)
    # forecast day 22 = day 15, day 28 = day 21, day 29 = day 15 again
    assert f[0].tolist() == [15, 16, 17, 18, 19, 20, 21, 15, 16, 17]

# ------------------------------------------------------------------ significance tests


def test_diebold_mariano_detects_clear_difference_and_not_noise():
    rng = np.random.default_rng(0)
    a = rng.normal(1.0, 0.1, 84)
    assert diebold_mariano(a, a)["mean_diff"] == 0
    res = diebold_mariano(a + 0.5, a)          # model B is clearly better
    assert res["p_b_better"] < 0.001
    noise = diebold_mariano(rng.normal(1, 0.1, 84), rng.normal(1, 0.1, 84))
    assert noise["p_two_sided"] > 0.001


def test_cluster_bootstrap_ci_contains_mean():
    rng = np.random.default_rng(0)
    v = rng.normal(0.2, 1, 3000)
    clusters = np.repeat(np.arange(300), 10)
    r = cluster_bootstrap_mean(v, clusters, n_boot=500)
    assert r["ci_low"] < r["mean"] < r["ci_high"]
    assert r["n_clusters"] == 300
