"""The ordering rule turns forecasts into money - test it against hand-worked examples and theory."""

from __future__ import annotations

import numpy as np
import pytest

from shelfwise.decision.newsvendor import (
    critical_ratio,
    interpolate_quantile,
    order_quantity,
    overage_for_ratio,
    simulate_carryover,
    simulate_newsvendor,
    to_units,
)


def test_critical_ratio():
    assert critical_ratio(0.30, 0.10) == pytest.approx(0.75)
    assert critical_ratio(1, 1) == pytest.approx(0.5)          # symmetric costs -> order the median
    assert overage_for_ratio(0.30, 0.75) == pytest.approx(0.10)
    with pytest.raises(ValueError):
        critical_ratio(0, 1)


def test_interpolation_exact_between_and_clamped():
    levels = [0.1, 0.5, 0.9]
    v = np.array([[2.0, 4.0, 10.0]])
    assert interpolate_quantile(levels, v, 0.5)[0] == pytest.approx(4.0)
    assert interpolate_quantile(levels, v, 0.7)[0] == pytest.approx(7.0)   # halfway between P50 and P90
    assert interpolate_quantile(levels, v, 0.3)[0] == pytest.approx(3.0)
    assert interpolate_quantile(levels, v, 0.95)[0] == pytest.approx(10.0)  # clamped to P90
    assert interpolate_quantile(levels, v, 0.05)[0] == pytest.approx(2.0)


def test_order_rounding_rules_and_never_negative():
    v = np.array([[0.0, 1.2, 3.0], [-1.0, -0.5, 0.0], [0.0, 0.3, 1.0], [0.0, 2.5, 4.0]])
    assert order_quantity([0.1, 0.5, 0.9], v, 0.5).tolist() == [1.0, 0.0, 0.0, 3.0]          # nearest (half up)
    assert order_quantity([0.1, 0.5, 0.9], v, 0.5, rounding="up").tolist() == [2.0, 0.0, 1.0, 3.0]
    with pytest.raises(ValueError):
        to_units(np.array([1.0]), rounding="sideways")


def test_newsvendor_costs_hand_example():
    # price 10, c_u = 30% -> 3 per lost unit; c_o = 10% -> 1 per unsold unit
    r = simulate_newsvendor(order=np.array([5, 2]), demand=np.array([3, 4]), price=np.array([10.0, 10.0]),
                            underage_frac=0.3, overage_frac=0.1)
    assert r["overage_cost"].tolist() == [2.0, 0.0]     # 2 unsold units x 1
    assert r["underage_cost"].tolist() == [0.0, 6.0]    # 2 lost units x 3
    assert r["sold_units"].tolist() == [3, 2]


def test_critical_quantile_minimises_expected_cost():
    """Theory check: on a known demand distribution the CR-quantile order beats every other order."""
    rng = np.random.default_rng(0)
    demand = rng.poisson(6.0, size=200_000)
    for cr in (0.3, 0.5, 0.75, 0.9):
        co = overage_for_ratio(1.0, cr)
        costs = [np.mean(np.maximum(demand - q, 0) * 1.0 + np.maximum(q - demand, 0) * co) for q in range(25)]
        best_q = int(np.argmin(costs))
        # discrete newsvendor: smallest q with F(q) >= CR
        theory_q = int(np.quantile(demand, cr, method="inverted_cdf"))
        assert best_q == theory_q, (cr, best_q, theory_q)


def test_carryover_hand_example():
    # Day 1: order up to 5, demand 3 -> 2 left (holding 2 x 0.1 x 10 = 2)
    # Day 2: S=4, on hand 2 -> order 2, demand 6 -> lose 2 (2 x 0.3 x 10 = 6), 0 left
    r = simulate_carryover(order_up_to=np.array([[5.0, 4.0]]), demand=np.array([[3.0, 6.0]]),
                           price=np.full((1, 2), 10.0), underage_frac=0.3, holding_frac=0.1)
    assert r["ordered_units"].tolist() == [[5.0, 2.0]]
    assert r["overage_cost"].tolist() == [[2.0, 0.0]]
    assert r["underage_cost"].tolist() == [[0.0, 6.0]]
    assert r["leftover_units"].tolist() == [[2.0, 0.0]]


def test_carryover_never_orders_negative_when_overstocked():
    r = simulate_carryover(order_up_to=np.array([[10.0, 1.0, 1.0]]), demand=np.array([[1.0, 1.0, 1.0]]),
                           price=np.ones((1, 3)), underage_frac=0.3, holding_frac=0.1)
    assert (r["ordered_units"] >= 0).all()
    assert r["ordered_units"].tolist() == [[10.0, 0.0, 0.0]]
