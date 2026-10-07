"""Turning a probabilistic forecast into an order quantity (the newsvendor rule).

The idea in plain English
-------------------------
Every unit you stock is a small bet:
  * order one unit too FEW  -> a customer leaves empty-handed: you lose the margin
    (the "underage" cost, c_u)
  * order one unit too MANY -> it sits unsold: holding / markdown / waste cost
    (the "overage" cost, c_o)
The cost-minimising order is the demand quantile at the CRITICAL RATIO

        CR = c_u / (c_u + c_o)

e.g. c_u = 30% of price, c_o = 10% of price -> CR = 0.75 -> order the P75 of demand.
If stock-outs are expensive relative to leftovers, CR is high and you stock up; if
leftovers are expensive (perishables), CR is low.

This is why we forecast QUANTILES: a point forecast (the mean) is the right order only
in the special case c_u = c_o. A forecast that is "accurate on average" can still
produce bad orders if its uncertainty is wrong.

Practical details
  * We have P10/P50/P90; other levels are linearly interpolated between them (and
    clamped outside [P10, P90] - a documented approximation).
  * Demand comes in whole units, so the quantile must become an integer. We round to the
    NEAREST unit by default. Textbook theory says "round up" (the discrete newsvendor
    order is the smallest integer Q with F(Q) >= CR) - but that applies to the TRUE
    discrete distribution. LightGBM gives a continuous *estimate* of what is really an
    integer quantile; for slow movers the estimated P75 is often 0.2-0.4 where the true
    P75 is 0. Rounding those up orders 1 unit nearly every day.
    DISCLOSURE: the first backtest used "up" and over-ordered (92% of item-days without a
    stock-out vs a 75% target); we switched to "nearest" after seeing that result.
    `decision.rounding: up` in config.yaml reproduces the original rule.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

log = logging.getLogger(__name__)


def critical_ratio(underage_cost: float, overage_cost: float) -> float:
    if underage_cost <= 0 or overage_cost <= 0:
        raise ValueError("costs must be positive")
    return underage_cost / (underage_cost + overage_cost)


def overage_for_ratio(underage_cost: float, cr: float) -> float:
    """Inverse of critical_ratio: the overage cost that makes the critical ratio equal `cr`."""
    if not 0 < cr < 1:
        raise ValueError("cr must be in (0, 1)")
    return underage_cost * (1 - cr) / cr


def interpolate_quantile(levels: Sequence[float], values: np.ndarray, target: float) -> np.ndarray:
    """Estimate the `target` quantile from known quantile columns.

    levels: increasing quantile levels, e.g. [0.1, 0.5, 0.9]
    values: [n, len(levels)] predicted quantiles (assumed non-decreasing along axis 1)
    """
    levels = np.asarray(levels, dtype=float)
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    if np.any(np.diff(levels) <= 0):
        raise ValueError("levels must be strictly increasing")
    if target <= levels[0]:
        if target < levels[0]:
            log.debug("target quantile %.2f below lowest level %.2f - clamping", target, levels[0])
        return values[:, 0].copy()
    if target >= levels[-1]:
        if target > levels[-1]:
            log.debug("target quantile %.2f above highest level %.2f - clamping", target, levels[-1])
        return values[:, -1].copy()
    j = int(np.searchsorted(levels, target, side="right"))   # levels[j-1] <= target < levels[j]
    w = (target - levels[j - 1]) / (levels[j] - levels[j - 1])
    return (1 - w) * values[:, j - 1] + w * values[:, j]


def to_units(q: np.ndarray, rounding: str = "nearest") -> np.ndarray:
    """Continuous quantile estimate -> whole units, never negative ('nearest' or 'up')."""
    q = np.asarray(q, dtype=float)
    if rounding == "nearest":
        units = np.floor(q + 0.5)          # half-up; np.round would send 0.5 and 2.5 to the even number
    elif rounding == "up":
        units = np.ceil(q - 1e-9)
    else:
        raise ValueError("rounding must be 'nearest' or 'up'")
    return np.maximum(units, 0)


def order_quantity(levels: Sequence[float], values: np.ndarray, cr: float, rounding: str = "nearest") -> np.ndarray:
    """Newsvendor order: interpolated CR-quantile converted to whole units (see module notes)."""
    return to_units(interpolate_quantile(levels, values, cr), rounding)


# ----------------------------------------------------------------- cost simulation

def simulate_newsvendor(order: np.ndarray, demand: np.ndarray, price: np.ndarray,
                        underage_frac: float, overage_frac: float) -> dict[str, np.ndarray]:
    """Single-period ('perishable') simulation: every day starts from zero stock.

    Arrays are element-wise (any shape). Returns per-element arrays.
    cost = c_u * unmet demand + c_o * unsold units, with c_u/c_o as fractions of price.
    """
    order = np.nan_to_num(order)
    demand = np.nan_to_num(demand)
    price = np.nan_to_num(price)
    sold = np.minimum(order, demand)
    lost = demand - sold
    leftover = order - sold
    return {
        "underage_cost": underage_frac * price * lost,
        "overage_cost": overage_frac * price * leftover,
        "lost_units": lost,
        "leftover_units": leftover,
        "sold_units": sold,
        "demand_units": demand,
        "ordered_units": order,
    }


def simulate_carryover(order_up_to: np.ndarray, demand: np.ndarray, price: np.ndarray,
                       underage_frac: float, holding_frac: float) -> dict[str, np.ndarray]:
    """Multi-period simulation with inventory carried over (non-perishable goods).

    Arrays are [n_series, n_days] in time order. Each morning we top stock up to the
    order-up-to level S_t (we cannot send stock back, so if we already hold more than
    S_t we order nothing). Lead time is zero and unmet demand is lost (no back-orders).
    Every unit still on the shelf at night pays the holding cost.
    """
    S = np.nan_to_num(order_up_to)
    D = np.nan_to_num(demand)
    P = np.nan_to_num(price)
    n, T = D.shape
    on_hand = np.zeros(n)
    out = {k: np.zeros((n, T)) for k in ("underage_cost", "overage_cost", "lost_units", "leftover_units",
                                          "sold_units", "demand_units", "ordered_units")}
    for t in range(T):
        order = np.maximum(S[:, t] - on_hand, 0)
        available = on_hand + order
        sold = np.minimum(available, D[:, t])
        lost = D[:, t] - sold
        on_hand = available - sold
        out["underage_cost"][:, t] = underage_frac * P[:, t] * lost
        out["overage_cost"][:, t] = holding_frac * P[:, t] * on_hand
        out["lost_units"][:, t] = lost
        out["leftover_units"][:, t] = on_hand
        out["sold_units"][:, t] = sold
        out["demand_units"][:, t] = D[:, t]
        out["ordered_units"][:, t] = order
    return out
