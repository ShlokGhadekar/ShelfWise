"""Inventory cost simulation on the backtest folds: does a better forecast save money?

We replay each backtest test window (where the true sales are known) under several
ordering policies and add up the costs. All cost parameters are ASSUMPTIONS from
config.yaml (M5 has no cost or stock data):

  unit cost       = (1 - gross_margin) x price
  underage c_u    = (gross_margin + goodwill_frac) x price  per unit of unmet demand
  overage  c_o    = overage_frac x price                    per unsold unit (per night in carryover mode)

Policies (all use only information available at the forecast origin):
  naive_snaive7          order = last week's sales on the same weekday   <- the "naive approach"
  ma28                   order = average of the last 28 days
  empirical_newsvendor   order = CR-quantile of the last 56 days' sales (newsvendor with a naive distribution)
  lgbm_point             order = LightGBM mean forecast (good forecast, no uncertainty)
  lgbm_newsvendor        order = CR-quantile from LightGBM P10/P50/P90  <- the proposed policy

Comparing lgbm_point and lgbm_newsvendor separates the value of a better forecast from
the value of using its uncertainty; empirical_newsvendor separates the value of the
newsvendor rule from the value of the model.

Important caveat: we observe SALES, not DEMAND. When an item was out of stock, recorded
sales understate demand ("censored demand"), so the simulated stock-out costs are
a lower bound. See README limitations.
"""

from __future__ import annotations

import json
import logging
import warnings

import numpy as np
import pandas as pd

from shelfwise.config import Settings
from shelfwise.db import connect, copy_df_to_table, execute
from shelfwise.decision.newsvendor import (
    order_quantity,
    overage_for_ratio,
    simulate_carryover,
    simulate_newsvendor,
    to_units,
)
from shelfwise.evaluation import plots
from shelfwise.evaluation.stats_tests import cluster_bootstrap_ratio
from shelfwise.forecasting.backtest import predictions_path
from shelfwise.forecasting.lgbm_model import qname
from shelfwise.transform.export import load_matrix, load_series

log = logging.getLogger(__name__)
BASE_POLICY = "naive_snaive7"
MAIN_POLICY = "lgbm_newsvendor"


def _empirical_cr_quantile(Y: np.ndarray, origin: int, cr: float, window: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nan_to_num(np.nanquantile(Y[:, origin - window:origin], cr, axis=1))


def policy_orders(f: pd.DataFrame, Y: np.ndarray, row: np.ndarray, origin: int, cr: float,
                  quantiles: list[float], emp_window: int, rounding: str = "nearest") -> dict[str, np.ndarray]:
    # Both quantile policies use the same rounding rule so the comparison stays fair.
    Q = f[[f"lgbm_{qname(q)}" for q in quantiles]].to_numpy()
    emp = _empirical_cr_quantile(Y, origin, cr, emp_window)[row]
    return {
        "naive_snaive7": np.round(f["snaive7"].to_numpy()),
        "ma28": np.round(f["ma28"].to_numpy()),
        "empirical_newsvendor": to_units(emp, rounding),
        "lgbm_point": np.round(f["lgbm_point"].to_numpy()),
        "lgbm_newsvendor": order_quantity(quantiles, Q, cr, rounding),
    }


def simulate_fold(f: pd.DataFrame, orders: dict[str, np.ndarray], mode: str, cu: float, co: float,
                  H: int) -> dict[str, pd.DataFrame]:
    """Returns per-policy per-row (newsvendor) or per-series-day (carryover) results."""
    out = {}
    if mode == "newsvendor":
        for name, q in orders.items():
            r = simulate_newsvendor(q, f["actual"].to_numpy(), f["sell_price"].to_numpy(), cu, co)
            out[name] = pd.DataFrame(r).assign(series_id=f["series_id"].to_numpy())
        return out
    # carryover: dense [series x day] matrices in time order
    sids, r = np.unique(f["series_id"].to_numpy(), return_inverse=True)
    c = f["h"].to_numpy() - 1

    def dense(v):
        M = np.zeros((len(sids), H))
        M[r, c] = np.nan_to_num(v)
        return M

    D, P = dense(f["actual"].to_numpy()), dense(f["sell_price"].to_numpy())
    for name, q in orders.items():
        res = simulate_carryover(dense(q), D, P, cu, co)
        out[name] = pd.DataFrame({k: v.ravel() for k, v in res.items()}).assign(series_id=np.repeat(sids, H))
    return out


def summarise(res: pd.DataFrame) -> dict:
    total = res["underage_cost"].sum() + res["overage_cost"].sum()
    demand = res["demand_units"].sum()
    return {
        "total_cost": float(total),
        "stockout_cost": float(res["underage_cost"].sum()),
        "overage_cost": float(res["overage_cost"].sum()),
        "fill_rate": float(res["sold_units"].sum() / demand) if demand else np.nan,
        "stockout_rate": float((res["lost_units"] > 0).mean()),       # share of item-days with unmet demand
        "service_level": float((res["lost_units"] == 0).mean()),      # compare with the critical ratio
        "units_ordered": float(res["ordered_units"].sum()),
        "leftover_units": float(res["leftover_units"].sum()),
    }


def run_decision(settings: Settings) -> dict:
    dcfg = settings.decision
    H = settings.forecast.horizon
    quantiles = sorted(settings.lgbm.quantiles)
    emp_window = settings.baselines.empirical_window_days
    out = settings.paths.reports_dir / "results"
    figs = settings.paths.reports_dir / "figures"

    preds = pd.read_parquet(predictions_path(settings))
    m = load_matrix(settings)
    Y = m["sales"]
    row_of = pd.Series(np.arange(len(m["series_id"])), index=m["series_id"])
    item_of = load_series(settings).set_index("series_id")["item_id"]
    cu, co, cr = dcfg.underage_frac, dcfg.overage_frac, dcfg.critical_ratio
    log.info("decision: mode=%s c_u=%.2f c_o=%.2f (x price) -> critical ratio %.3f", dcfg.mode, cu, co, cr)

    def run(cr_value: float, co_value: float) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        rows, per_policy = [], {}
        for k, f in preds.groupby("fold"):
            origin = int(f["origin"].iloc[0])
            row = row_of.loc[f["series_id"].to_numpy()].to_numpy()
            orders = policy_orders(f, Y, row, origin, cr_value, quantiles, emp_window, dcfg.rounding)
            for name, res in simulate_fold(f, orders, dcfg.mode, cu, co_value, H).items():
                rows.append({"fold": k, "policy": name} | summarise(res))
                per_policy.setdefault(name, []).append(res.assign(fold=k))
        return pd.DataFrame(rows), {k: pd.concat(v) for k, v in per_policy.items()}

    by_fold, detail = run(cr, co)
    summary = by_fold.groupby("policy").agg(
        total_cost=("total_cost", "sum"), stockout_cost=("stockout_cost", "sum"),
        overage_cost=("overage_cost", "sum"), fill_rate=("fill_rate", "mean"),
        stockout_rate=("stockout_rate", "mean"), service_level=("service_level", "mean"),
        units_ordered=("units_ordered", "sum"), leftover_units=("leftover_units", "sum")).reset_index()
    base_cost = summary.set_index("policy").loc[BASE_POLICY, "total_cost"]
    summary["saving_vs_naive_pct"] = 100 * (base_cost - summary["total_cost"]) / base_cost
    summary = summary.sort_values("total_cost")

    # Uncertainty of the headline saving: cluster bootstrap over items
    def per_series_cost(name):
        d = detail[name]
        return (d["underage_cost"] + d["overage_cost"]).groupby([d["fold"], d["series_id"]]).sum()

    ci = {}
    main = per_series_cost(MAIN_POLICY)
    for other in (BASE_POLICY, "empirical_newsvendor", "lgbm_point"):
        base = per_series_cost(other).reindex(main.index).fillna(0)
        clusters = item_of.loc[main.index.get_level_values("series_id")].to_numpy()
        r = cluster_bootstrap_ratio((base - main).to_numpy(), base.to_numpy(), clusters,
                                    settings.evaluation.bootstrap_reps)
        ci[other] = {k: (100 * v if k != "n_clusters" else v) for k, v in r.items()}

    # Sensitivity: how do savings change if our cost assumptions are wrong?
    sens = []
    for cr_s in dcfg.sensitivity_critical_ratios:
        co_s = overage_for_ratio(cu, cr_s)
        bf, _ = run(cr_s, co_s)
        tot = bf.groupby("policy")["total_cost"].sum()
        for pol, cost in tot.items():
            sens.append({"critical_ratio": cr_s, "overage_frac": co_s, "policy": pol, "total_cost": cost,
                         "saving_vs_naive_pct": 100 * (tot[BASE_POLICY] - cost) / tot[BASE_POLICY]})
    sens = pd.DataFrame(sens)

    summary.to_csv(out / "decision_summary.csv", index=False)
    by_fold.to_csv(out / "decision_by_fold.csv", index=False)
    sens.to_csv(out / "decision_sensitivity.csv", index=False)
    (out / "decision_bootstrap.json").write_text(json.dumps(ci, indent=2))

    plots.bar_metric(summary, "total_cost", "policy",
                     f"Simulated cost over {preds['fold'].nunique()} backtest windows (lower is better)",
                     "total cost ($, assumption-based)", figs / "decision_costs.png", MAIN_POLICY, BASE_POLICY,
                     fmt="{:,.0f}")
    plots.lines_by_x(sens[sens["policy"] != BASE_POLICY], "critical_ratio", "saving_vs_naive_pct", "policy",
                     "Saving vs naive ordering under different cost assumptions", "critical ratio c_u/(c_u+c_o)",
                     "% cost saved vs naive", figs / "decision_sensitivity.png", MAIN_POLICY, "empirical_newsvendor")

    long = by_fold.melt(id_vars=["fold", "policy"], var_name="metric", value_name="value")
    long_all = summary.melt(id_vars=["policy"], var_name="metric", value_name="value").assign(fold="all")
    db = pd.concat([long.assign(fold=long["fold"].astype(str)), long_all]).assign(critical_ratio=cr)
    with connect() as conn:
        execute(conn, "TRUNCATE analytics.decision_results")
        copy_df_to_table(conn, db[["policy", "fold", "critical_ratio", "metric", "value"]],
                         "analytics.decision_results")

    md = [
        "# Decision layer - simulated inventory costs (auto-generated)", "",
        f"Mode `{dcfg.mode}`; costs as a fraction of price: underage c_u = {cu:.2f}, overage c_o = {co:.2f} "
        f"-> critical ratio {cr:.3f}; quantile orders rounded to `{dcfg.rounding}` unit. "
        "Costs are assumptions - see the sensitivity table.", "",
        summary.to_markdown(index=False, floatfmt=",.4g"), "",
        "## Saving of `lgbm_newsvendor` with 95% cluster-bootstrap CI (by item)", "",
        pd.DataFrame([{"vs policy": k, "saving_pct": v["ratio"], "ci_low_pct": v["ci_low"],
                       "ci_high_pct": v["ci_high"]} for k, v in ci.items()]).to_markdown(index=False, floatfmt=".2f"),
        "", "## Sensitivity to the cost assumptions", "",
        sens.pivot_table(index="critical_ratio", columns="policy", values="saving_vs_naive_pct")
        .reset_index().to_markdown(index=False, floatfmt=".2f"), "",
        "`service_level` = share of item-days without a stock-out; for the newsvendor policies it should be close "
        "to the critical ratio if the quantile forecasts are well calibrated.",
    ]
    (out / "decision.md").write_text("\n".join(md))
    headline = {"critical_ratio": cr, "summary": summary.to_dict(orient="records"), "bootstrap": ci}
    log.info("decision: lgbm_newsvendor saving vs naive = %.2f%% (95%% CI %.2f to %.2f)",
             ci[BASE_POLICY]["ratio"], ci[BASE_POLICY]["ci_low"], ci[BASE_POLICY]["ci_high"])
    return headline
