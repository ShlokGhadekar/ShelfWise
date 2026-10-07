"""Evaluate the backtest: metrics, significance tests, error analysis, SHAP, ETS/SARIMA.

Reads data/cache/backtest_predictions.parquet (written by the backtest step), so it can
be re-run in minutes without re-training. Everything it reports is computed from real
backtest output - reports/results/results.md is the source for the README tables.
"""

from __future__ import annotations

import datetime as dt
import json
import logging

import numpy as np
import pandas as pd

from shelfwise.config import Settings
from shelfwise.db import connect, copy_df_to_table, execute
from shelfwise.evaluation import plots
from shelfwise.evaluation.aggregate_models import compare_aggregate
from shelfwise.evaluation.error_analysis import add_segments, demand_class, interval_by_segment, segment_table
from shelfwise.evaluation.explain import compute_shap
from shelfwise.evaluation.metrics import mase, naive_scale, pinball, rmsse, wrmsse_by_level
from shelfwise.evaluation.stats_tests import cluster_bootstrap_mean, diebold_mariano, wilcoxon_signed_rank
from shelfwise.forecasting.backtest import predictions_path
from shelfwise.forecasting.lgbm_model import qname
from shelfwise.transform.export import load_matrix, load_series

log = logging.getLogger(__name__)

MAIN_MODEL = "lgbm_point"
BASELINES = ["naive", "snaive7", "snaive28", "ma28"]
POINT_MODELS = ["lgbm_point", "lgbm_q50", "naive", "snaive7", "snaive28", "ma28", "emp_q50", "zeros"]


def _md(df: pd.DataFrame, fmt: str = ".4f") -> str:
    return df.to_markdown(index=False, floatfmt=fmt)


def _dense(f: pd.DataFrame, col: str, n: int, H: int) -> np.ndarray:
    M = np.full((n, H), np.nan)
    M[f["row"].to_numpy(), f["h"].to_numpy() - 1] = f[col].to_numpy()
    return M


def point_metrics(preds: pd.DataFrame, Y: np.ndarray, prices: np.ndarray, meta: pd.DataFrame, H: int,
                  models: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Adds scale/segment columns to `preds` in place; returns per-series, WRMSSE and summary tables."""
    n = len(meta)
    sids = meta["series_id"].to_numpy()
    preds["scale_sq"] = np.nan
    preds["scale_abs"] = np.nan
    preds["seg_demand"] = "no_sales"
    per_series, wr = [], []
    for k, idx in preds.groupby("fold").groups.items():
        f = preds.loc[idx]
        origin = int(f["origin"].iloc[0])
        hist = Y[:, :origin]                          # ONLY data up to the forecast origin
        s2, s1 = naive_scale(hist, 2), naive_scale(hist, 1)
        r = f["row"].to_numpy()
        preds.loc[idx, "scale_sq"] = s2[r]
        preds.loc[idx, "scale_abs"] = s1[r]
        preds.loc[idx, "seg_demand"] = demand_class(hist)[r]

        A = _dense(f, "actual", n, H)
        dense = {m: _dense(f, m, n, H) for m in models}
        evaluated = ~np.all(np.isnan(A), axis=1)
        for m, F in dense.items():
            per_series.append(pd.DataFrame({
                "fold": k, "series_id": sids[evaluated], "model": m,
                "rmsse": rmsse(A, F, s2)[evaluated], "mase": mase(A, F, s1)[evaluated],
            }))
        dollars = np.nansum(Y[:, origin - 28:origin] * prices[:, origin - 28:origin], axis=1)
        w = wrmsse_by_level(meta, hist, A, dense, dollars)
        w["fold"] = k
        wr.append(w)
        log.info("evaluate: fold %s point metrics done", k)

    per_series = pd.concat(per_series, ignore_index=True)
    wr = pd.concat(wr, ignore_index=True)

    rows = []
    for (k, m), g in per_series.groupby(["fold", "model"]):
        p = preds[preds["fold"] == k]
        err = p["actual"] - p[m]
        rows.append({
            "fold": k, "model": m,
            "rmsse_mean": float(np.nanmean(g["rmsse"].replace(np.inf, np.nan))),
            "wrmsse": float(wr[(wr["fold"] == k) & (wr["model"] == m)]["score"].mean()),
            "mase_mean": float(np.nanmean(g["mase"].replace(np.inf, np.nan))),
            "rmse_units": float(np.sqrt(np.mean(err ** 2))),
            "mae_units": float(np.mean(np.abs(err))),
            "bias_pct": float(100 * (p[m].sum() - p["actual"].sum()) / p["actual"].sum()),
            "n_series_scored": int(g["rmsse"].notna().sum()),
        })
    by_fold = pd.DataFrame(rows)
    return per_series, wr, by_fold


def quantile_metrics(preds: pd.DataFrame, quantiles: list[float]) -> pd.DataFrame:
    rows = []
    lo, hi = qname(min(quantiles)), qname(max(quantiles))
    ok = preds["scale_abs"].notna()
    for prefix, label in (("lgbm", "lightgbm_quantile"), ("emp", "empirical_quantile_baseline")):
        for fold, p in list(preds.groupby("fold")) + [("all", preds)]:
            y = p["actual"].to_numpy()
            row = {"model": label, "fold": str(fold)}
            spl = []
            for q in quantiles:
                loss = pinball(y, p[f"{prefix}_{qname(q)}"].to_numpy(), q)
                row[f"pinball_{qname(q)}"] = float(loss.mean())
                sel = ok.loc[p.index].to_numpy()
                spl.append(loss[sel] / p["scale_abs"].to_numpy()[sel])
            row["scaled_pinball_mean"] = float(np.mean(np.concatenate(spl)))
            L, U = p[f"{prefix}_{lo}"].to_numpy(), p[f"{prefix}_{hi}"].to_numpy()
            row[f"coverage_{lo}_{hi}"] = float(np.mean((y >= L) & (y <= U)))
            row["nominal_coverage"] = max(quantiles) - min(quantiles)
            row[f"share_below_{lo}"] = float(np.mean(y < L))
            row[f"share_above_{hi}"] = float(np.mean(y > U))
            row["mean_interval_width"] = float(np.mean(U - L))
            rows.append(row)
    return pd.DataFrame(rows)


def significance(preds: pd.DataFrame, per_series: pd.DataFrame, meta: pd.DataFrame, best_base: str,
                 quantiles: list[float], settings: Settings) -> dict:
    ev = settings.evaluation
    ok = preds["scale_sq"].notna()
    p = preds[ok]
    # Daily mean scaled squared error for each model -> one loss series per model
    daily = p.assign(
        l_base=(p["actual"] - p[best_base]) ** 2 / p["scale_sq"],
        l_lgbm=(p["actual"] - p[MAIN_MODEL]) ** 2 / p["scale_sq"],
    ).groupby(["fold", "d_num"])[["l_base", "l_lgbm"]].mean().sort_index()

    res: dict = {"comparison": f"{MAIN_MODEL} vs best baseline ({best_base})"}
    res["diebold_mariano_all_folds"] = diebold_mariano(daily["l_base"], daily["l_lgbm"], ev.dm_hac_lag)
    res["diebold_mariano_by_fold"] = {
        str(k): diebold_mariano(g["l_base"], g["l_lgbm"], ev.dm_hac_lag) for k, g in daily.groupby(level=0)}

    wide = per_series.pivot_table(index=["fold", "series_id"], columns="model", values="rmsse")
    wide = wide.replace(np.inf, np.nan).dropna(subset=[best_base, MAIN_MODEL])
    diff = (wide[best_base] - wide[MAIN_MODEL]).to_numpy()       # > 0 means LightGBM is better
    sid = wide.index.get_level_values("series_id").to_numpy()
    lookup = meta.set_index("series_id")
    if ev.bootstrap_cluster == "item":
        clusters = lookup.loc[sid, "item_id"].to_numpy()
    elif ev.bootstrap_cluster == "store_dept":
        clusters = (lookup.loc[sid, "store_id"] + "|" + lookup.loc[sid, "dept_id"]).to_numpy()
    else:
        clusters = sid
    res["rmsse_difference_bootstrap"] = cluster_bootstrap_mean(diff, clusters, ev.bootstrap_reps) | {
        "cluster": ev.bootstrap_cluster, "meaning": "baseline RMSSE minus LightGBM RMSSE; > 0 = LightGBM better"}
    res["wilcoxon_signed_rank"] = wilcoxon_signed_rank(diff) | {
        "caveat": "assumes independent series - optimistic here; prefer the bootstrap / DM results"}
    res["win_rate_series"] = float(np.mean(diff > 0))
    base_mean, lgbm_mean = float(np.mean(wide[best_base])), float(np.mean(wide[MAIN_MODEL]))
    res["mean_rmsse"] = {"baseline": base_mean, "lightgbm": lgbm_mean,
                         "relative_improvement_pct": 100 * (base_mean - lgbm_mean) / base_mean}

    # Quantile forecasts vs the empirical-quantile baseline, on daily mean scaled pinball loss
    okq = preds["scale_abs"].notna()
    pq = preds[okq]
    lq = {pref: sum(pinball(pq["actual"], pq[f"{pref}_{qname(q)}"], q) for q in quantiles) / pq["scale_abs"]
          for pref in ("lgbm", "emp")}
    dq = pd.DataFrame({"fold": pq["fold"], "d_num": pq["d_num"], "emp": lq["emp"], "lgbm": lq["lgbm"]}) \
        .groupby(["fold", "d_num"]).mean().sort_index()
    res["diebold_mariano_quantile_loss"] = diebold_mariano(dq["emp"], dq["lgbm"], ev.dm_hac_lag)
    return res


def example_figure(preds: pd.DataFrame, Y: np.ndarray, meta: pd.DataFrame, row_of: pd.Series,
                   settings: Settings) -> None:
    last = preds[preds["fold"] == preds["fold"].max()]
    top = last.groupby("series_id")["actual"].sum().sort_values(ascending=False)
    pick = [top.index[0], top.index[len(top) // 20], top.index[len(top) // 5]]   # fast, medium, slow mover
    hist, fc = {}, {}
    names = meta.set_index("series_id")["id"]
    for sid in pick:
        g = last[last["series_id"] == sid].sort_values("d_num")
        origin = int(g["origin"].iloc[0])
        r = int(row_of.loc[sid])
        days = np.arange(origin - 55, origin + 1)
        name = str(names.loc[sid]).replace("_evaluation", "")
        hist[name] = (days, Y[r, days - 1])
        lo, hi = qname(min(settings.lgbm.quantiles)), qname(max(settings.lgbm.quantiles))
        fc[name] = (g["d_num"].to_numpy(), g["actual"].to_numpy(), g[MAIN_MODEL].to_numpy(),
                    g[f"lgbm_{lo}"].to_numpy(), g[f"lgbm_{hi}"].to_numpy())
    plots.forecast_examples(hist, fc, settings.paths.reports_dir / "figures" / "forecast_examples.png",
                            "Last backtest fold: 56 days of history, then the 28-day forecast vs actual")


def run_evaluation(settings: Settings) -> dict:
    out = settings.paths.reports_dir / "results"
    figs = settings.paths.reports_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    H = settings.forecast.horizon
    quantiles = sorted(settings.lgbm.quantiles)

    preds = pd.read_parquet(predictions_path(settings))
    m = load_matrix(settings)
    Y, prices, sids = m["sales"], m["prices"], m["series_id"]
    meta = load_series(settings).set_index("series_id").loc[sids].reset_index()
    row_of = pd.Series(np.arange(len(sids)), index=sids)
    preds["row"] = row_of.loc[preds["series_id"].to_numpy()].to_numpy()
    models = [c for c in POINT_MODELS if c in preds.columns]

    # ---------------- point accuracy
    per_series, wr, by_fold = point_metrics(preds, Y, prices, meta, H, models)
    summary = by_fold.drop(columns="fold").groupby("model").mean().reset_index().sort_values("rmsse_mean")
    best_base = summary[summary["model"].isin(BASELINES)].iloc[0]["model"]
    wr_level = wr.groupby(["level", "model"])["score"].mean().unstack("model")
    wr_level = wr_level.loc[sorted(wr_level.index, key=lambda s: int(s.split("_")[0]))].reset_index()

    # ---------------- quantiles, significance
    qm = quantile_metrics(preds, quantiles)
    sig = significance(preds, per_series, meta, best_base, quantiles, settings)

    # ---------------- error analysis
    cat_names = meta.drop_duplicates("cat_code").set_index("cat_code")["cat_id"].to_dict()
    preds = add_segments(preds, cat_names)
    seg = segment_table(preds, [MAIN_MODEL, best_base])
    cov_seg = interval_by_segment(preds, f"lgbm_{qname(quantiles[0])}", f"lgbm_{qname(quantiles[-1])}")

    # ---------------- aggregate-level classical models
    agg = compare_aggregate(preds, Y, meta, settings.evaluation.aggregate_models,
                            settings.evaluation.aggregate_train_days, H)
    agg_summary = agg.groupby(["level", "model"])["rmsse"].mean().unstack("model").reset_index() \
        if not agg.empty else pd.DataFrame()

    # ---------------- explainability
    shap_imp = compute_shap(settings)

    # ---------------- save tables
    summary.to_csv(out / "metrics_summary.csv", index=False)
    by_fold.to_csv(out / "metrics_by_fold.csv", index=False)
    wr_level.to_csv(out / "wrmsse_by_level.csv", index=False)
    qm.to_csv(out / "quantile_metrics.csv", index=False)
    seg.to_csv(out / "error_by_segment.csv", index=False)
    cov_seg.to_csv(out / "interval_coverage_by_segment.csv", index=False)
    agg.to_csv(out / "aggregate_models.csv", index=False)
    (out / "significance_tests.json").write_text(json.dumps(sig, indent=2))
    per_series.to_parquet(settings.paths.cache_dir / "per_series_metrics.parquet", index=False)

    # ---------------- figures
    plots.bar_metric(summary[summary["model"] != "zeros"], "rmsse_mean", "model",
                     "Mean RMSSE by model (lower is better, averaged over folds)", "mean RMSSE",
                     figs / "model_comparison_rmsse.png", MAIN_MODEL, best_base)
    plots.bar_metric(summary, "mae_units", "model",
                     "MAE in units - note how well 'zeros' does: why MAE misleads here", "MAE (units/day)",
                     figs / "mae_trap.png", MAIN_MODEL, "zeros")
    pz = preds[preds["scale_sq"].notna()]
    hz = pd.concat([pd.DataFrame({"h": pz["h"], "model": mm, "e": (pz["actual"] - pz[mm]) ** 2 / pz["scale_sq"]})
                    for mm in (MAIN_MODEL, best_base)])
    hz = np.sqrt(hz.groupby(["h", "model"])["e"].mean()).reset_index(name="pooled_rmsse")
    plots.lines_by_x(hz, "h", "pooled_rmsse", "model", "Error by forecast horizon",
                     "days ahead", "pooled RMSSE", figs / "error_by_horizon.png", MAIN_MODEL, best_base)
    plots.grouped_segments(seg.rename(columns={"pooled_rmsse": "value"}), figs / "error_by_segment.png",
                           [MAIN_MODEL, best_base], "Pooled RMSSE by segment (lower is better)")
    example_figure(preds, Y, meta, row_of, settings)

    # ---------------- DB (for the dashboard / API)
    long = by_fold.melt(id_vars=["fold", "model"], var_name="metric", value_name="value")
    long_all = summary.melt(id_vars=["model"], var_name="metric", value_name="value").assign(fold="all")
    db_rows = pd.concat([long.assign(fold=long["fold"].astype(str)), long_all])[["model", "fold", "metric", "value"]]
    with connect() as conn:
        execute(conn, "TRUNCATE analytics.model_metrics")
        copy_df_to_table(conn, db_rows, "analytics.model_metrics")

    headline = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "subset": settings.subset.signature,
        "n_series_scored": int(per_series[per_series["model"] == MAIN_MODEL]["rmsse"].notna().sum()),
        "best_baseline": best_base,
        "rmsse": {r["model"]: r["rmsse_mean"] for _, r in summary.iterrows()},
        "wrmsse": {r["model"]: r["wrmsse"] for _, r in summary.iterrows()},
        "significance": {
            "dm_p_lgbm_better": sig["diebold_mariano_all_folds"]["p_b_better"],
            "bootstrap_ci": [sig["rmsse_difference_bootstrap"]["ci_low"], sig["rmsse_difference_bootstrap"]["ci_high"]],
            "relative_improvement_pct": sig["mean_rmsse"]["relative_improvement_pct"],
            "win_rate_series": sig["win_rate_series"],
        },
        "coverage": qm[(qm["fold"] == "all")].set_index("model")
        [f"coverage_{qname(quantiles[0])}_{qname(quantiles[-1])}"].to_dict(),
    }
    (out / "headline.json").write_text(json.dumps(headline, indent=2))

    write_results_md(settings, summary, by_fold, wr_level, qm, sig, seg, cov_seg, agg_summary, shap_imp, best_base)
    log.info("evaluate: results written to %s", out / "results.md")
    return headline


def write_results_md(settings, summary, by_fold, wr_level, qm, sig, seg, cov_seg, agg_summary, shap_imp,
                     best_base) -> None:
    out = settings.paths.reports_dir / "results"
    folds = json.loads((out / "backtest_folds.json").read_text()) if (out / "backtest_folds.json").exists() else []
    dm = sig["diebold_mariano_all_folds"]
    bs = sig["rmsse_difference_bootstrap"]
    lines = [
        "# ShelfWise - backtest results (auto-generated)", "",
        f"Generated {dt.datetime.now():%Y-%m-%d %H:%M} | subset `{settings.subset.signature}` | "
        f"{len(folds)} folds x {settings.forecast.horizon} days | best baseline: `{best_base}`", "",
        "## Folds", "",
        _md(pd.DataFrame([{k: f[k] for k in ("k", "train_start", "train_end", "test_start", "test_end",
                                             "n_train_rows", "n_test_rows")} for f in folds]), ".0f")
        if folds else "_fold metadata missing_", "",
        "## Point accuracy (mean over folds)", "",
        "`rmsse_mean`/`wrmsse`/`mase_mean` are the metrics to judge by. `rmse_units`/`mae_units` are shown only "
        "to illustrate why unscaled metrics mislead (see `zeros`).", "",
        _md(summary), "",
        "## Point accuracy by fold", "", _md(by_fold.sort_values(["fold", "rmsse_mean"])), "",
        "## WRMSSE by aggregation level (mean over folds)", "", _md(wr_level), "",
        "## Quantile forecasts", "", _md(qm), "",
        "## Is the improvement real?", "",
        f"- Comparison: {sig['comparison']}",
        f"- Mean RMSSE: baseline {sig['mean_rmsse']['baseline']:.4f} vs LightGBM {sig['mean_rmsse']['lightgbm']:.4f} "
        f"({sig['mean_rmsse']['relative_improvement_pct']:.1f}% relative improvement)",
        f"- Diebold-Mariano (daily scaled loss, {dm['n']} days, HAC lag {settings.evaluation.dm_hac_lag}): "
        f"stat {dm['dm_stat']:.2f}, one-sided p (LightGBM better) = {dm['p_b_better']:.4g}",
        f"- Cluster bootstrap ({bs['cluster']}, {bs['n_clusters']} clusters): mean RMSSE gain {bs['mean']:.4f}, "
        f"95% CI [{bs['ci_low']:.4f}, {bs['ci_high']:.4f}], share of resamples <= 0: {bs['p_mean_le_0']:.4f}",
        f"- LightGBM beats the baseline on {100 * sig['win_rate_series']:.1f}% of series-folds",
        f"- Quantile loss DM test vs empirical quantiles: one-sided p = "
        f"{sig['diebold_mariano_quantile_loss']['p_b_better']:.4g}", "",
        "Full details: `significance_tests.json`.", "",
        "## Error analysis by segment", "", _md(seg, ".3f"), "",
        "## Interval coverage by segment (LightGBM)", "", _md(cov_seg, ".3f"), "",
    ]
    if not agg_summary.empty:
        lines += ["## Aggregate levels: classical models vs LightGBM bottom-up (mean RMSSE)", "",
                  _md(agg_summary), ""]
    if shap_imp is not None:
        lines += ["## Top 15 features by mean |SHAP|", "", _md(shap_imp.head(15)), ""]
    lines += ["Figures: `reports/figures/`. Decision-layer results: `decision.md`."]
    (out / "results.md").write_text("\n".join(lines))
