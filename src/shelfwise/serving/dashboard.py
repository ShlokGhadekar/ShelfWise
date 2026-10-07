"""Streamlit dashboard.  Run:  make dashboard  (http://localhost:8501)

Tabs: item forecast | model comparison | inventory decisions | data quality
Reads Postgres through the same query layer as the API, plus the generated report files.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from shelfwise.config import get_settings
from shelfwise.serving import queries

BLUE, ORANGE, GRAY, INK, BAND = "#2a78d6", "#eb6834", "#898781", "#0b0b0b", "rgba(42,120,214,0.18)"

st.set_page_config(page_title="ShelfWise", layout="wide")
settings = get_settings()


def _layout(fig: go.Figure, title: str, ytitle: str) -> go.Figure:
    fig.update_layout(title=title, height=420, margin=dict(l=10, r=10, t=50, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=1.02, x=0, yanchor="bottom"), yaxis_title=ytitle,
                      template="plotly_white")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#e1e0d9")
    return fig


@st.cache_data(ttl=300)
def _stores():
    return queries.stores()


@st.cache_data(ttl=300)
def _items(store):
    return queries.items(store)


st.title("ShelfWise - demand forecasts & order recommendations")
st.caption(f"Subset `{settings.subset.signature}` | decision mode `{settings.decision.mode}` | "
           f"critical ratio {settings.decision.critical_ratio:.2f} (assumed costs, see README)")

try:
    store_list = _stores()
except Exception as exc:
    st.error(f"Cannot reach the database: {exc}. Is `make db-up` running and has the pipeline been run?")
    st.stop()

tab_fc, tab_models, tab_dec, tab_dq = st.tabs(["Item forecast", "Model comparison", "Inventory decisions",
                                                "Data quality"])

# ------------------------------------------------------------------ item forecast
with tab_fc:
    c1, c2, c3 = st.columns([1, 1, 2])
    store = c1.selectbox("Store", store_list)
    items = _items(store)
    dept = c2.selectbox("Department", sorted(items["dept_id"].unique()))
    item = c3.selectbox("Item", items[items["dept_id"] == dept]["item_id"])
    hist_days = st.slider("Days of history", 28, 365, 112, step=7)

    series = queries.find_series(item, store)
    hist = queries.history(int(series["series_id"]), hist_days)
    fc = queries.forecast(int(series["series_id"]))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["date"], y=hist["sales"], name="actual sales", mode="lines",
                             line=dict(color=GRAY, width=1.5)))
    if not fc.empty:
        fig.add_trace(go.Scatter(x=fc["date"], y=fc["p90"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=fc["date"], y=fc["p10"], fill="tonexty", fillcolor=BAND, line=dict(width=0),
                                 name="P10-P90 interval"))
        fig.add_trace(go.Scatter(x=fc["date"], y=fc["point"], name="forecast (mean)", line=dict(color=BLUE, width=2)))
        fig.add_trace(go.Scatter(x=fc["date"], y=fc["baseline_snaive7"], name="naive (last week)",
                                 line=dict(color=ORANGE, width=1.5)))
    st.plotly_chart(_layout(fig, f"{item} @ {store}", "units / day"), use_container_width=True)

    if fc.empty:
        st.info("No forecast stored for this series yet - run `make forecast`.")
    else:
        k1, k2, k3 = st.columns(3)
        k1.metric("Forecast, next 7 days", f"{fc['point'].head(7).sum():.1f} units")
        k2.metric("Recommended orders, next 7 days", f"{int(fc['order_qty'].head(7).sum())} units")
        k3.metric("Model version", fc["model_version"].iloc[0])
        show = fc[["date", "point", "p10", "p50", "p90", "baseline_snaive7", "order_qty"]].rename(columns={
            "point": "forecast", "baseline_snaive7": "naive", "order_qty": "recommended order"})
        st.dataframe(show.round(2), use_container_width=True, hide_index=True)

# ------------------------------------------------------------------ model comparison
with tab_models:
    metrics = queries.model_metrics("all")
    if metrics.empty:
        st.info("No backtest metrics yet - run `make backtest evaluate`.")
    else:
        metrics = metrics.sort_values("rmsse_mean")
        st.markdown("**Lower is better.** RMSSE / WRMSSE are scale-free (M5 metrics). "
                    "MAE in units is shown to illustrate why it misleads: the all-zeros forecast scores well on it.")
        colors = [BLUE if m == "lgbm_point" else GRAY for m in metrics["model"]]
        fig = go.Figure(go.Bar(x=metrics["rmsse_mean"], y=metrics["model"], orientation="h", marker_color=colors,
                               hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(_layout(fig, "Mean RMSSE across backtest folds", ""), use_container_width=True)
        cols = [c for c in ["model", "rmsse_mean", "wrmsse", "mase_mean", "bias_pct", "mae_units", "rmse_units"]
                if c in metrics]
        st.dataframe(metrics[cols].round(4), use_container_width=True, hide_index=True)
    figs = settings.paths.reports_dir / "figures"
    for name, caption in [("error_by_horizon.png", "Error by horizon"), ("error_by_segment.png", "Error by segment"),
                          ("shap_importance.png", "SHAP feature importance")]:
        if (figs / name).exists():
            st.image(str(figs / name), caption=caption)

# ------------------------------------------------------------------ decisions
with tab_dec:
    dec = queries.decision_results("all")
    if dec.empty:
        st.info("No decision results yet - run `make evaluate`.")
    else:
        dec = dec.sort_values("total_cost")
        colors = [BLUE if p == "lgbm_newsvendor" else ORANGE if p == "naive_snaive7" else GRAY for p in dec["policy"]]
        fig = go.Figure(go.Bar(x=dec["total_cost"], y=dec["policy"], orientation="h", marker_color=colors,
                               hovertemplate="%{y}: $%{x:,.0f}<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(_layout(fig, "Simulated holding + stock-out cost over the backtest windows", ""),
                        use_container_width=True)
        cols = [c for c in ["policy", "total_cost", "saving_vs_naive_pct", "fill_rate", "service_level",
                            "stockout_cost", "overage_cost"] if c in dec]
        st.dataframe(dec[cols].round(3), use_container_width=True, hide_index=True)
        st.caption("Costs rest on assumed margins and holding costs (config.yaml -> decision). "
                   "See reports/results/decision.md for the sensitivity analysis.")

# ------------------------------------------------------------------ data quality
with tab_dq:
    dq = queries.latest_dq()
    if dq.empty:
        st.info("No data-quality results yet - run the pipeline.")
    else:
        failed = dq[dq["status"] == "FAIL"]
        st.metric("Checks failed", f"{len(failed)} / {len(dq)}")
        st.dataframe(pd.DataFrame(dq), use_container_width=True, hide_index=True)
