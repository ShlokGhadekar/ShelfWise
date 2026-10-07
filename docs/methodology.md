# Methodology and design choices

This document explains why ShelfWise is built the way it is. The code comments repeat
the key points where they matter; this is the narrative version.

## 1. Problem framing

- **Unit of forecast:** daily unit sales for one item in one store (30,490 series).
- **Horizon:** 28 days ahead from a forecast origin T (the last observed day).
- **Output:** a mean forecast, the P10/P50/P90 quantiles, and an order quantity.
- **Data:** Walmart M5 (Kaggle). Days `d_1` = 2011-01-29 to `d_1941` = 2016-05-22, plus
  calendar/events/SNAP and weekly prices. Prices and the calendar extend to `d_1969`, so
  the "future" 28 days have known prices.

## 2. Warehouse layers

| Layer | Contents | Why |
|---|---|---|
| `raw` | CSVs loaded with COPY, only typed. Sales stay one row per series, with days in an `integer[]` | Rebuild anything downstream without re-reading files. The array works around Postgres's 1,600-column limit. |
| `staging` | `series` (dimension, integer keys and codes), `calendar`, `prices`, and `sales` in long format (one row per series per day), range-partitioned by year | Clean, conformed, and query-friendly. The reshape uses `unnest(... ) WITH ORDINALITY`, which is the SQL equivalent of `pandas.melt`. |
| `analytics` | `features`, `forecasts`, `model_metrics`, `decision_results`, `store_daily_sales` (materialized view) | What models, the API and the dashboard consume. |
| `meta` | watermarks, file registry, pipeline runs, data-quality results | Lineage and incremental state. |

Cleaning rules (all counted in the data-quality report, never silently applied):
negative sales become NULL; prices that are NULL or ≤ 0 are dropped (that week counts as
not on sale); duplicate price rows are deduplicated deterministically; days before an item
is first on sale are kept in staging but excluded from features and evaluation.

### Performance choices
- **Partitioning by year:** date-filtered queries touch one or two partitions. Partition
  pruning only works when the `WHERE` clause filters on `date`, so the SQL always adds a
  date predicate even when it also filters on `d_num`.
- **B-tree PK `(series_id, date)`:** single-series lookups (API, dashboard, exports) are
  index scans.
- **BRIN on `date`:** a tiny index that works because rows are inserted in date order, in
  28-day blocks.
- **Load first, index later:** a full refresh drops the PK, bulk-inserts, then rebuilds
  the index once.
- **Window functions on integers:** Postgres can slide `AVG`/`STDDEV` windows over integer
  columns in O(1) per row (it has "inverse transition" functions for them), but not over
  float columns.
- **Materialized view** for the store × day totals the dashboard needs.
- `make benchmark` measures all of this against an unpartitioned, unindexed copy of the
  same data using `EXPLAIN (ANALYZE, BUFFERS)`.

## 3. Leakage-safe features

**Rule:** for target day t, every sales-derived feature may only use sales up to day
t − 28. At forecast origin T we want days T+1 … T+28, and the latest known sales day is
T = t − h with h ≤ 28. A feature that is valid for every horizon therefore has to stop at
t − 28.

| Group | Features | Leakage status |
|---|---|---|
| Lags | `lag_28, lag_35, lag_42, lag_49, lag_56`, `rmean_dow_4w` (mean of the four same-weekday lags) | end at t−28 ✔ |
| Rolling | `rmean_7/14/28/56`, `rstd_7/28`, `rzero_28` (share of zero days), all windows ending at t−28 | ✔ |
| Intermittency | `dsls`: days since last sale as of t−28 (capped) | ✔ |
| Price | `sell_price`, `price_change_1w`, `price_rel_max_8w` (discount depth), `price_rel_mean_52w`, `is_discount` | uses prices up to t, which is allowed because prices are set by the retailer in advance (assumption) |
| Calendar | `wday, month, day_of_month, is_weekend, event_type_code, is_event, days_to_next_event, snap` | known in advance ✔ (`days_to_next_event` looks forward, which is valid only because the event calendar is published) |
| Item | `days_since_listed`, categorical codes for item/dept/cat/store/state | ✔ |

**Common beginner mistakes this avoids:**
1. *Rolling means that include yesterday.* They give great validation scores and are
   impossible to compute 20 days ahead.
2. *Random train/test splits.* The model sees the future.
3. *Pre-launch zeros treated as zero demand.* They are turned into NULL.
4. *ROWS windows over data with gaps.* `ROWS 28 PRECEDING` is 28 rows, not 28 days. A
   data-quality ERROR check guarantees exactly one row per series per day.
5. *Category codes built separately for train and test.* The codes are fixed integers
   from the warehouse.
6. *Target encodings computed on the full dataset.* Not used.

**How it is tested** (`tests/test_features_sql.py`, which runs the same SQL file on DuckDB):
- *Perturbation test:* add +50 to every sale from day P onwards. Features for all t ≤ P+27
  must not change at all, and `lag_28` at t = P+28 must change by exactly 50.
- Hand-computed lags, rolling means, standard deviations, zero shares and `dsls` match the SQL.
- Pre-launch rows are excluded, and rolling means ignore pre-launch days.
- *Incremental build = full build,* row for row.
- Future rows have a NULL target and valid lags.
- A DB integration test re-checks `lag_28` against staging on real data.

**Incremental correctness.** An incremental build re-reads `lookback_buffer_days` (180)
of history. Every window spans at most 84 days, and `dsls` is capped at 120 days, so a
180-day buffer (≥ 120 + 28) reproduces the full-rebuild values exactly. `config.py`
refuses settings that would break this.

## 4. Models

- **Baselines:** naive (last value), seasonal naive 7 and 28, 28-day moving average, an
  empirical-quantile baseline (quantiles of the last 56 days), and an all-zeros
  "forecast" (shown only to demonstrate why MAE misleads).
- **LightGBM, global:** one model across all series.
  - Point model: `tweedie` objective (variance power 1.1). It estimates the mean, which
    is the right target for squared-error metrics and the forecast that sums correctly
    to store/total level.
  - Quantile models: one `quantile` objective per level (0.1, 0.5, 0.9). Predictions are
    clipped at 0 and sorted per row to prevent quantile crossing.
  - Early stopping uses the last 28 days of each training period (time-based). The model
    is then refit on all training days with the best iteration count, because the most
    recent weeks are the most valuable data.
- **ETS / SARIMA** at total/state/store level, compared with the LightGBM mean forecasts
  summed bottom-up. Classical models are at their strongest on these smooth aggregate series.

## 5. Evaluation

**Walk-forward backtest:** 3 folds. Origins are d_1857, d_1885 and d_1913 when the last
day is 1941. Each fold trains on the 700 days before its origin and tests on the next 28.
Test windows never overlap, and the last one ends on the last observed day.

**Metrics:**
- RMSSE per series: MSE divided by the in-sample mean squared one-step naive difference,
  counted from the first sale. Averaged over series.
- WRMSSE: RMSSE at the 12 M5 aggregation levels, weighted by each series' dollar sales in
  the 28 days before the origin, then averaged over levels (the official M5 score).
- MASE, pinball loss per quantile, scaled pinball loss, P10–P90 coverage (nominal 80%),
  one-sided exceedance rates and interval width.
- RMSE/MAE in units and bias are shown for contrast.

**Why plain RMSE/MAE/MAPE/accuracy mislead here:** errors on fast movers dominate
unscaled metrics; MAPE is undefined at zero; MAE is minimised by the median, which is 0
for intermittent series, so "never order anything" looks good; and "accuracy" ignores
the size of the miss.

**Significance:**
- *Diebold-Mariano:* the daily mean scaled squared error of the best baseline minus that
  of LightGBM, across folds. It uses a Newey-West (HAC) variance with lag 7, the
  Harvey-Leybourne-Newbold correction and a t-distribution. With 84 daily observations it is conservative.
- *Cluster bootstrap:* resample whole items (each item across its stores and folds) to get
  a 95% CI for the mean per-series RMSSE improvement. Treating correlated series as
  independent would overstate certainty.
- *Wilcoxon signed-rank* is reported with that caveat.
- The same DM test compares quantile loss against the empirical-quantile baseline.

**Error analysis:** pooled RMSSE and bias by Syntetos–Boylan demand class (smooth,
erratic, intermittent, lumpy), item age, discount vs. regular price, event days, SNAP days,
horizon bucket and category, plus interval coverage by segment.

**SHAP:** TreeExplainer on 20k rows of the last fold for the point model, with LightGBM's
built-in TreeSHAP as a fallback. Values are in log space because of the tweedie log link.

## 6. Decision layer

**Newsvendor rule:** with underage cost c_u (lost margin per unit of unmet demand) and
overage cost c_o (cost per unsold unit), the expected-cost-minimising order is the demand
quantile at the critical ratio CR = c_u / (c_u + c_o). Quantiles between the trained
levels are linearly interpolated; outside P10–P90 they are clamped. The quantile is then
rounded to the **nearest** whole unit (`decision.rounding: nearest`). A unit test checks
the newsvendor rule against brute-force cost minimisation on simulated demand.

> **Disclosed post-hoc change.** The first backtest rounded the quantile **up**,
> following the textbook rule for discrete demand (order the smallest integer Q with
> F(Q) ≥ CR). That rule assumes you know the true discrete distribution. LightGBM
> instead gives a continuous estimate of a quantile that is really an integer. On the
> CA_1 run, 36.5% of item-days had an estimated P75 between 0 and 1 (e.g. 0.3), so
> rounding up ordered one unit almost every day for slow movers.
>
> The effect on that run (3 folds, critical ratio 0.75):
>
> | | Round up (original) | Nearest unit (current) |
> |---|---|---|
> | Share of item-days without a stock-out (target 0.75) | 0.917 | 0.876 |
> | Saving vs. naive ordering | 13.6% | 22.4% |
> | vs. LightGBM point-forecast orders | 4.7% **more** expensive | 6.0% cheaper |
> | vs. empirical-quantile newsvendor | 8.7% **more** expensive | 2.7% cheaper |
>
> The change was made **after** seeing test-window results, which is a mild form of
> tuning on the test set. Both quantile-based policies use the same rule, so the
> comparison between them stays fair. Setting `decision.rounding: up` reproduces the
> original numbers. Even after the fix, service level (0.876) is above target. This
> matches the over-wide prediction intervals (87% coverage vs. 80% nominal): the
> quantiles are not yet well calibrated (see "What I'd do next").

**Cost assumptions (`config.yaml → decision`), all as a fraction of the selling price:**

| Parameter | Default | Meaning |
|---|---|---|
| `gross_margin` | 0.30 | unit cost = 70% of price; a lost sale loses 30% of price |
| `goodwill_frac` | 0.00 | extra penalty per lost sale (0 = conservative) |
| `overage_frac` | 0.10 | cost per unsold unit per day (holding, markdown, waste) |
| → critical ratio | 0.75 | order the P75 of demand |

**Simulation modes:**
- `newsvendor` (default): single-period. Each day's stock is independent and leftovers are
  written off at c_o. This is exact for perishables and a common simplification otherwise.
- `carryover`: leftovers are carried and pay c_o per night. Each morning stock is topped up
  to the order-up-to level. Zero lead time, lost sales.

**Policies compared:** seasonal-naive orders (the naive approach), MA28 orders, an
empirical-quantile newsvendor, LightGBM point-forecast orders, and the LightGBM quantile
newsvendor. The comparisons separate *better forecast* from *better decision rule*. The
headline saving uses a ratio of totals with a cluster-bootstrap CI. A sensitivity
analysis sweeps CR from 0.5 to 0.9. Realised service level (share of item-days without a
stock-out) should be close to CR if the quantiles are calibrated.

**Caveat:** observed sales are censored demand, so stock-out costs are underestimated.
