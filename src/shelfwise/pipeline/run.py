"""One command for the whole pipeline.

    python -m shelfwise.pipeline.run                         # everything, incremental where possible
    python -m shelfwise.pipeline.run --steps load,staging,quality,features,export
    python -m shelfwise.pipeline.run --last-day 1913         # pretend "today" is d_1913
    python -m shelfwise.pipeline.run --full-refresh          # rebuild staging + features from scratch

Steps (in order):
  init      create schemas/tables (idempotent)
  load      CSV -> raw (skipped per file if unchanged)
  staging   raw -> staging (incremental: only new days)
  quality   data-quality checks (stops on ERROR)
  features  staging -> analytics.features with SQL window functions (incremental)
  export    analytics.features -> data/cache/*.parquet / .npz
  backtest  walk-forward backtest: baselines + LightGBM point & quantile models
  evaluate  metrics, significance tests, error analysis, SHAP, ETS/SARIMA comparison
  decide    newsvendor ordering simulation on the backtest windows
  forecast  retrain on all data, write the next-28-day forecast + orders to analytics.forecasts
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Callable

from shelfwise.config import get_settings

STEP_ORDER = ["init", "load", "staging", "quality", "features", "export", "backtest", "evaluate", "decide",
              "forecast"]
log = logging.getLogger("shelfwise.pipeline")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ShelfWise end-to-end pipeline")
    p.add_argument("--steps", default="all", help=f"comma list or 'all'. Steps: {', '.join(STEP_ORDER)}")
    p.add_argument("--full-refresh", action="store_true", help="rebuild staging and features from scratch")
    p.add_argument("--force-load", action="store_true", help="reload raw files even if unchanged")
    p.add_argument("--last-day", type=int, default=None, help="override LAST_DAY (simulate 'today')")
    p.add_argument("--subset", default=None, help="override SUBSET, e.g. CA, CA_1, ALL")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    # CLI overrides go through the environment so get_settings() sees them everywhere.
    if args.last_day is not None:
        os.environ["LAST_DAY"] = str(args.last_day)
    if args.subset is not None:
        os.environ["SUBSET"] = args.subset
    get_settings.cache_clear()
    settings = get_settings()

    from shelfwise.utils.logs import setup_logging, step_timer

    log_file = setup_logging(settings.paths.logs_dir)
    steps = STEP_ORDER if args.steps == "all" else [s.strip() for s in args.steps.split(",") if s.strip()]
    unknown = set(steps) - set(STEP_ORDER)
    if unknown:
        log.error("Unknown steps: %s", ", ".join(sorted(unknown)))
        return 2
    steps = [s for s in STEP_ORDER if s in steps]    # always run in dependency order
    log.info("ShelfWise pipeline | subset=%s | last_day=%s | steps=%s | log=%s",
             settings.subset.signature, settings.data.last_day or "file max", ",".join(steps), log_file)

    # Imports are lazy so e.g. `--steps load` doesn't need LightGBM installed/working.
    def init(_):
        from shelfwise.db import connect, run_sql_file
        with connect() as conn:
            for f in ("ddl/00_schemas.sql", "ddl/01_raw.sql", "ddl/02_staging.sql", "ddl/03_analytics.sql"):
                run_sql_file(conn, f)

    def load(_):
        from shelfwise.ingest.load_raw import load_raw
        load_raw(settings, force=args.force_load)

    def staging(_):
        from shelfwise.transform.staging import build_staging
        log.info("staging result: %s", build_staging(settings, full_refresh=args.full_refresh))

    def quality(run_id):
        from shelfwise.quality.checks import run_checks
        run_checks(settings, run_id=run_id)

    def features(_):
        from shelfwise.transform.features import build_features
        log.info("features result: %s", build_features(settings, full_refresh=args.full_refresh))

    def export(_):
        from shelfwise.transform.export import export_all
        export_all(settings, batch_series=settings.features.series_batch_size)

    def backtest(_):
        from shelfwise.forecasting.backtest import run_backtest
        run_backtest(settings)

    def evaluate(_):
        from shelfwise.evaluation.evaluate import run_evaluation
        log.info("headline: %s", run_evaluation(settings))

    def decide(_):
        from shelfwise.decision.simulate import run_decision
        run_decision(settings)

    def forecast(_):
        from shelfwise.forecasting.forecast import run_forecast
        run_forecast(settings)

    actions: dict[str, Callable] = {"init": init, "load": load, "staging": staging, "quality": quality,
                                    "features": features, "export": export, "backtest": backtest,
                                    "evaluate": evaluate, "decide": decide, "forecast": forecast}

    run_id = _start_run(settings, steps)
    timings = settings.paths.reports_dir / "run_timings.csv"
    try:
        for step in steps:
            with step_timer(step, timings, subset=settings.subset.signature,
                            last_day=settings.data.last_day or "max"):
                actions[step](run_id)
    except Exception as exc:
        log.exception("Pipeline failed in a step: %s", exc)
        _finish_run(run_id, "failed", str(exc)[:500])
        return 1
    _finish_run(run_id, "success", None)
    log.info("Pipeline finished OK. Step timings: %s", timings)
    return 0


def _start_run(settings, steps) -> int | None:
    try:
        from shelfwise.db import connect, run_sql_file, scalar
        with connect() as conn:
            run_sql_file(conn, "ddl/00_schemas.sql")
            return scalar(conn, "INSERT INTO meta.pipeline_runs (steps, subset, last_day) VALUES (%s, %s, %s) "
                                "RETURNING run_id", (",".join(steps), settings.subset.signature,
                                                     settings.data.last_day))
    except Exception as exc:
        log.error("Cannot connect to Postgres (%s). Is the container up? Try: make db-up", exc)
        raise SystemExit(1) from exc


def _finish_run(run_id, status, message) -> None:
    if run_id is None:
        return
    from shelfwise.db import connect
    try:
        with connect() as conn, conn.cursor() as cur:
            cur.execute("UPDATE meta.pipeline_runs SET finished_at = now(), status = %s, message = %s "
                        "WHERE run_id = %s", (status, message, run_id))
    except Exception:
        log.warning("could not record run status")


if __name__ == "__main__":
    sys.exit(main())
