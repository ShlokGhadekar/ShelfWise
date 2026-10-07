"""Query performance benchmark: before vs after optimisation, measured with EXPLAIN ANALYZE.

    python -m shelfwise.benchmark.run_benchmark            (or: make benchmark)
    python -m shelfwise.benchmark.run_benchmark --rebuild  (recreate the unoptimised copy)

"Before" = bench.sales_heap: the same rows as staging.sales in ONE plain table with NO
indexes - what you get from a naive `CREATE TABLE AS` / pandas `to_sql`.
"After"  = staging.sales: range-partitioned by year + B-tree primary key (series_id, date)
           + BRIN on date; plus a materialized view for the dashboard aggregate.

Method (so the numbers are honest):
  * both tables hold identical rows and are VACUUM ANALYZEd (fresh statistics)
  * each query runs once as a warm-up (results are reported for a warm cache - the
    realistic case for an API; cold-cache numbers would need a Postgres restart)
  * then `repeats` timed runs with EXPLAIN (ANALYZE, BUFFERS); we report the median of
    Postgres' own "Execution Time" (excludes network/Python overhead)
  * the full plans are saved, so you can show *why* it got faster (Seq Scan -> Index Scan,
    partitions pruned, buffers read)
Results: reports/benchmarks/benchmark_results.csv + .md + plans/
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import logging
import statistics

import pandas as pd

from shelfwise.config import get_settings
from shelfwise.db import connect, execute, get_watermark, scalar
from shelfwise.sqlrender import render
from shelfwise.utils.logs import setup_logging

log = logging.getLogger("shelfwise.benchmark")


@dataclasses.dataclass
class Case:
    name: str
    description: str
    variants: dict[str, str]     # variant name -> SQL (first = baseline)


HEAP, PART = "bench.sales_heap", "staging.sales"


def build_cases() -> list[Case]:
    def both(sql: str) -> dict[str, str]:
        return {"before: heap, no index": sql.replace("{t}", HEAP),
                "after: partitioned + indexed": sql.replace("{t}", PART)}

    return [
        Case("series_history", "API pattern: one series' last 365 days",
             both("SELECT date, sales, sell_price FROM {t} WHERE series_id = {sid} "
                  "AND date >= DATE '{d365}' ORDER BY date")),
        Case("single_day_snapshot", "All series on the latest day",
             both("SELECT COUNT(*), SUM(sales) FROM {t} WHERE date = DATE '{dlast}'")),
        Case("store_totals_last_28d", "Dashboard: daily totals per store, last 28 days",
             both("SELECT sr.store_id, s.date, SUM(s.sales) FROM {t} s JOIN staging.series sr USING (series_id) "
                  "WHERE s.date >= DATE '{d28}' GROUP BY sr.store_id, s.date")
             | {"after: materialized view": "SELECT store_id, date, units FROM analytics.store_daily_sales "
                                            "WHERE date >= DATE '{d28}'"}),
        Case("rolling_mean_last_90d", "Feature-style window function over the last 90 days",
             both("SELECT COUNT(*), AVG(rm7) FROM (SELECT AVG(sales) OVER (PARTITION BY series_id ORDER BY date "
                  "ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS rm7 FROM {t} WHERE date >= DATE '{d90}') x")),
        Case("dept_monthly_revenue_1y", "Analyst query: monthly revenue by department, last year",
             both("SELECT sr.dept_id, date_trunc('month', s.date) AS m, SUM(s.sales * s.sell_price) "
                  "FROM {t} s JOIN staging.series sr USING (series_id) WHERE s.date >= DATE '{d365}' "
                  "GROUP BY 1, 2")),
    ]


def ensure_heap(rebuild: bool) -> None:
    with connect(autocommit=True) as conn:
        exists = scalar(conn, "SELECT to_regclass('bench.sales_heap') IS NOT NULL")
        n_part = scalar(conn, "SELECT COUNT(*) FROM staging.sales")
        n_heap = scalar(conn, "SELECT COUNT(*) FROM bench.sales_heap") if exists else -1
        if rebuild or not exists or n_heap != n_part:
            log.info("building bench.sales_heap (%s rows, unpartitioned, no indexes)...", f"{n_part:,}")
            execute(conn, "DROP TABLE IF EXISTS bench.sales_heap")
            execute(conn, "CREATE TABLE bench.sales_heap AS SELECT * FROM staging.sales")
        execute(conn, "VACUUM ANALYZE bench.sales_heap")
        execute(conn, "VACUUM ANALYZE staging.sales")
        execute(conn, "REFRESH MATERIALIZED VIEW analytics.store_daily_sales")
        execute(conn, "ANALYZE analytics.store_daily_sales")


def explain(conn, sql: str) -> dict:
    with conn.cursor() as cur:
        cur.execute(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {sql}")
        plan = cur.fetchone()[0]
    if isinstance(plan, str):
        plan = json.loads(plan)
    root = plan[0]
    return {"execution_ms": root["Execution Time"], "planning_ms": root["Planning Time"],
            "shared_hit": root["Plan"].get("Shared Hit Blocks", 0),
            "shared_read": root["Plan"].get("Shared Read Blocks", 0), "plan": plan}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--repeats", type=int, default=None)
    args = ap.parse_args(argv)
    settings = get_settings()
    setup_logging(settings.paths.logs_dir)
    repeats = args.repeats or settings.benchmark.repeats
    out = settings.paths.reports_dir / "benchmarks"
    (out / "plans").mkdir(parents=True, exist_ok=True)

    ensure_heap(args.rebuild)
    with connect(autocommit=True) as conn:
        last_day, _ = get_watermark(conn, "staging_sales_max_day")
        dlast = scalar(conn, "SELECT date FROM staging.calendar WHERE d_num = %s", (last_day,))
        sid = scalar(conn, "SELECT series_id FROM staging.series ORDER BY series_id "
                           "OFFSET (SELECT COUNT(*) / 2 FROM staging.series) LIMIT 1")
        params = {"sid": int(sid), "dlast": dlast, "d28": dlast - dt.timedelta(days=27),
                  "d90": dlast - dt.timedelta(days=89), "d365": dlast - dt.timedelta(days=364)}
        sizes = {
            "rows": scalar(conn, "SELECT COUNT(*) FROM staging.sales"),
            "heap_total_mb": scalar(conn, "SELECT pg_total_relation_size('bench.sales_heap') / 1e6"),
            "partitioned_total_mb": scalar(conn, """SELECT SUM(pg_total_relation_size(inhrelid)) / 1e6
                                                    FROM pg_inherits WHERE inhparent = 'staging.sales'::regclass"""),
            "partitioned_index_mb": scalar(conn, """SELECT SUM(pg_indexes_size(inhrelid)) / 1e6
                                                    FROM pg_inherits WHERE inhparent = 'staging.sales'::regclass"""),
        }

        rows = []
        for case in build_cases():
            for variant, sql_t in case.variants.items():
                sql = render(sql_t, **params)
                explain(conn, sql)                                  # warm-up
                runs = [explain(conn, sql) for _ in range(repeats)]
                ex = [r["execution_ms"] for r in runs]
                rows.append({
                    "case": case.name, "description": case.description, "variant": variant,
                    "median_ms": statistics.median(ex), "min_ms": min(ex), "max_ms": max(ex),
                    "planning_ms": statistics.median(r["planning_ms"] for r in runs),
                    "shared_hit_blocks": runs[-1]["shared_hit"], "shared_read_blocks": runs[-1]["shared_read"],
                })
                slug = f"{case.name}__{variant.split(':')[0]}_{variant.split(':')[-1].strip().replace(' ', '_')}"
                (out / "plans" / f"{slug}.json").write_text(json.dumps(runs[-1]["plan"], indent=2, default=str))
                with conn.cursor() as cur:
                    cur.execute(f"EXPLAIN (ANALYZE, BUFFERS) {sql}")
                    (out / "plans" / f"{slug}.txt").write_text("\n".join(r[0] for r in cur.fetchall()))
                log.info("%-24s %-32s median %9.2f ms", case.name, variant, rows[-1]["median_ms"])

    df = pd.DataFrame(rows)
    base = df.groupby("case")["median_ms"].transform("first")
    df["speedup_vs_before"] = base / df["median_ms"]
    df.insert(0, "timestamp", dt.datetime.now().isoformat(timespec="seconds"))
    df.insert(1, "subset", settings.subset.signature)
    df.to_csv(out / "benchmark_results.csv", index=False)
    md = ["# Query benchmark (auto-generated)", "",
          f"Subset `{settings.subset.signature}`, {sizes['rows']:,} rows in staging.sales; median of {repeats} warm "
          "runs of EXPLAIN ANALYZE execution time.", "",
          f"Storage: heap {sizes['heap_total_mb']:.0f} MB | partitioned {sizes['partitioned_total_mb']:.0f} MB "
          f"(of which indexes {sizes['partitioned_index_mb']:.0f} MB).", "",
          df[["case", "variant", "median_ms", "min_ms", "max_ms", "speedup_vs_before", "shared_read_blocks"]]
          .to_markdown(index=False, floatfmt=".2f"), "",
          "Plans for every run are in `reports/benchmarks/plans/` (look for Seq Scan vs Index Scan and the number "
          "of partitions scanned)."]
    (out / "benchmark_results.md").write_text("\n".join(md))
    log.info("benchmark written to %s", out / "benchmark_results.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
