"""Logging + step timing.

Every pipeline step is wrapped in `step_timer`, which logs the duration and appends
it to reports/run_timings.csv. Those real timings are what you should quote in the
README ("full pipeline runs in X min on a laptop"), not guesses.
"""

from __future__ import annotations

import csv
import datetime as dt
import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_CONFIGURED = False


def setup_logging(logs_dir: Path, level: int = logging.INFO) -> Path:
    global _CONFIGURED
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"pipeline_{dt.datetime.now():%Y%m%d_%H%M%S}.log"
    if not _CONFIGURED:
        fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S")
        root = logging.getLogger()
        root.setLevel(level)
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(fmt)
        fileh = logging.FileHandler(log_file)
        fileh.setFormatter(fmt)
        root.addHandler(console)
        root.addHandler(fileh)
        for noisy in ("matplotlib", "PIL", "urllib3", "numba", "shap"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
        _CONFIGURED = True
    return log_file


@contextmanager
def step_timer(name: str, timings_file: Path | None = None, **context: object) -> Iterator[None]:
    log = logging.getLogger("shelfwise.timer")
    log.info("▶ %s started", name)
    t0 = time.perf_counter()
    status = "ok"
    try:
        yield
    except Exception:
        status = "failed"
        raise
    finally:
        secs = time.perf_counter() - t0
        log.info("■ %s %s in %.1fs (%.1f min)", name, status, secs, secs / 60)
        if timings_file is not None:
            new = not timings_file.exists()
            timings_file.parent.mkdir(parents=True, exist_ok=True)
            with open(timings_file, "a", newline="") as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(["timestamp", "step", "status", "seconds", *context.keys()])
                w.writerow([dt.datetime.now().isoformat(timespec="seconds"), name, status, f"{secs:.1f}",
                            *context.values()])
