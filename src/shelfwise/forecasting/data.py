"""Read model-ready rows from the exported Parquet file."""

from __future__ import annotations

import pandas as pd
import pyarrow.dataset as ds

from shelfwise.config import Settings
from shelfwise.transform.export import cache_paths


def load_features(settings: Settings, d_lo: int, d_hi: int, columns: list[str] | None = None,
                  observed_only: bool = True) -> pd.DataFrame:
    """Feature rows with target day in [d_lo, d_hi].

    observed_only=True drops rows whose target is unknown (future rows). Training and
    backtesting must only ever see observed targets.
    """
    path = cache_paths(settings)["features"]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run the 'export' step first")
    filt = (ds.field("d_num") >= d_lo) & (ds.field("d_num") <= d_hi)
    if observed_only:
        filt = filt & ds.field("target").is_valid()
    table = ds.dataset(path, format="parquet").to_table(columns=columns, filter=filt)
    return table.to_pandas()
