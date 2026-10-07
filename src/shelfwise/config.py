"""Configuration: one YAML file + environment variables.

Design choice: anything secret or machine-specific (DB password, which subset you
are developing on, "today") comes from environment variables / `.env`, everything
methodological (feature windows, model params, cost assumptions) lives in
`config/config.yaml` so it is version-controlled and visible in code review.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The feature SQL never looks closer than 28 days to the target day (lag_28 is the
# shortest lag). That is what makes a single model valid for every horizon 1..28.
MIN_LAG = 28
# Longest look-back used by any feature: rmean_56 ends 28 days back -> needs 28 + 56 days.
MAX_WINDOW_LOOKBACK = MIN_LAG + 56

STATES = ("CA", "TX", "WI")
_STORE_RE = re.compile(r"^(CA|TX|WI)_\d{1,2}$")


class Paths(BaseModel):
    raw_dir: Path
    cache_dir: Path
    models_dir: Path
    reports_dir: Path
    logs_dir: Path

    def resolve(self) -> Paths:
        return Paths(**{k: (PROJECT_ROOT / v if not Path(v).is_absolute() else Path(v))
                        for k, v in self.model_dump().items()})


class Files(BaseModel):
    sales: str
    calendar: str
    prices: str


class DataCfg(BaseModel):
    last_day: int | None = None
    staging_chunk_days: int = 28


class FeaturesCfg(BaseModel):
    start_day: int
    lookback_buffer_days: int
    dsls_cap_days: int
    series_batch_size: int = 2000


class ForecastCfg(BaseModel):
    horizon: int = 28


class BacktestCfg(BaseModel):
    n_folds: int = 3
    step_days: int = 28


class LgbmCfg(BaseModel):
    train_window_days: int = 700
    max_train_rows: int | None = None
    num_boost_round: int = 2000
    early_stopping_rounds: int = 50
    refit_full: bool = True
    point_objective: str = "tweedie"
    tweedie_variance_power: float = 1.1
    quantiles: list[float] = Field(default_factory=lambda: [0.1, 0.5, 0.9])
    params: dict = Field(default_factory=dict)


class BaselinesCfg(BaseModel):
    empirical_window_days: int = 56


class EvaluationCfg(BaseModel):
    shap_sample_rows: int = 20000
    bootstrap_reps: int = 2000
    bootstrap_cluster: str = "item"
    dm_hac_lag: int = 7
    aggregate_models: list[str] = Field(default_factory=lambda: ["ets", "sarima"])
    aggregate_train_days: int = 728


class DecisionCfg(BaseModel):
    gross_margin: float = 0.30
    goodwill_frac: float = 0.0
    overage_frac: float = 0.10
    mode: str = "newsvendor"
    sensitivity_critical_ratios: list[float] = Field(default_factory=lambda: [0.5, 0.75, 0.9])

    @property
    def underage_frac(self) -> float:
        """Cost of one unit of unmet demand, as a fraction of price (lost margin + goodwill)."""
        return self.gross_margin + self.goodwill_frac

    @property
    def critical_ratio(self) -> float:
        cu, co = self.underage_frac, self.overage_frac
        return cu / (cu + co)


class QualityCfg(BaseModel):
    fail_on_error: bool = True
    price_jump_ratio: float = 3.0
    price_outlier_ratio: float = 5.0
    sample_rows: int = 10


class BenchmarkCfg(BaseModel):
    repeats: int = 5


class DbCfg(BaseModel):
    user: str
    password: str
    host: str
    port: int
    dbname: str

    @property
    def sqlalchemy_url(self) -> str:
        return f"postgresql+psycopg://{self.user}:{self.password}@{self.host}:{self.port}/{self.dbname}"

    @property
    def conninfo(self) -> str:
        return f"host={self.host} port={self.port} dbname={self.dbname} user={self.user} password={self.password}"


class Subset(BaseModel):
    """Which slice of M5 to process. `states`/`stores` empty -> everything."""

    raw: str
    states: list[str] = Field(default_factory=list)
    stores: list[str] = Field(default_factory=list)

    @property
    def is_all(self) -> bool:
        return not self.states and not self.stores

    @property
    def signature(self) -> str:
        """Canonical string stored next to watermarks, so we can detect a subset change."""
        return "ALL" if self.is_all else ",".join(sorted(self.states + self.stores))

    def sql_filter(self, alias: str = "w") -> str:
        """SQL predicate for raw.sales_wide. Safe to inline: tokens were validated by regex."""
        if self.is_all:
            return "TRUE"
        parts = []
        if self.states:
            parts.append(f"{alias}.state_id IN ({', '.join(repr(s) for s in self.states)})")
        if self.stores:
            parts.append(f"{alias}.store_id IN ({', '.join(repr(s) for s in self.stores)})")
        return "(" + " OR ".join(parts) + ")"


def parse_subset(value: str | None) -> Subset:
    value = (value or "ALL").strip().upper()
    if value in ("", "ALL", "FULL", "NONE"):
        return Subset(raw="ALL")
    states, stores = [], []
    for token in (t.strip() for t in value.split(",") if t.strip()):
        if token in STATES:
            states.append(token)
        elif _STORE_RE.match(token):
            stores.append(token)
        else:
            raise ValueError(f"Invalid SUBSET token {token!r}. Use ALL, CA/TX/WI, or a store like CA_1.")
    return Subset(raw=value, states=sorted(set(states)), stores=sorted(set(stores)))


class Settings(BaseModel):
    paths: Paths
    files: Files
    subset: Subset
    data: DataCfg
    features: FeaturesCfg
    forecast: ForecastCfg
    backtest: BacktestCfg
    lgbm: LgbmCfg
    baselines: BaselinesCfg
    evaluation: EvaluationCfg
    decision: DecisionCfg
    quality: QualityCfg
    benchmark: BenchmarkCfg
    db: DbCfg

    @model_validator(mode="after")
    def _check(self) -> Settings:
        h = self.forecast.horizon
        if not 1 <= h <= MIN_LAG:
            raise ValueError(
                f"forecast.horizon={h} but the feature SQL only uses lags >= {MIN_LAG}. "
                f"A horizon above {MIN_LAG} would leak future data; update sql/analytics/features_select.sql first."
            )
        f = self.features
        needed = max(MAX_WINDOW_LOOKBACK, f.dsls_cap_days + MIN_LAG)
        if f.lookback_buffer_days < needed:
            raise ValueError(
                f"features.lookback_buffer_days={f.lookback_buffer_days} must be >= {needed} "
                "or incremental feature builds would differ from full rebuilds."
            )
        for q in self.lgbm.quantiles:
            if not 0 < q < 1:
                raise ValueError(f"quantile {q} outside (0, 1)")
        if self.decision.mode not in ("newsvendor", "carryover"):
            raise ValueError("decision.mode must be 'newsvendor' or 'carryover'")
        return self

    @property
    def raw_files(self) -> dict[str, Path]:
        return {k: self.paths.raw_dir / v for k, v in self.files.model_dump().items()}


def _env_int(name: str) -> int | None:
    v = os.getenv(name, "").strip()
    return int(v) if v else None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    cfg_path = Path(os.getenv("SHELFWISE_CONFIG", PROJECT_ROOT / "config" / "config.yaml"))
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    with open(cfg_path) as fh:
        raw = yaml.safe_load(fh)

    # Environment overrides (handy for `SUBSET=CA make run`)
    raw["subset"] = parse_subset(os.getenv("SUBSET") or raw.get("subset"))
    last_day = _env_int("LAST_DAY")
    if last_day is not None:
        raw.setdefault("data", {})["last_day"] = last_day

    raw["db"] = DbCfg(
        user=os.getenv("POSTGRES_USER", "shelfwise"),
        password=os.getenv("POSTGRES_PASSWORD", "change_me_locally"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "shelfwise"),
    )
    settings = Settings(**raw)
    settings.paths = settings.paths.resolve()
    return settings
