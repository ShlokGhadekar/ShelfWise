"""FastAPI service: (item, store) -> 28-day forecast, prediction interval, recommended order.

Run:  make api   (http://localhost:8000/docs for the interactive Swagger UI)
Example:
  curl "http://localhost:8000/forecast?item_id=FOODS_3_090&store_id=CA_1&history_days=28"
"""

from __future__ import annotations

import datetime as dt

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from shelfwise.config import get_settings
from shelfwise.serving import queries

app = FastAPI(
    title="ShelfWise forecast API",
    description="Daily unit-sales forecasts with P10-P90 intervals and newsvendor order recommendations "
                "for Walmart M5 item-store series.",
    version="0.1.0",
)


class ForecastDay(BaseModel):
    date: dt.date
    horizon: int = Field(description="days after the last observed day")
    forecast: float = Field(description="expected units (mean forecast)")
    p10: float | None
    p50: float | None
    p90: float | None
    baseline_snaive7: float | None = Field(description="naive forecast: same weekday last week")
    recommended_order: int = Field(description="newsvendor order quantity (or order-up-to level in carryover mode)")


class HistoryDay(BaseModel):
    date: dt.date
    sales: int | None
    sell_price: float | None


class ForecastResponse(BaseModel):
    item_id: str
    store_id: str
    series_id: int
    model_version: str
    critical_ratio: float = Field(description="c_u / (c_u + c_o): the demand quantile we order up to")
    decision_mode: str
    on_hand: int
    first_day_net_order: int = Field(description="max(0, day-1 recommended order - on_hand)")
    forecast: list[ForecastDay]
    history: list[HistoryDay] = []


def _f(v):
    return None if v is None or v != v else float(v)   # NaN -> None


@app.get("/health")
def health() -> dict:
    try:
        n = len(queries.stores())
        return {"status": "ok", "stores": n}
    except Exception as exc:  # DB down -> 503, not a stack trace
        raise HTTPException(status_code=503, detail=f"database unavailable: {exc}") from exc


@app.get("/forecast", response_model=ForecastResponse)
def get_forecast(
    item_id: str = Query(..., examples=["FOODS_3_090"], max_length=40),
    store_id: str = Query(..., examples=["CA_1"], max_length=10),
    history_days: int = Query(0, ge=0, le=730, description="also return this many days of history"),
    on_hand: int = Query(0, ge=0, description="units currently in stock (carryover mode)"),
) -> ForecastResponse:
    series = queries.find_series(item_id, store_id)
    if series is None:
        raise HTTPException(404, f"No series for item {item_id!r} in store {store_id!r} "
                                 "(is it in the configured SUBSET?)")
    fc = queries.forecast(int(series["series_id"]))
    if fc.empty:
        raise HTTPException(404, "No forecast stored yet for this series - run `make forecast`")
    hist = queries.history(int(series["series_id"]), history_days) if history_days else None

    days = [ForecastDay(date=r.date, horizon=int(r.horizon), forecast=float(r.point), p10=_f(r.p10), p50=_f(r.p50),
                        p90=_f(r.p90), baseline_snaive7=_f(r.baseline_snaive7), recommended_order=int(r.order_qty))
            for r in fc.itertuples()]
    return ForecastResponse(
        item_id=series["item_id"], store_id=series["store_id"], series_id=int(series["series_id"]),
        model_version=str(fc["model_version"].iloc[0]), critical_ratio=float(fc["critical_ratio"].iloc[0]),
        decision_mode=get_settings().decision.mode, on_hand=on_hand,
        first_day_net_order=max(0, days[0].recommended_order - on_hand),
        forecast=days,
        history=[] if hist is None else [
            HistoryDay(date=r.date, sales=None if r.sales != r.sales else int(r.sales), sell_price=_f(r.sell_price))
            for r in hist.itertuples()],
    )


@app.get("/models/comparison")
def model_comparison(fold: str = "all") -> list[dict]:
    df = queries.model_metrics(fold)
    if df.empty:
        raise HTTPException(404, "No metrics yet - run `make evaluate`")
    return df.to_dict(orient="records")
