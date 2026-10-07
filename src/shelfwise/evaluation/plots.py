"""Static figures for the README / reports (matplotlib, saved as PNG).

Style rules: one y-axis per chart, thin marks, hairline grid, the model of interest in
blue, the strongest baseline in orange, everything else in neutral gray. Text is never
coloured like a series.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed (runs in terminals / CI)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, INK, INK2, GRID, SURFACE = "#b9b8b2", "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
BLUE_BAND = "#b7d3f6"


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.labelsize": 10,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2,
        "font.family": "sans-serif",
    })


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def color_for(model: str, highlight: str, runner_up: str | None) -> str:
    if model == highlight or highlight == "*":   # "*" = single-series chart, all bars one colour
        return BLUE
    if model == runner_up:
        return ORANGE
    return GRAY


def bar_metric(df: pd.DataFrame, value_col: str, label_col: str, title: str, xlabel: str, path: Path,
               highlight: str, runner_up: str | None = None, fmt: str = "{:.3f}") -> Path:
    _style()
    d = df.sort_values(value_col, ascending=False)
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(d) + 1.2))
    colors = [color_for(m, highlight, runner_up) for m in d[label_col]]
    ax.barh(d[label_col], d[value_col], color=colors, height=0.6)
    for y, v in enumerate(d[value_col]):
        ax.text(v, y, " " + fmt.format(v), va="center", fontsize=8, color=INK2)
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def lines_by_x(df: pd.DataFrame, x: str, y: str, series_col: str, title: str, xlabel: str, ylabel: str,
               path: Path, highlight: str, runner_up: str | None = None) -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for name, g in df.groupby(series_col):
        c = color_for(name, highlight, runner_up)
        g = g.sort_values(x)
        ax.plot(g[x], g[y], color=c, lw=2 if c != GRAY else 1.2, label=name)
        ax.annotate(name, (g[x].iloc[-1], g[y].iloc[-1]), xytext=(4, 0), textcoords="offset points",
                    fontsize=8, color=INK2, va="center")
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left")
    return _save(fig, path)


def grouped_segments(df: pd.DataFrame, path: Path, models: list[str], title: str) -> Path:
    """df: segment_type, segment, model, value -> small multiples, one panel per segment type."""
    _style()
    types = list(dict.fromkeys(df["segment_type"]))
    n_segs = [df.loc[df["segment_type"] == t, "segment"].nunique() for t in types]
    fig, axes = plt.subplots(len(types), 1, figsize=(7, 1.0 + 0.45 * sum(n_segs) + 0.35 * len(types)),
                             squeeze=False, gridspec_kw={"height_ratios": n_segs})
    palette = [BLUE, ORANGE, GRAY]
    for ax, t in zip(axes[:, 0], types, strict=True):
        d = df[df["segment_type"] == t]
        segs = list(dict.fromkeys(d["segment"]))
        y = np.arange(len(segs))
        width = 0.8 / len(models)
        for i, mname in enumerate(models):
            vals = d[d["model"] == mname].set_index("segment").reindex(segs)["value"]
            ax.barh(y + i * width, vals, height=width * 0.9, color=palette[min(i, 2)], label=mname)
        ax.set_yticks(y + width * (len(models) - 1) / 2, segs)
        ax.set_title(t, loc="left", fontsize=10)
        ax.grid(axis="y", visible=False)
    axes[0, 0].legend(loc="lower right")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="semibold", color=INK)
    return _save(fig, path)


def forecast_examples(hist: dict, fc: dict, path: Path, title: str) -> Path:
    """hist[name] = (days, values); fc[name] = (days, actual, point, p10, p90)."""
    _style()
    n = len(fc)
    fig, axes = plt.subplots(n, 1, figsize=(8, 2.4 * n), squeeze=False)
    for ax, name in zip(axes[:, 0], fc, strict=True):
        hd, hv = hist[name]
        d, actual, point, lo, hi = fc[name]
        ax.plot(hd, hv, color=GRAY, lw=1.2, label="history")
        ax.fill_between(d, lo, hi, color=BLUE_BAND, lw=0, label="P10-P90")
        ax.plot(d, point, color=BLUE, lw=2, label="forecast (mean)")
        ax.plot(d, actual, color=INK, lw=1, marker="o", ms=3, label="actual")
        ax.set_title(name, loc="left", fontsize=10)
        ax.set_ylabel("units / day")
    axes[0, 0].legend(loc="upper left", ncol=4)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="semibold", color=INK)
    return _save(fig, path)
