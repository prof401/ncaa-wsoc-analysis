"""Histogram charts for EDA."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


def setup_matplotlib_agg() -> None:
    """Use non-interactive backend (safe for headless / file output)."""
    import matplotlib

    matplotlib.use("Agg")


def save_fig(fig: Figure, path: Path | str, *, dpi: int = 150) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)


def _series_from_input(
    data: pd.Series | pd.DataFrame,
    column: str | None,
) -> pd.Series:
    if isinstance(data, pd.Series):
        return data.dropna()
    if column is None:
        raise ValueError("column is required when data is a DataFrame")
    return data[column].dropna()


def plot_ncaa_win_pct_histogram(
    data: pd.Series | pd.DataFrame,
    *,
    column: str | None = "ncaa_win_pct",
    bins: int = 25,
    range_: tuple[float, float] = (0.0, 1.0),
    highlight_half: bool = True,
    mean_line: bool = True,
    title: str | None = None,
    figsize: tuple[float, float] = (11.0, 5.0),
    ax: plt.Axes | None = None,
) -> Figure:
    """
    Histogram of NCAA-style win percentage.

    When ``highlight_half`` is True, the bin whose interval contains 0.5 is
    drawn in a distinct color (``darkorange`` vs ``seagreen``).
    """
    s = _series_from_input(data, column)
    fig = None
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure

    counts, edges = np.histogram(s, bins=bins, range=range_)
    widths = np.diff(edges)
    centers = edges[:-1] + widths / 2

    if highlight_half:
        highlight_idx = int(np.searchsorted(edges, 0.5, side="right") - 1)
        highlight_idx = max(0, min(highlight_idx, len(counts) - 1))
        colors = ["seagreen"] * len(counts)
        colors[highlight_idx] = "darkorange"
    else:
        colors = ["seagreen"] * len(counts)

    ax.bar(
        centers,
        counts,
        width=widths,
        color=colors,
        edgecolor="white",
        linewidth=0.7,
        alpha=0.92,
        align="center",
    )
    ax.set_xlim(range_)
    ax.set_xlabel("NCAA-style winning percentage ((W + 0.5×T) ÷ games)")
    ax.set_ylabel("Count (team-seasons)")
    if title:
        ax.set_title(title)
    if mean_line and len(s):
        m = float(s.mean())
        ax.axvline(m, color="darkred", linestyle="--", linewidth=1.2, label=f"mean={m:.3f}")
    ax.axvline(0.5, color="orange", linestyle=":", linewidth=1, alpha=0.9, label="0.5")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_games_played_histogram(
    data: pd.Series | pd.DataFrame,
    *,
    column: str | None = "games",
    integer_bins: bool = True,
    mean_line: bool = True,
    title: str | None = None,
    figsize: tuple[float, float] = (11.0, 5.0),
    ax: plt.Axes | None = None,
) -> Figure:
    """Histogram of games played (W + L + T)."""
    s = _series_from_input(data, column)
    fig = None
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure

    if integer_bins and len(s):
        min_g = int(s.min())
        max_g = int(s.max())
        bins = np.arange(min_g - 0.5, max_g + 1.5, 1)
        ax.set_xticks(np.arange(min_g, max_g + 1, 1))
        ax.tick_params(axis="x", labelrotation=90)
    else:
        bins = 30

    ax.hist(s, bins=bins, color="slateblue", edgecolor="white", alpha=0.92)
    ax.set_xlabel("Number of games played (W + L + T)")
    ax.set_ylabel("Count (team-seasons)")
    if title:
        ax.set_title(title)
    if mean_line and len(s):
        m = float(s.mean())
        ax.axvline(m, color="darkred", linestyle="--", linewidth=1.2, label=f"mean={m:.2f}")
        ax.legend()
    fig.tight_layout()
    return fig
