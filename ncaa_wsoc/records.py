"""Parse overall_record and derive wins, losses, ties, NCAA win percentage."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def parse_overall_record(value: Any) -> tuple[int, int, int] | None:
    """
    Parse ``overall_record`` strings ``W-L`` or ``W-L-T``.

    Returns
    -------
    (wins, losses, ties) or None if invalid or missing.
    """
    if value is None or pd.isna(value):
        return None
    s = str(value).strip()
    if not s:
        return None
    parts = s.split("-")
    try:
        if len(parts) == 2:
            return int(parts[0]), int(parts[1]), 0
        if len(parts) == 3:
            return int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None
    return None


def enrich_teams(df: pd.DataFrame, *, inplace: bool = False) -> pd.DataFrame:
    """
    Add ``wins``, ``losses``, ``ties``, ``games``, ``ncaa_win_pct``.

    ``ncaa_win_pct`` is (W + 0.5*T) / games when games > 0; otherwise NaN.
    """
    out = df if inplace else df.copy()
    parsed = out["overall_record"].map(parse_overall_record)
    out["wins"] = parsed.map(lambda p: p[0] if p else np.nan)
    out["losses"] = parsed.map(lambda p: p[1] if p else np.nan)
    out["ties"] = parsed.map(lambda p: p[2] if p else np.nan)
    out["games"] = out["wins"] + out["losses"] + out["ties"]
    out["ncaa_win_pct"] = (out["wins"] + 0.5 * out["ties"]) / out["games"]
    out.loc[out["games"].isna() | (out["games"] <= 0), "ncaa_win_pct"] = np.nan
    return out


def mask_valid_games(df: pd.DataFrame, min_games: int = 1) -> pd.Series:
    """True where ``games`` is finite and ``games >= min_games``."""
    g = df["games"]
    return g.notna() & (g >= min_games)
