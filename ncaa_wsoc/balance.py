"""Aggregate wins / losses / ties and diagnose imbalance vs perfect double-entry parity."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def _apply_mask(df: pd.DataFrame, mask: pd.Series | None) -> pd.DataFrame:
    if mask is None:
        return df
    return df.loc[mask]


def summarize_wlt(df: pd.DataFrame, *, mask: pd.Series | None = None) -> pd.Series:
    """
    Sum wins, losses, ties, and games over rows.

    Returns
    -------
    Series with keys: n_rows, wins, losses, ties, games, imbalance (wins - losses),
    ties_parity_ok (True if ties sum is even), half_tie_games_proxy (ties / 2).
    """
    sub = _apply_mask(df, mask)
    w = sub["wins"].sum(min_count=1)
    l = sub["losses"].sum(min_count=1)
    t = sub["ties"].sum(min_count=1)
    g = sub["games"].sum(min_count=1)
    w = 0 if pd.isna(w) else int(w)
    l = 0 if pd.isna(l) else int(l)
    t = 0 if pd.isna(t) else int(t)
    g = 0 if pd.isna(g) else int(g)
    imb = w - l
    ties_even = (t % 2 == 0)
    return pd.Series(
        {
            "n_rows": int(len(sub)),
            "wins": w,
            "losses": l,
            "ties": t,
            "games": g,
            "imbalance": imb,
            "ties_parity_ok": ties_even,
            "half_tie_games_proxy": t / 2.0,
        }
    )


def breakdown_wlt(
    df: pd.DataFrame,
    by: Iterable[str] | str,
    *,
    mask: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Grouped sums plus imbalance per group.

    Parameters
    ----------
    by
        Column name(s) present on ``df`` (e.g. ``\"season\"``, ``[\"season\", \"division\"]``).
    """
    if isinstance(by, str):
        by_list = [by]
    else:
        by_list = list(by)
    sub = _apply_mask(df, mask)
    grp = sub.groupby(by_list, dropna=False, sort=True)
    sizes = grp.size().rename("n_rows")
    sums = grp[["wins", "losses", "ties", "games"]].sum()
    out = pd.concat([sizes, sums], axis=1)
    out["imbalance"] = out["wins"] - out["losses"]
    out["ties_parity_ok"] = out["ties"] % 2 == 0
    out = out.reset_index()
    return out


def row_net_wins_minus_losses(df: pd.DataFrame, *, mask: pd.Series | None = None) -> pd.Series:
    """Per-row (wins - losses); sum equals global imbalance for the same mask."""
    sub = _apply_mask(df, mask)
    return sub["wins"] - sub["losses"]


def parse_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tag rows by whether ``overall_record`` parsed and games > 0.

    Useful when excluding bad rows changes totals.
    """
    from ncaa_wsoc.records import parse_overall_record

    def status(row) -> str:
        raw = row.get("overall_record")
        p = parse_overall_record(raw)
        if p is None:
            return "unparseable_or_missing"
        w, l, t = p
        if w + l + t <= 0:
            return "zero_games"
        return "ok"

    out = df.copy()
    out["_parse_status"] = out.apply(status, axis=1)
    return out


def summarize_by_parse_status(df_enriched: pd.DataFrame) -> pd.DataFrame:
    """Requires columns from :func:`parse_status` (``_parse_status``)."""
    if "_parse_status" not in df_enriched.columns:
        df_enriched = parse_status(df_enriched)
    rows = []
    for label, sub in df_enriched.groupby("_parse_status", dropna=False):
        s = summarize_wlt(sub)
        s["_parse_status"] = label
        rows.append(s)
    return pd.DataFrame(rows).set_index("_parse_status")


def top_groups_by_abs_imbalance(
    breakdown: pd.DataFrame,
    *,
    k: int = 10,
) -> pd.DataFrame:
    """Sort breakdown by |imbalance| descending and take ``k`` rows."""
    b = breakdown.copy()
    b["_abs_imb"] = b["imbalance"].abs()
    b = b.sort_values("_abs_imb", ascending=False).head(k).drop(columns=["_abs_imb"])
    return b.reset_index(drop=True)
