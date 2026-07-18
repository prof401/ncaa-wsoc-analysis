"""Contest rows: season windows, outcomes, deduplication, W–L–T from contests_raw-style CSVs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd

_SCORE_RE = re.compile(
    r"^[WLT]\s+(\d+)-(\d+)(?:\s+\((\d+)\s+OT\))?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedResult:
    goals_for: int
    goals_against: int
    ot_periods: int | None  # None = regulation only


def season_date_bounds(season: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Map a teams.csv season label (e.g. ``\"2019-20\"``, ``\"2024-25\"``) to
    inclusive [start, end] dates: Aug 1 of the first year through Jul 31 of the second.
    """
    parts = season.strip().split("-")
    if len(parts) != 2:
        raise ValueError(f"Expected season like '2019-20', got {season!r}")
    y_start = int(parts[0])
    suffix = int(parts[1])
    century = (y_start // 100) * 100
    if suffix <= y_start % 100:
        y_end = century + 100 + suffix
    else:
        y_end = century + suffix
    start = pd.Timestamp(year=y_start, month=8, day=1)
    end = pd.Timestamp(year=y_end, month=7, day=31)
    return start, end


def parse_scores_from_result(result: Any) -> ParsedResult | None:
    """
    Parse goals and optional OT suffix from strings like ``W 2-1 (1 OT)``.

    Returns None when the score segment cannot be parsed.
    """
    if result is None or (isinstance(result, float) and pd.isna(result)):
        return None
    s = str(result).strip()
    if not s:
        return None
    m = _SCORE_RE.match(s)
    if not m:
        return None
    ot = int(m.group(3)) if m.group(3) is not None else None
    return ParsedResult(
        goals_for=int(m.group(1)),
        goals_against=int(m.group(2)),
        ot_periods=ot,
    )


def outcome_from_result(result: Any) -> str | None:
    """Return 'W', 'L', or 'T' from strings like ``W 2-1 (1 OT)``; else None."""
    if result is None or (isinstance(result, float) and pd.isna(result)):
        return None
    s = str(result).strip().upper()
    if not s:
        return None
    if s.startswith("W"):
        return "W"
    if s.startswith("L"):
        return "L"
    if s.startswith("T"):
        return "T"
    return None


def dedupe_contests_per_team(contests: pd.DataFrame) -> pd.DataFrame:
    """
    Keep one row per (contest_id, team_id).

    Raw scrapes may repeat the same game multiple times; duplicates would inflate W/L/T.
    """
    required = {"contest_id", "team_id"}
    missing = required - set(contests.columns)
    if missing:
        raise ValueError(f"contests missing columns: {sorted(missing)}")
    return contests.drop_duplicates(subset=["contest_id", "team_id"], keep="first")


def prepare_contests(contests: pd.DataFrame, *, date_column: str = "date") -> pd.DataFrame:
    """Parse dates and dedupe. Copies the frame."""
    out = contests.copy()
    if date_column not in out.columns:
        raise ValueError(f"Missing date column {date_column!r}")
    out[date_column] = pd.to_datetime(out[date_column], format="%m/%d/%Y", errors="coerce")
    return dedupe_contests_per_team(out)


def format_wlt(w: int, l: int, t: int) -> str:
    """Match teams.csv style: omit ties segment when zero."""
    if t == 0:
        return f"{w}-{l}"
    return f"{w}-{l}-{t}"


def contest_bilateral_mask(contests: pd.DataFrame) -> pd.Series:
    """
    True for rows whose ``contest_id`` appears on exactly two team rows (both sides present).
    """
    sizes = contests.groupby("contest_id")["contest_id"].transform("size")
    return sizes == 2


def aggregate_wlt_for_season(
    contests_prepared: pd.DataFrame,
    season: str,
    *,
    date_column: str = "date",
    bilateral_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For one season label, count W/L/T per team_id from deduped contest rows in the date window.

    Parameters
    ----------
    bilateral_only
        If True, only count rows for contests where both teams have a row (``contest_id``
        appears exactly twice in the deduped file for that season window).

    Returns
    -------
    counts
        Columns: team_id, wins_c, losses_c, ties_c, games_c
    unknown
        Rows in-window with unparseable ``result`` (still deduped).
    """
    lo, hi = season_date_bounds(season)
    sub = contests_prepared[
        (contests_prepared[date_column] >= lo) & (contests_prepared[date_column] <= hi)
    ].copy()
    if bilateral_only:
        sub = sub[contest_bilateral_mask(sub)]
    sub["outcome"] = sub["result"].map(outcome_from_result)
    unknown = sub[sub["outcome"].isna()].copy()
    good = sub[sub["outcome"].notna()]
    if good.empty:
        empty = pd.DataFrame(columns=["team_id", "wins_c", "losses_c", "ties_c", "games_c"])
        return empty, unknown
    vc = good.groupby(["team_id", "outcome"]).size().unstack(fill_value=0)
    for col in ("W", "L", "T"):
        if col not in vc.columns:
            vc[col] = 0
    vc = vc.rename(columns={"W": "wins_c", "L": "losses_c", "T": "ties_c"})
    vc["games_c"] = vc["wins_c"] + vc["losses_c"] + vc["ties_c"]
    vc["ncaa_win_pct_c"] = (vc["wins_c"] + 0.5 * vc["ties_c"]) / vc["games_c"].replace(0, pd.NA)
    vc = vc.reset_index()
    return vc, unknown


def aggregate_wlt_all_seasons(
    contests_prepared: pd.DataFrame,
    seasons: Iterable[str],
    *,
    bilateral_only: bool = False,
) -> pd.DataFrame:
    """Stack :func:`aggregate_wlt_for_season` for many season labels."""
    frames: list[pd.DataFrame] = []
    for season in seasons:
        counts, _ = aggregate_wlt_for_season(
            contests_prepared, season, bilateral_only=bilateral_only
        )
        if counts.empty:
            continue
        counts = counts.assign(season=season)
        frames.append(counts)
    if not frames:
        return pd.DataFrame(
            columns=[
                "team_id",
                "wins_c",
                "losses_c",
                "ties_c",
                "games_c",
                "ncaa_win_pct_c",
                "season",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def stated_tuple_from_teams_row(overall_record: Any) -> tuple[int, int, int] | None:
    from ncaa_wsoc.records import parse_overall_record

    return parse_overall_record(overall_record)


def contests_rows_for_team(
    contests_prepared: pd.DataFrame,
    team_id: int,
    *,
    season: str | None = None,
    date_column: str = "date",
) -> pd.DataFrame:
    """All deduped contest rows for ``team_id``, optionally restricted to a season window."""
    sub = contests_prepared[contests_prepared["team_id"] == team_id].copy()
    if season is not None:
        lo, hi = season_date_bounds(season)
        sub = sub[(sub[date_column] >= lo) & (sub[date_column] <= hi)]
    return sub.sort_values([date_column, "contest_id"])
