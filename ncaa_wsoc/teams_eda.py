"""Exploratory summaries for team-season win totals and NCAA win percentage."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ncaa_wsoc.contests import aggregate_wlt_all_seasons, prepare_contests
from ncaa_wsoc.io import load_teams
from ncaa_wsoc.records import enrich_teams, mask_valid_games


@dataclass(frozen=True)
class WinPctSummary:
    label: str
    n: int
    wins_mean: float
    wins_median: float
    wins_std: float
    ncaa_win_pct_mean: float
    ncaa_win_pct_median: float
    ncaa_win_pct_std: float
    games_mean: float


def _summarize(label: str, df: pd.DataFrame, wins_col: str, pct_col: str, games_col: str) -> WinPctSummary:
    sub = df.dropna(subset=[wins_col, pct_col, games_col])
    return WinPctSummary(
        label=label,
        n=len(sub),
        wins_mean=float(sub[wins_col].mean()),
        wins_median=float(sub[wins_col].median()),
        wins_std=float(sub[wins_col].std(ddof=1)) if len(sub) > 1 else 0.0,
        ncaa_win_pct_mean=float(sub[pct_col].mean()),
        ncaa_win_pct_median=float(sub[pct_col].median()),
        ncaa_win_pct_std=float(sub[pct_col].std(ddof=1)) if len(sub) > 1 else 0.0,
        games_mean=float(sub[games_col].mean()),
    )


def build_team_season_frames(
    *,
    teams_csv=None,
    contests_csv=None,
    min_games: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Return stated, contests-all, contests-bilateral, and one-sided-impact frames.

    All contest frames are merged to ``team_id`` + ``season`` and filtered to ``min_games``.
    """
    from pathlib import Path

    from ncaa_wsoc.config import DEFAULT_CONTESTS_RAW_CSV, DEFAULT_TEAMS_CSV

    teams = enrich_teams(load_teams(teams_csv or DEFAULT_TEAMS_CSV))
    contests_path = Path(contests_csv) if contests_csv else DEFAULT_CONTESTS_RAW_CSV
    prep = prepare_contests(pd.read_csv(contests_path))
    seasons = sorted(teams["season"].dropna().unique())

    all_c = aggregate_wlt_all_seasons(prep, seasons, bilateral_only=False)
    bi_c = aggregate_wlt_all_seasons(prep, seasons, bilateral_only=True)

    def _merge_counts(counts: pd.DataFrame, suffix: str) -> pd.DataFrame:
        m = teams.merge(counts, on=["team_id", "season"], how="left")
        m = m.rename(
            columns={
                "wins_c": f"wins_{suffix}",
                "losses_c": f"losses_{suffix}",
                "ties_c": f"ties_{suffix}",
                "games_c": f"games_{suffix}",
                "ncaa_win_pct_c": f"ncaa_win_pct_{suffix}",
            }
        )
        return m

    stated = teams[mask_valid_games(teams, min_games)].copy()
    all_m = _merge_counts(all_c, "all")
    bi_m = _merge_counts(bi_c, "bi")
    all_m = all_m[all_m["games_all"].fillna(0) >= min_games].copy()
    bi_m = bi_m[bi_m["games_bi"].fillna(0) >= min_games].copy()

    impact = all_m.merge(
        bi_m[["team_id", "season", "wins_bi", "games_bi"]],
        on=["team_id", "season"],
        how="left",
    )
    impact["games_one_sided"] = impact["games_all"] - impact["games_bi"].fillna(0)
    impact["wins_delta"] = impact["wins_all"] - impact["wins_bi"].fillna(0)

    return stated, all_m, bi_m, impact[
        [
            "team_id",
            "season",
            "name",
            "wins_all",
            "wins_bi",
            "wins_delta",
            "games_all",
            "games_bi",
            "games_one_sided",
        ]
    ]


def summarize_sources(
    stated: pd.DataFrame,
    all_m: pd.DataFrame,
    bi_m: pd.DataFrame,
) -> list[WinPctSummary]:
    return [
        _summarize("teams.csv (stated overall_record)", stated, "wins", "ncaa_win_pct", "games"),
        _summarize(
            "contests_raw (all deduped games)",
            all_m,
            "wins_all",
            "ncaa_win_pct_all",
            "games_all",
        ),
        _summarize(
            "contests_raw (bilateral only — both teams have row)",
            bi_m,
            "wins_bi",
            "ncaa_win_pct_bi",
            "games_bi",
        ),
    ]


def dataset_overview(teams: pd.DataFrame, contests_raw: pd.DataFrame, prep: pd.DataFrame) -> pd.Series:
    from ncaa_wsoc.contests import contest_bilateral_mask

    bilateral = contest_bilateral_mask(prep)
    n_one_sided_rows = int((~bilateral).sum())
    n_one_sided_contests = int(prep.loc[~bilateral, "contest_id"].nunique())
    ex = contests_raw.get("exhibition")
    n_exhibition = int((ex == True).sum()) if ex is not None else 0  # noqa: E712
    return pd.Series(
        {
            "team_season_rows": len(teams),
            "seasons": teams["season"].nunique(),
            "contests_raw_rows": len(contests_raw),
            "contests_deduped_rows": len(prep),
            "contest_rows_one_sided": n_one_sided_rows,
            "contest_ids_one_sided": n_one_sided_contests,
            "exhibition_rows_raw": n_exhibition,
        }
    )


def format_summary_table(summaries: list[WinPctSummary]) -> pd.DataFrame:
    return pd.DataFrame([s.__dict__ for s in summaries])
