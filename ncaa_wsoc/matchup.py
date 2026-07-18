"""Match outcome probabilities, season highlights, and contest-level lookups."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd
from scipy.stats import poisson

from ncaa_wsoc.contests import (
    contest_bilateral_mask,
    outcome_from_result,
    parse_scores_from_result,
    prepare_contests,
    season_date_bounds,
)
from ncaa_wsoc.io import load_contests, load_teams
from ncaa_wsoc.records import enrich_teams

RatingsAsOf = Literal["pregame", "season_end"]

_GAME_LOG_COLUMNS = [
    "contest_id",
    "team_id",
    "opponent_id",
    "date",
    "season",
    "outcome",
    "goals_for",
    "goals_against",
    "margin",
    "went_to_ot",
    "name",
    "division",
    "ncaa_win_pct",
    "opponent_name",
    "opponent_ncaa_win_pct",
    "result",
]


@dataclass(frozen=True)
class GameHighlight:
    contest_id: int
    date: pd.Timestamp
    opponent_id: int
    opponent_name: str
    scoreline: str
    margin: int
    surprise: float | None


@dataclass(frozen=True)
class ContestProbabilities:
    contest_id: int
    team_a_id: int
    team_b_id: int
    p_win_a: float
    p_tie: float
    p_win_b: float


def _is_exhibition(value: object) -> bool:
    if value is True or value == 1:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return False


def season_from_date(date: pd.Timestamp, seasons: Iterable[str]) -> str | None:
    """Return the teams.csv season label containing ``date``, if any."""
    for season in seasons:
        lo, hi = season_date_bounds(season)
        if lo <= date <= hi:
            return season
    return None


def build_game_log(
    contests: pd.DataFrame,
    teams: pd.DataFrame,
    *,
    season: str | None = None,
) -> pd.DataFrame:
    """
    One row per team-game with parsed scores and team metadata.

    Filters to non-exhibition bilateral contests in the NCAA season window.
    """
    prepared = prepare_contests(contests)
    teams_enriched = enrich_teams(teams)

    sub = prepared[~prepared["exhibition"].map(_is_exhibition)].copy()
    sub = sub[contest_bilateral_mask(sub)]

    seasons = sorted(teams_enriched["season"].dropna().unique())
    if season is not None:
        lo, hi = season_date_bounds(season)
        sub = sub[(sub["date"] >= lo) & (sub["date"] <= hi)]
        seasons = [season]

    sub["outcome"] = sub["result"].map(outcome_from_result)
    parsed = sub["result"].map(parse_scores_from_result)
    sub["goals_for"] = parsed.map(lambda p: p.goals_for if p else np.nan)
    sub["goals_against"] = parsed.map(lambda p: p.goals_against if p else np.nan)
    sub["went_to_ot"] = parsed.map(lambda p: p.ot_periods is not None if p else False)
    sub["season"] = sub["date"].map(lambda d: season_from_date(d, seasons))
    sub = sub[sub["outcome"].notna() & sub["goals_for"].notna() & sub["season"].notna()].copy()
    sub["margin"] = (sub["goals_for"] - sub["goals_against"]).astype(int)

    team_cols = teams_enriched[
        ["team_id", "season", "name", "division", "ncaa_win_pct"]
    ].drop_duplicates(subset=["team_id", "season"])
    opp_cols = team_cols.rename(
        columns={
            "team_id": "opponent_id",
            "name": "opponent_name",
            "ncaa_win_pct": "opponent_ncaa_win_pct",
        }
    ).drop(columns=["division"])

    out = sub.merge(team_cols, on=["team_id", "season"], how="left")
    out = out.merge(opp_cols, on=["opponent_id", "season"], how="left")
    return out[_GAME_LOG_COLUMNS].sort_values(["season", "date", "contest_id", "team_id"])


def league_baseline(game_log: pd.DataFrame) -> float:
    """Mean goals scored per team per game (λ₀)."""
    if game_log.empty:
        return 1.0
    return float(game_log["goals_for"].mean())


def fit_poisson_rates(
    game_log: pd.DataFrame,
    *,
    as_of_date: pd.Timestamp | None = None,
    min_games: int = 5,
) -> pd.DataFrame:
    """
    Per-team attack and defense rates from a game log.

    When ``as_of_date`` is set, only games strictly before that date are used
    (pregame ratings). Teams with fewer than ``min_games`` use league averages.
    """
    log = game_log
    if as_of_date is not None:
        log = log[log["date"] < as_of_date]
    if log.empty:
        return pd.DataFrame(
            columns=["team_id", "attack", "defense", "games", "low_confidence"]
        )

    lambda0 = league_baseline(log)
    agg = (
        log.groupby("team_id", as_index=False)
        .agg(
            goals_for=("goals_for", "sum"),
            goals_against=("goals_against", "sum"),
            games=("contest_id", "count"),
        )
    )
    agg["attack"] = agg["goals_for"] / agg["games"]
    agg["defense"] = agg["goals_against"] / agg["games"]
    low = agg["games"] < min_games
    agg.loc[low, "attack"] = lambda0
    agg.loc[low, "defense"] = lambda0
    agg["low_confidence"] = low
    return agg[["team_id", "attack", "defense", "games", "low_confidence"]]


def _team_rate(rates: pd.DataFrame, team_id: int, column: str) -> float | None:
    rows = rates.loc[rates["team_id"] == team_id, column]
    if rows.empty:
        return None
    return float(rows.iloc[0])


def _win_pct_fallback_probabilities(
    team_a_id: int,
    team_b_id: int,
    game_log: pd.DataFrame,
) -> tuple[float, float, float]:
    """Secondary path when Poisson rates are unavailable."""
    sub = game_log[game_log["team_id"].isin([team_a_id, team_b_id])]
    league_tie = float((sub["outcome"] == "T").mean()) if not sub.empty else 0.15

    def strength(team_id: int) -> float:
        rows = game_log[game_log["team_id"] == team_id]
        if rows.empty or rows["ncaa_win_pct"].isna().all():
            return 0.5
        return float(rows["ncaa_win_pct"].iloc[-1])

    sa, sb = strength(team_a_id), strength(team_b_id)
    denom = sa + sb
    if denom <= 0:
        p_win_a = p_win_b = 0.5 * (1.0 - league_tie)
        return p_win_a, league_tie, p_win_b

    p_non_tie = 1.0 - league_tie
    closeness = 1.0 - abs(sa - sb)
    p_tie = league_tie * closeness
    p_tie = min(max(p_tie, 0.0), 1.0)
    remainder = 1.0 - p_tie
    p_win_a = remainder * (sa / denom)
    p_win_b = remainder * (sb / denom)
    return p_win_a, p_tie, p_win_b


def matchup_probabilities(
    team_a_id: int,
    team_b_id: int,
    rates: pd.DataFrame,
    *,
    league_lambda: float,
    game_log: pd.DataFrame | None = None,
    max_goals: int = 10,
) -> tuple[float, float, float]:
    """
    Return (p_win_a, p_tie, p_win_b) from independent Poisson goal counts.

  Falls back to win-% proxy when either team's rates are missing.
    """
    attack_a = _team_rate(rates, team_a_id, "attack")
    defense_a = _team_rate(rates, team_a_id, "defense")
    attack_b = _team_rate(rates, team_b_id, "attack")
    defense_b = _team_rate(rates, team_b_id, "defense")

    if None in (attack_a, defense_a, attack_b, defense_b):
        if game_log is None:
            raise ValueError("game_log required when team rates are missing")
        return _win_pct_fallback_probabilities(team_a_id, team_b_id, game_log)

    if league_lambda <= 0:
        league_lambda = 1.0

    lambda_a = attack_a * (defense_b / league_lambda)
    lambda_b = attack_b * (defense_a / league_lambda)

    goals = np.arange(max_goals + 1)
    pmf_a = poisson.pmf(goals, lambda_a)
    pmf_b = poisson.pmf(goals, lambda_b)

    p_win_a = 0.0
    p_tie = 0.0
    p_win_b = 0.0
    for i, pi in enumerate(pmf_a):
        for j, pj in enumerate(pmf_b):
            p = pi * pj
            if i > j:
                p_win_a += p
            elif i == j:
                p_tie += p
            else:
                p_win_b += p

    total = p_win_a + p_tie + p_win_b
    if total <= 0:
        return _win_pct_fallback_probabilities(team_a_id, team_b_id, game_log or pd.DataFrame())
    return p_win_a / total, p_tie / total, p_win_b / total


def _row_to_highlight(row: pd.Series, surprise: float | None) -> GameHighlight:
    return GameHighlight(
        contest_id=int(row["contest_id"]),
        date=row["date"],
        opponent_id=int(row["opponent_id"]),
        opponent_name=str(row["opponent_name"]),
        scoreline=str(row["result"]),
        margin=int(row["margin"]),
        surprise=surprise,
    )


def _pick_highlight(
    rows: pd.DataFrame,
    *,
    kind: Literal["win", "loss"],
    rates: pd.DataFrame,
    strength_adjusted: bool,
) -> GameHighlight | None:
    if rows.empty:
        return None

    work = rows.copy()
    if strength_adjusted:
        rate_map = rates.set_index("team_id")["attack"].to_dict()

        def surprise(row: pd.Series) -> float:
            team_attack = rate_map.get(int(row["team_id"]), np.nan)
            opp_attack = rate_map.get(int(row["opponent_id"]), np.nan)
            if pd.isna(team_attack) or pd.isna(opp_attack):
                expected = 0.0
            else:
                expected = team_attack - opp_attack
            return float(row["margin"]) - expected

        work["surprise"] = work.apply(surprise, axis=1)
        if kind == "win":
            work = work.sort_values(
                ["surprise", "margin", "opponent_ncaa_win_pct"],
                ascending=[False, False, False],
            )
        else:
            work = work.sort_values(
                ["surprise", "margin", "opponent_ncaa_win_pct"],
                ascending=[True, True, False],
            )
        best = work.iloc[0]
        return _row_to_highlight(best, float(best["surprise"]))

    if kind == "win":
        work = work.sort_values(
            ["margin", "opponent_ncaa_win_pct"],
            ascending=[False, False],
        )
    else:
        work = work.sort_values(
            ["margin", "opponent_ncaa_win_pct"],
            ascending=[True, False],
        )
    return _row_to_highlight(work.iloc[0], None)


def season_highlights(
    team_id: int,
    season: str,
    game_log: pd.DataFrame,
    rates: pd.DataFrame,
) -> dict[str, GameHighlight | None]:
    """Biggest win and loss by raw margin, plus strength-adjusted variants."""
    sub = game_log[(game_log["team_id"] == team_id) & (game_log["season"] == season)]
    wins = sub[sub["outcome"] == "W"]
    losses = sub[sub["outcome"] == "L"]
    return {
        "biggest_win": _pick_highlight(wins, kind="win", rates=rates, strength_adjusted=False),
        "biggest_loss": _pick_highlight(losses, kind="loss", rates=rates, strength_adjusted=False),
        "biggest_win_adjusted": _pick_highlight(
            wins, kind="win", rates=rates, strength_adjusted=True
        ),
        "biggest_loss_adjusted": _pick_highlight(
            losses, kind="loss", rates=rates, strength_adjusted=True
        ),
    }


def contest_probabilities(
    contest_id: int,
    *,
    ratings_as_of: RatingsAsOf = "pregame",
    contests: pd.DataFrame | None = None,
    teams: pd.DataFrame | None = None,
) -> ContestProbabilities:
    """
    Pre-game or season-end win/tie/loss probabilities for a bilateral contest.

    Team A is the lower ``team_id``; Team B is the higher ``team_id``.
    """
    if contests is None:
        contests = load_contests()
    if teams is None:
        teams = load_teams()

    prepared = prepare_contests(contests)
    rows = prepared[prepared["contest_id"] == contest_id]
    if len(rows) != 2:
        raise ValueError(f"contest_id {contest_id} must have exactly 2 rows, got {len(rows)}")

    team_a_id = int(min(rows["team_id"]))
    team_b_id = int(max(rows["team_id"]))
    contest_date = rows["date"].iloc[0]
    season = season_from_date(contest_date, teams["season"].dropna().unique())
    if season is None:
        raise ValueError(f"contest_id {contest_id} date {contest_date} is outside known seasons")

    game_log = build_game_log(contests, teams, season=season)
    as_of = contest_date if ratings_as_of == "pregame" else None
    rates = fit_poisson_rates(game_log, as_of_date=as_of)
    lambda0 = league_baseline(game_log if as_of is None else game_log[game_log["date"] < as_of])

    p_win_a, p_tie, p_win_b = matchup_probabilities(
        team_a_id,
        team_b_id,
        rates,
        league_lambda=lambda0,
        game_log=game_log,
    )
    return ContestProbabilities(
        contest_id=contest_id,
        team_a_id=team_a_id,
        team_b_id=team_b_id,
        p_win_a=p_win_a,
        p_tie=p_tie,
        p_win_b=p_win_b,
    )


def event_impact_weight(
    team_id: int,
    contest_id: int,
    *,
    contests: pd.DataFrame | None = None,
    teams: pd.DataFrame | None = None,
) -> float:
    """
    Discount favorite events and boost underdog events using pregame win probability.

    ``p > 0.5`` → ``1 - p``; ``p <= 0.5`` → ``1 + (0.5 - p)``.
    """
    probs = contest_probabilities(
        contest_id,
        ratings_as_of="pregame",
        contests=contests,
        teams=teams,
    )
    if team_id == probs.team_a_id:
        p = probs.p_win_a
    elif team_id == probs.team_b_id:
        p = probs.p_win_b
    else:
        raise ValueError(f"team_id {team_id} is not in contest_id {contest_id}")

    if p > 0.5:
        return 1.0 - p
    return 1.0 + (0.5 - p)


def league_average_opponent_id(game_log: pd.DataFrame, season: str) -> int | None:
    """Team id closest to league-average ncaa_win_pct in a season (for vs-average matchups)."""
    sub = game_log[game_log["season"] == season]
    if sub.empty:
        return None
    team_pcts = sub.groupby("team_id")["ncaa_win_pct"].first()
    if team_pcts.empty:
        return None
    median_pct = float(team_pcts.median())
    return int(team_pcts.sub(median_pct).abs().idxmin())
