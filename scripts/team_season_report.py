#!/usr/bin/env python3
"""
Team-season matchup report: Poisson rates, win/tie/loss probabilities, and highlights.

Examples
--------
  python scripts/team_season_report.py --team-id 603188 --season 2025-26
  python scripts/team_season_report.py --team-id 603188 --season 2025-26 --vs-team-id 603525
  python scripts/team_season_report.py --contest-id 6399888
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from ncaa_wsoc.config import DEFAULT_CONTESTS_RAW_CSV, DEFAULT_TEAMS_CSV
from ncaa_wsoc.contests import format_wlt
from ncaa_wsoc.io import load_contests, load_teams
from ncaa_wsoc.matchup import (
    build_game_log,
    contest_probabilities,
    event_impact_weight,
    fit_poisson_rates,
    league_average_opponent_id,
    league_baseline,
    matchup_probabilities,
    season_highlights,
)


def _pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def _print_highlight(label: str, highlight) -> None:
    print(f"\n{label}")
    if highlight is None:
        print("  (none)")
        return
    print(f"  {highlight.date.date()} vs {highlight.opponent_name} ({highlight.scoreline})")
    print(f"  margin={highlight.margin:+d}", end="")
    if highlight.surprise is not None:
        print(f", surprise={highlight.surprise:+.2f}")
    else:
        print()


def run_team_season(args: argparse.Namespace) -> int:
    teams = load_teams(args.teams_csv)
    contests = load_contests(args.contests_csv)
    game_log = build_game_log(contests, teams, season=args.season)
    sub = game_log[game_log["team_id"] == args.team_id]
    if sub.empty:
        print(f"No games for team_id={args.team_id} in season {args.season!r}", file=sys.stderr)
        return 1

    team_name = str(sub["name"].iloc[0])
    wins = int((sub["outcome"] == "W").sum())
    losses = int((sub["outcome"] == "L").sum())
    ties = int((sub["outcome"] == "T").sum())

    print(f"=== {team_name} ({args.season}) ===")
    print(f"Record (contest-derived): {format_wlt(wins, losses, ties)}")

    rates = fit_poisson_rates(game_log)
    lambda0 = league_baseline(game_log)
    team_rates = rates[rates["team_id"] == args.team_id]
    if team_rates.empty:
        print("No Poisson rates for this team.")
    else:
        tr = team_rates.iloc[0]
        print(
            f"Poisson rates: attack={tr['attack']:.3f}, defense={tr['defense']:.3f}, "
            f"games={int(tr['games'])}, low_confidence={bool(tr['low_confidence'])}"
        )
    print(f"League baseline λ₀={lambda0:.3f}")

    vs_id = args.vs_team_id
    if vs_id is None:
        vs_id = league_average_opponent_id(game_log, args.season)
        vs_label = "league-average opponent"
    else:
        vs_rows = game_log[game_log["team_id"] == vs_id]
        vs_label = str(vs_rows["name"].iloc[0]) if not vs_rows.empty else f"team_id={vs_id}"

    if vs_id is not None:
        team_a, team_b = sorted([args.team_id, vs_id])
        p_a, p_tie, p_b = matchup_probabilities(
            team_a,
            team_b,
            rates,
            league_lambda=lambda0,
            game_log=game_log,
        )
        if args.team_id == team_a:
            p_win, p_loss = p_a, p_b
        else:
            p_win, p_loss = p_b, p_a
        print(f"\nvs {vs_label} (season-end ratings):")
        print(f"  P(win)={_pct(p_win)}, P(tie)={_pct(p_tie)}, P(loss)={_pct(p_loss)}")

    highlights = season_highlights(args.team_id, args.season, game_log, rates)
    _print_highlight("Biggest win (raw margin)", highlights["biggest_win"])
    _print_highlight("Biggest loss (raw margin)", highlights["biggest_loss"])
    _print_highlight("Biggest win (strength-adjusted)", highlights["biggest_win_adjusted"])
    _print_highlight("Biggest loss (strength-adjusted)", highlights["biggest_loss_adjusted"])
    return 0


def run_contest(args: argparse.Namespace) -> int:
    teams = load_teams(args.teams_csv)
    contests = load_contests(args.contests_csv)
    probs = contest_probabilities(
        args.contest_id,
        ratings_as_of=args.ratings_as_of,
        contests=contests,
        teams=teams,
    )
    team_names = teams.set_index("team_id")["name"].to_dict()
    name_a = team_names.get(probs.team_a_id, str(probs.team_a_id))
    name_b = team_names.get(probs.team_b_id, str(probs.team_b_id))

    print(f"=== contest_id {probs.contest_id} ({args.ratings_as_of} ratings) ===")
    print(f"Team A: {name_a} ({probs.team_a_id})")
    print(f"  P(win)={_pct(probs.p_win_a)}, P(tie)={_pct(probs.p_tie)}")
    print(f"Team B: {name_b} ({probs.team_b_id})")
    print(f"  P(win)={_pct(probs.p_win_b)}, P(tie)={_pct(probs.p_tie)}")
    print(f"Sum: {_pct(probs.p_win_a + probs.p_tie + probs.p_win_b)}")

    for tid, label in ((probs.team_a_id, name_a), (probs.team_b_id, name_b)):
        w = event_impact_weight(
            tid,
            args.contest_id,
            contests=contests,
            teams=teams,
        )
        print(f"event_impact_weight({label}): {w:.3f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teams-csv", type=Path, default=DEFAULT_TEAMS_CSV)
    p.add_argument("--contests-csv", type=Path, default=DEFAULT_CONTESTS_RAW_CSV)

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--contest-id", type=int, help="Print probabilities for one contest")
    mode.add_argument("--team-id", type=int, help="Team id for season report")

    p.add_argument("--season", help="Season label (required with --team-id)")
    p.add_argument("--vs-team-id", type=int, default=None, help="Opponent for head-to-head probabilities")
    p.add_argument(
        "--ratings-as-of",
        choices=["pregame", "season_end"],
        default="pregame",
        help="Rating window for --contest-id (default: pregame)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.contest_id is not None:
        return run_contest(args)
    if args.season is None:
        print("--season is required with --team-id", file=sys.stderr)
        return 2
    return run_team_season(args)


if __name__ == "__main__":
    raise SystemExit(main())
