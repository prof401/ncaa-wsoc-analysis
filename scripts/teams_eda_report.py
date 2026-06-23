#!/usr/bin/env python3
"""
Exploratory summaries of team win totals and NCAA-style win percentage.

Compares three sources (filtered to ``--min-games``):

1. **stated** — ``overall_record`` on teams.csv
2. **contests_all** — W/L/T from deduped contests_raw in each season date window
3. **contests_bilateral** — same, but drops contests where only one team has a row
   (the other team did not count / report that game in contests_raw)

Optional histogram PNGs under ``figures/``.

Examples
--------
  python scripts/teams_eda_report.py
  python scripts/teams_eda_report.py --min-games 7 --plots
  python scripts/teams_eda_report.py --export data/teams_eda_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ncaa_wsoc.charts.histograms import plot_ncaa_win_pct_histogram, save_fig, setup_matplotlib_agg
from ncaa_wsoc.config import DEFAULT_CONTESTS_RAW_CSV, DEFAULT_FIGURE_DIR, DEFAULT_TEAMS_CSV
from ncaa_wsoc.contests import prepare_contests
from ncaa_wsoc.io import load_teams
from ncaa_wsoc.records import enrich_teams
from ncaa_wsoc.teams_eda import (
    build_team_season_frames,
    dataset_overview,
    format_summary_table,
    summarize_sources,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teams-csv", type=Path, default=DEFAULT_TEAMS_CSV)
    p.add_argument("--contests-csv", type=Path, default=DEFAULT_CONTESTS_RAW_CSV)
    p.add_argument("--min-games", type=int, default=7)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    p.add_argument("--plots", action="store_true", help="Write NCAA win %% histograms for each source")
    p.add_argument("--export", type=Path, default=None, help="Write summary stats table to CSV")
    p.add_argument("--export-impact", type=Path, default=None, help="Write per team-season one-sided game counts")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    teams = enrich_teams(load_teams(args.teams_csv))
    contests_raw = pd.read_csv(args.contests_csv)
    prep = prepare_contests(contests_raw)

    print("=== Dataset overview ===")
    print(dataset_overview(teams, contests_raw, prep).to_string())
    print(
        "\nOne-sided contests: only one team has a row for that contest_id in the deduped file "
        "(opponent may not have counted the game)."
    )

    stated, all_m, bi_m, impact = build_team_season_frames(
        teams_csv=args.teams_csv,
        contests_csv=args.contests_csv,
        min_games=args.min_games,
    )

    summaries = summarize_sources(stated, all_m, bi_m)
    table = format_summary_table(summaries)

    print(f"\n=== Win totals & NCAA win % (min_games={args.min_games}) ===")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    n_affected = int((impact["games_one_sided"] > 0).sum())
    print(f"\n=== One-sided contest impact (team-seasons) ===")
    print(f"team-seasons with at least one one-sided game: {n_affected} / {len(impact)}")
    if n_affected:
        sub = impact.loc[impact["games_one_sided"] > 0, "games_one_sided"]
        print(f"  mean one-sided games removed (bilateral filter): {sub.mean():.2f}")
        print(f"  max: {sub.max():.0f}")

    # Stated vs contests agreement
    cmp = all_m.merge(
        stated[["team_id", "season", "wins", "losses", "ties"]],
        on=["team_id", "season"],
        how="inner",
        suffixes=("", "_stated"),
    )
    match_all = (
        (cmp["wins_all"] == cmp["wins"])
        & (cmp["losses_all"] == cmp["losses"])
        & (cmp["ties_all"] == cmp["ties"])
    )
    print(f"\n=== teams.csv vs contests (all) ===")
    print(f"matching W-L-T: {match_all.sum()} / {len(match_all)} ({100 * match_all.mean():.1f}%)")

    if args.export:
        table.to_csv(args.export, index=False)
        print(f"\nWrote summary: {args.export.resolve()}")
    if args.export_impact:
        impact.to_csv(args.export_impact, index=False)
        print(f"Wrote impact: {args.export_impact.resolve()}")

    if args.plots:
        setup_matplotlib_agg()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        specs = [
            ("stated", stated["ncaa_win_pct"], "eda_ncaa_win_pct_stated.png", "teams.csv stated record"),
            (
                "contests_all",
                all_m["ncaa_win_pct_all"],
                "eda_ncaa_win_pct_contests_all.png",
                "contests_raw (all deduped)",
            ),
            (
                "contests_bilateral",
                bi_m["ncaa_win_pct_bi"],
                "eda_ncaa_win_pct_contests_bilateral.png",
                "contests_raw (bilateral only)",
            ),
        ]
        for _key, series, fname, title in specs:
            fig = plot_ncaa_win_pct_histogram(
                series.dropna(),
                bins=25,
                title=f"{title} (min {args.min_games} games)",
            )
            path = args.output_dir / fname
            save_fig(fig, path)
            plt.close(fig)
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
