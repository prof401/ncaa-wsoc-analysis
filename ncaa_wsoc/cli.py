"""Command-line entry to generate EDA histogram PNGs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from ncaa_wsoc.charts.histograms import (
    plot_games_played_histogram,
    plot_ncaa_win_pct_histogram,
    save_fig,
    setup_matplotlib_agg,
)
from ncaa_wsoc.config import DEFAULT_FIGURE_DIR, DEFAULT_TEAMS_CSV
from ncaa_wsoc.io import load_teams
from ncaa_wsoc.records import enrich_teams, mask_valid_games


def _ncaa_output_name(min_games: int) -> str:
    if min_games <= 1:
        return "eda_win_pct_ncaa_histogram.png"
    if min_games == 7:
        return "eda_win_pct_ncaa_histogram_gte7games.png"
    return f"eda_win_pct_ncaa_histogram_min{min_games}games.png"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate NCAA women's soccer EDA histogram charts from teams.csv.",
    )
    p.add_argument(
        "--teams-csv",
        type=Path,
        default=DEFAULT_TEAMS_CSV,
        help=f"Path to teams.csv (default: {DEFAULT_TEAMS_CSV})",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_FIGURE_DIR,
        help=f"Directory for PNG output (default: {DEFAULT_FIGURE_DIR})",
    )
    p.add_argument(
        "--plots",
        nargs="+",
        choices=("ncaa-all", "ncaa-filtered", "games"),
        default=("ncaa-all", "ncaa-filtered", "games"),
        help="Which charts to write (default: all three).",
    )
    p.add_argument(
        "--filtered-min-games",
        type=int,
        default=7,
        help="Minimum games for the ncaa-filtered plot (default: 7).",
    )
    p.add_argument("--bins", type=int, default=25, help="Histogram bins for NCAA win %% (default: 25).")
    p.add_argument(
        "--no-highlight-half",
        action="store_true",
        help="Do not highlight the bin containing 0.500 on NCAA histograms.",
    )
    p.add_argument("--dpi", type=int, default=150, help="Figure DPI (default: 150).")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    setup_matplotlib_agg()

    df = load_teams(args.teams_csv)
    df = enrich_teams(df)

    out = args.output_dir
    highlight = not args.no_highlight_half

    if "ncaa-all" in args.plots:
        m = mask_valid_games(df, min_games=1)
        sub = df.loc[m]
        title = "NCAA-style winning percentage (teams with at least 1 game)"
        fig = plot_ncaa_win_pct_histogram(
            sub,
            bins=args.bins,
            highlight_half=highlight,
            title=title,
        )
        save_fig(fig, out / _ncaa_output_name(1), dpi=args.dpi)
        plt.close(fig)

    if "ncaa-filtered" in args.plots:
        mg = max(1, args.filtered_min_games)
        m = mask_valid_games(df, min_games=mg)
        sub = df.loc[m]
        title = f"NCAA-style winning percentage (teams with at least {mg} games, {args.bins} bins)"
        fig = plot_ncaa_win_pct_histogram(
            sub,
            bins=args.bins,
            highlight_half=highlight,
            title=title,
        )
        save_fig(fig, out / _ncaa_output_name(mg), dpi=args.dpi)
        plt.close(fig)

    if "games" in args.plots:
        m = mask_valid_games(df, min_games=1)
        sub = df.loc[m, "games"]
        fig = plot_games_played_histogram(
            sub,
            title="Distribution of games played per team-season",
        )
        save_fig(fig, out / "eda_games_played_histogram.png", dpi=args.dpi)
        plt.close(fig)


if __name__ == "__main__":
    main()
