#!/usr/bin/env python3
"""
Report why aggregate wins may not equal aggregate losses.

If every game appears on exactly two team-season rows (one win / one loss, or two ties),
then sum(wins) should equal sum(losses) and sum(ties) should be even. This script
prints global totals, optional group breakdowns, parse-status slices, and top imbalanced
groups so you can iterate on filters or data cleanup and rerun.

Examples
--------
  python scripts/analyze_win_loss_balance.py
  python scripts/analyze_win_loss_balance.py --min-games 7 --by season division
  python scripts/analyze_win_loss_balance.py --export data/wlt_balance_by_season.csv --by season
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ncaa_wsoc.balance import (
    breakdown_wlt,
    parse_status,
    summarize_by_parse_status,
    summarize_wlt,
    top_groups_by_abs_imbalance,
)
from ncaa_wsoc.io import load_teams
from ncaa_wsoc.records import enrich_teams, mask_valid_games


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teams-csv", type=Path, default=None, help="Path to teams.csv (default: package default)")
    p.add_argument(
        "--min-games",
        type=int,
        default=1,
        help="Only include rows with at least this many games (default: 1). Use 0 to include zero-game rows.",
    )
    p.add_argument(
        "--by",
        nargs="*",
        default=[],
        metavar="COLUMN",
        help="Optional groupby columns (e.g. season division). Omit for global summary only.",
    )
    p.add_argument(
        "--top",
        type=int,
        default=10,
        help="With --by, also print this many groups with largest |wins-losses| (default: 10).",
    )
    p.add_argument(
        "--export",
        type=Path,
        default=None,
        help="Write results to CSV: breakdown if --by is set, else a one-row global summary.",
    )
    p.add_argument(
        "--no-parse-status",
        action="store_true",
        help="Skip the parse-status summary block.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    raw = load_teams(args.teams_csv) if args.teams_csv is not None else load_teams()
    df = enrich_teams(raw)

    if args.min_games <= 0:
        mask = df["games"].notna()
    else:
        mask = mask_valid_games(df, min_games=args.min_games)

    print("=== Filter ===")
    print(f"min_games: {args.min_games}")
    print(f"rows included: {mask.sum()} / {len(df)}")

    print("\n=== Global totals (filtered) ===")
    g = summarize_wlt(df, mask=mask)
    print(g.to_string())
    print(
        "\nNote: If every result is double-counted across two teams, expect imbalance=0 "
        "and ties_parity_ok=True (each tie adds 1 tie to two teams)."
    )

    if not args.no_parse_status:
        print("\n=== By parse status (all rows, not min-games filtered) ===")
        tagged = parse_status(df)
        ps = summarize_by_parse_status(tagged)
        print(ps.to_string())
        print(
            "\nRows with zero_games (e.g. 0-0) or unparseable records do not affect W/L sums "
            "but explain dropped row counts when you require games >= 1."
        )

    if args.by:
        bd = breakdown_wlt(df, by=args.by, mask=mask)
        print(f"\n=== Breakdown by {args.by} ===")
        with pd.option_context("display.max_rows", 200, "display.width", 120):
            print(bd.to_string(index=False))
        top = top_groups_by_abs_imbalance(bd, k=args.top)
        print(f"\n=== Top {args.top} groups by |imbalance| ===")
        print(top.to_string(index=False))
        if args.export:
            bd.to_csv(args.export, index=False)
            print(f"\nWrote: {args.export.resolve()}")
    elif args.export:
        pd.DataFrame([g.to_dict()]).to_csv(args.export, index=False)
        print(f"\nWrote global summary: {args.export.resolve()}")


if __name__ == "__main__":
    main()
