#!/usr/bin/env python3
"""
Compare ``overall_record`` in teams.csv to W–L–T derived from contests_raw.csv.

Contests are filtered to the NCAA season window (Aug 1 – Jul 31) matching each
teams.csv ``season`` label. Duplicate (contest_id, team_id) rows are dropped
(keep first) before counting.

Commands
--------
  # All team-seasons where stated record != contests record (stdout table)
  python scripts/compare_team_records_to_contests.py compare

  # Write mismatches to CSV
  python scripts/compare_team_records_to_contests.py compare --output data/record_mismatches.csv

  # Dump teams row(s) and contest rows for one team (optional season filter on contests)
  python scripts/compare_team_records_to_contests.py inspect 481169
  python scripts/compare_team_records_to_contests.py inspect 481169 --season 2019-20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from ncaa_wsoc.config import DEFAULT_CONTESTS_RAW_CSV, DEFAULT_TEAMS_CSV
from ncaa_wsoc.contests import (
    aggregate_wlt_for_season,
    contests_rows_for_team,
    format_wlt,
    prepare_contests,
    stated_tuple_from_teams_row,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teams-csv", type=Path, default=DEFAULT_TEAMS_CSV)
    p.add_argument("--contests-csv", type=Path, default=DEFAULT_CONTESTS_RAW_CSV)

    sub = p.add_subparsers(dest="command", required=True)

    pc = sub.add_parser("compare", help="List team-seasons where records disagree")
    pc.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Write mismatch rows to this CSV path.",
    )
    pc.add_argument(
        "--all-rows",
        action="store_true",
        help="Include every team-season (not only mismatches); still useful with -o for full join.",
    )

    pi = sub.add_parser("inspect", help="Print teams row(s) and contest rows for one team_id")
    pi.add_argument("team_id", type=int)
    pi.add_argument(
        "--season",
        default=None,
        help="If set, limit contests to this teams.csv season label (e.g. 2019-20).",
    )

    return p


def run_compare(args: argparse.Namespace) -> int:
    teams = pd.read_csv(args.teams_csv)
    contests_raw = pd.read_csv(args.contests_csv)
    prepared = prepare_contests(contests_raw)

    rows_out: list[dict] = []
    for season in sorted(teams["season"].dropna().unique()):
        counts, unknown = aggregate_wlt_for_season(prepared, season)
        unk_n = unknown.groupby("team_id").size().rename("n_unparsed_results")
        ts = teams[teams["season"] == season].copy()
        merged = ts.merge(counts, on="team_id", how="left")
        merged = merged.merge(unk_n, on="team_id", how="left")
        merged["n_unparsed_results"] = merged["n_unparsed_results"].fillna(0).astype(int)
        for c in ("wins_c", "losses_c", "ties_c", "games_c"):
            if c not in merged.columns:
                merged[c] = 0
        merged[["wins_c", "losses_c", "ties_c", "games_c"]] = merged[
            ["wins_c", "losses_c", "ties_c", "games_c"]
        ].fillna(0).astype(int)

        for _, r in merged.iterrows():
            stated = stated_tuple_from_teams_row(r["overall_record"])
            if stated is None:
                rows_out.append(
                    {
                        "team_id": r["team_id"],
                        "season": season,
                        "name": r.get("name"),
                        "record_teams": r["overall_record"],
                        "record_contests": None,
                        "wins_teams": None,
                        "losses_teams": None,
                        "ties_teams": None,
                        "wins_contests": int(r["wins_c"]),
                        "losses_contests": int(r["losses_c"]),
                        "ties_contests": int(r["ties_c"]),
                        "games_contests": int(r["games_c"]),
                        "n_unparsed_results": int(r["n_unparsed_results"]),
                        "mismatch_reason": "unparseable_teams_record",
                    }
                )
                continue
            sw, sl, st = stated
            cw, cl, ct = int(r["wins_c"]), int(r["losses_c"]), int(r["ties_c"])
            record_c = format_wlt(cw, cl, ct)
            match = (sw, sl, st) == (cw, cl, ct)
            row_dict = {
                "team_id": r["team_id"],
                "season": season,
                "name": r.get("name"),
                "record_teams": format_wlt(sw, sl, st),
                "record_contests": record_c,
                "wins_teams": sw,
                "losses_teams": sl,
                "ties_teams": st,
                "wins_contests": cw,
                "losses_contests": cl,
                "ties_contests": ct,
                "games_contests": int(r["games_c"]),
                "n_unparsed_results": int(r["n_unparsed_results"]),
                "mismatch_reason": "" if match else "record_mismatch",
            }
            if args.all_rows or not match:
                rows_out.append(row_dict)

    out = pd.DataFrame(rows_out)
    if out.empty:
        print("No rows to report.", file=sys.stderr)
        return 0

    if not args.all_rows:
        out = out[out["mismatch_reason"] != ""]

    cols = [
        "team_id",
        "season",
        "name",
        "record_teams",
        "record_contests",
        "wins_teams",
        "losses_teams",
        "ties_teams",
        "wins_contests",
        "losses_contests",
        "ties_contests",
        "games_contests",
        "n_unparsed_results",
        "mismatch_reason",
    ]
    out = out[[c for c in cols if c in out.columns]]

    if args.output:
        out.to_csv(args.output, index=False)
        print(f"Wrote {len(out)} rows to {args.output.resolve()}", file=sys.stderr)

    with pd.option_context("display.max_rows", 200, "display.width", 200, "display.max_colwidth", 20):
        print(out.to_string(index=False))
    return 0


def run_inspect(args: argparse.Namespace) -> int:
    teams = pd.read_csv(args.teams_csv)
    contests_raw = pd.read_csv(args.contests_csv)
    prepared = prepare_contests(contests_raw)

    tid = args.team_id
    trows = teams[teams["team_id"] == tid]
    print("=== teams.csv row(s) ===")
    if trows.empty:
        print(f"No team_id={tid} in {args.teams_csv}")
    else:
        print(trows.to_string(index=False))

    print("\n=== contests_raw.csv row(s) (all raw rows; comparison uses deduped contest_id+team_id) ===")
    if args.season:
        print(f"(date filter: season {args.season!r} -> Aug–Jul window)")
    raw_all = contests_raw[contests_raw["team_id"] == tid].copy()
    raw_all["_date"] = pd.to_datetime(raw_all["date"], format="%m/%d/%Y", errors="coerce")
    if args.season:
        from ncaa_wsoc.contests import season_date_bounds

        lo, hi = season_date_bounds(args.season)
        raw_all = raw_all[(raw_all["_date"] >= lo) & (raw_all["_date"] <= hi)]
    nuniq = raw_all.groupby(["contest_id", "team_id"]).ngroups
    raw_all = raw_all.sort_values(["_date", "contest_id"])
    raw_all = raw_all.drop(columns=["_date"])
    print(f"Rows: {len(raw_all)}; unique (contest_id, team_id): {nuniq}")
    if raw_all.empty:
        print("(none)")
    else:
        print(raw_all.to_string(index=False))

    print("\n=== Deduped view (same rows the compare command counts) ===")
    deduped = contests_rows_for_team(prepared, tid, season=args.season)
    if deduped.empty:
        print("(none)")
    else:
        print(deduped.to_string(index=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "compare":
        return run_compare(args)
    if args.command == "inspect":
        return run_inspect(args)
    raise SystemExit(f"Unknown command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
