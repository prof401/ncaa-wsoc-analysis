# Scripts

Run these from the repo root with the project virtualenv active and CSVs present in [`data/`](../data/) (`teams.csv`, `contests_raw.csv`).

```bash
cd ncaa-wsoc-analysis
source .venv/bin/activate
```

## Team season / matchup report

[`team_season_report.py`](team_season_report.py) prints contest-derived records, Poisson attack/defense rates, win/tie/loss probabilities, and biggest win/loss highlights. It also supports contest-level pregame probabilities for future in-game event weighting.

### Unit tests (score parsing)

```bash
python -m unittest tests.test_contests_scores -v
```

Expected: 6 tests OK (`W`/`L`/`T` scores, OT suffix, bad input).

### Season report for one team

```bash
python scripts/team_season_report.py --team-id 603188 --season 2025-26
```

Example output:

```text
=== Lynchburg (2025-26) ===
Record (contest-derived): 10-6-3
Poisson rates: attack=2.263, defense=0.895, games=19, low_confidence=False
League baseline λ₀=1.477

vs league-average opponent (season-end ratings):
  P(win)=66.7%, P(tie)=19.6%, P(loss)=13.7%

Biggest win (raw margin)
  2025-10-04 vs Averett (W 9-0)
  margin=+9

Biggest loss (raw margin)
  2025-09-21 vs Emory (L 1-6)
  margin=-5

Biggest win (strength-adjusted)
  2025-10-04 vs Averett (W 9-0)
  margin=+9, surprise=+8.11

Biggest loss (strength-adjusted)
  2025-09-21 vs Emory (L 1-6)
  margin=-5, surprise=-4.35
```

| Field | Meaning |
|-------|---------|
| Record | W–L–T from bilateral, non-exhibition contests in the season window |
| attack / defense | Goals scored / conceded per game (season-end fit) |
| λ₀ | League mean goals per team per game |
| P(win/tie/loss) | Independent Poisson matchup probs (default: vs a median-strength opponent) |
| Biggest win/loss (raw) | Largest / most negative goal margin |
| strength-adjusted | Margin minus expected margin from attack rates (`surprise`) |

### Head-to-head vs a specific opponent

```bash
python scripts/team_season_report.py --team-id 603188 --season 2025-26 --vs-team-id 603525
```

Same report as above, but probabilities are vs that opponent instead of a league-average stand-in:

```text
vs Denison (season-end ratings):
  P(win)=43.3%, P(tie)=30.9%, P(loss)=25.7%
```

### Contest-level probabilities

Looks up both teams for a `contest_id`, uses **pregame** ratings by default (games before kickoff only), and prints event-impact weights.

```bash
python scripts/team_season_report.py --contest-id 6407438
```

Example output:

```text
=== contest_id 6407438 (pregame ratings) ===
Team A: Boise St. (602610)
  P(win)=51.5%, P(tie)=31.4%
Team B: San Jose St. (602809)
  P(win)=17.1%, P(tie)=31.4%
Sum: 100.0%
event_impact_weight(Boise St.): 0.485
event_impact_weight(San Jose St.): 1.329
```

| Field | Meaning |
|-------|---------|
| Team A / Team B | Stable order: lower `team_id` is A, higher is B |
| P(win) / P(tie) | Pregame three-way probabilities (sum to 100%) |
| event_impact_weight | Discount favorites (`p > 0.5` → `1 − p`); boost underdogs (`p ≤ 0.5` → `1 + (0.5 − p)`) |

Optional: use full-season ratings instead of pregame:

```bash
python scripts/team_season_report.py --contest-id 6407438 --ratings-as-of season_end
```

Paths default to `data/teams.csv` and `data/contests_raw.csv`. Override with `--teams-csv` / `--contests-csv` if needed.

## Other scripts

- [`compare_team_records_to_contests.py`](compare_team_records_to_contests.py) — reconcile `teams.csv` overall records with contest-derived W–L–T
- [`teams_eda_report.py`](teams_eda_report.py) / [`analyze_win_loss_balance.py`](analyze_win_loss_balance.py) — EDA / balance reports
