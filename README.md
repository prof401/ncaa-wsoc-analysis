# NCAA women’s soccer — analysis

This repository holds **analysis** (plots, models, notebooks) for data produced by the separate scraper project. It does **not** contain scraper code.

## Data

Manually copy CSV files from your scraper run into [`data/`](data/). CSV files are ignored by Git and are not uploaded to GitHub.

Expected files (column names are defined in the scraper’s `ncaa_wsoc/storage.py`):

- `teams.csv`
- `contests_raw.csv` (contest rows used by matchup / season reports; some scripts also accept a path via `--contests-csv`)
- `scoring_summary.csv` (optional, from the contest / box-score command)

## Setup

```bash
cd ncaa-wsoc-analysis
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Layout

| Path | Purpose |
|------|---------|
| `data/` | Local CSV exports (gitignored) |
| `figures/` | Generated charts (EDA histograms, etc.; default CLI output) |
| `notebooks/` | Jupyter notebooks |
| `scripts/` | Analysis CLIs (team season report, record compare, EDA) |
| `tests/` | Unit tests |

The `ncaa_wsoc` CLI writes PNGs under `figures/` by default (`python -m ncaa_wsoc.cli`).

## Match outcome / team season report

Poisson-based win/tie/loss probabilities, season highlights, and contest-level event weights live in [`ncaa_wsoc/matchup.py`](ncaa_wsoc/matchup.py). How to run the CLI and what each field means (with sample output) is in [`scripts/README.md`](scripts/README.md).

Quick start:

```bash
source .venv/bin/activate
python -m unittest tests.test_contests_scores -v
python scripts/team_season_report.py --team-id 603188 --season 2025-26
python scripts/team_season_report.py --contest-id 6407438
```

Requires `data/teams.csv` and `data/contests_raw.csv`.

## Related project

The scraper that produces these CSVs lives in the **`ncaa-wsoc-python`** repository (or your local clone). Keep scraper versions and analysis in sync by noting the scraper git commit and CLI options (`--season`, `--division`, `--output-dir`) when you copy new data.

## License

MIT (match the scraper project if you prefer a single license across repos).
