# NCAA women’s soccer — analysis

This repository holds **analysis** (plots, models, notebooks) for data produced by the separate scraper project. It does **not** contain scraper code.

## Data

Manually copy CSV files from your scraper run into [`data/`](data/). CSV files are ignored by Git and are not uploaded to GitHub.

Expected files (column names are defined in the scraper’s `ncaa_wsoc/storage.py`):

- `teams.csv`
- `contests.csv`
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
| `notebooks/` | Jupyter notebooks |

Add a `scripts/` folder later if you prefer plain Python modules over notebooks.

## Related project

The scraper that produces these CSVs lives in the **`ncaa-wsoc-python`** repository (or your local clone). Keep scraper versions and analysis in sync by noting the scraper git commit and CLI options (`--season`, `--division`, `--output-dir`) when you copy new data.

## License

MIT (match the scraper project if you prefer a single license across repos).
