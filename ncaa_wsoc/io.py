"""Load tabular data from CSV."""

from pathlib import Path

import pandas as pd

from ncaa_wsoc.config import DEFAULT_TEAMS_CSV


def load_teams(csv_path: Path | str | None = None, **read_csv_kwargs) -> pd.DataFrame:
    """
    Load the teams table.

    Parameters
    ----------
    csv_path
        Path to teams.csv. Defaults to ``DEFAULT_TEAMS_CSV``.
    **read_csv_kwargs
        Forwarded to :func:`pandas.read_csv`.
    """
    path = Path(csv_path) if csv_path is not None else DEFAULT_TEAMS_CSV
    return pd.read_csv(path, **read_csv_kwargs)
