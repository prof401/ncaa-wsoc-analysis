"""NCAA women's soccer analysis utilities."""

__version__ = "0.1.0"

from ncaa_wsoc.config import DATA_DIR, DEFAULT_FIGURE_DIR, DEFAULT_TEAMS_CSV, repo_root
from ncaa_wsoc.io import load_teams
from ncaa_wsoc.records import enrich_teams, mask_valid_games, parse_overall_record

__all__ = [
    "__version__",
    "DATA_DIR",
    "DEFAULT_FIGURE_DIR",
    "DEFAULT_TEAMS_CSV",
    "repo_root",
    "load_teams",
    "enrich_teams",
    "mask_valid_games",
    "parse_overall_record",
]
