"""Default paths and package constants."""

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR: Path = _REPO_ROOT / "data"
DEFAULT_TEAMS_CSV: Path = DATA_DIR / "teams.csv"
DEFAULT_FIGURE_DIR: Path = _REPO_ROOT / "figures"

__all__ = ["DATA_DIR", "DEFAULT_TEAMS_CSV", "DEFAULT_FIGURE_DIR", "repo_root"]


def repo_root() -> Path:
    return _REPO_ROOT
