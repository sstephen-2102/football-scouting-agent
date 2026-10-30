"""Load Transfermarkt CSVs from data/raw/. Shared; do not train here."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.config import KAGGLE_TABLES
from src.common.exceptions import not_built

REQUIRED_TABLES = (
    "appearances",
    "games",
    "players",
    "player_valuations",
    "clubs",
    "transfers",
)


def required_tables() -> tuple[str, ...]:
    """Tables needed before `build-seasons`. Lineups stay optional."""
    return REQUIRED_TABLES


def known_tables() -> tuple[str, ...]:
    """Every Kaggle file name in this snapshot, without the `.csv` suffix."""
    return KAGGLE_TABLES


def load_table(name: str, raw_dir: Path | None = None) -> pd.DataFrame:
    """Read `{name}.csv` from `data/raw/` (or `raw_dir`).

    Raises DataLoadError if the file is missing. Does not filter leagues.
    """
    not_built("shared", "load_table")


def load_appearances(raw_dir: Path | None = None) -> pd.DataFrame:
    """Player × game rows: minutes, goals, assists, cards, date, club, competition."""
    not_built("shared", "load_appearances")


def load_games(raw_dir: Path | None = None) -> pd.DataFrame:
    """Match rows including competition_id and date."""
    not_built("shared", "load_games")


def load_players(raw_dir: Path | None = None) -> pd.DataFrame:
    """Player attributes. Drop snapshot value columns before joining features."""
    not_built("shared", "load_players")


def load_valuations(raw_dir: Path | None = None) -> pd.DataFrame:
    """Historical Transfermarkt valuations. These are the only value labels."""
    not_built("shared", "load_valuations")


def load_clubs(raw_dir: Path | None = None) -> pd.DataFrame:
    """Club rows for domestic competition and names."""
    not_built("shared", "load_clubs")


def load_transfers(raw_dir: Path | None = None) -> pd.DataFrame:
    """Transfer rows. Use only as *past* context; never future fees as features."""
    not_built("shared", "load_transfers")
