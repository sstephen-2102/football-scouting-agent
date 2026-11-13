"""Load Transfermarkt CSVs from data/raw/. Shared; do not train here."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.config import DATA_RAW, KAGGLE_TABLES
from src.common.exceptions import DataLoadError, InvalidParameterError

REQUIRED_TABLES = (
    "appearances",
    "games",
    "players",
    "player_valuations",
    "clubs",
    "transfers",
)

_DATE_COLUMNS = {
    "appearances": ("date",),
    "games": ("date",),
    "players": ("date_of_birth",),
    "player_valuations": ("date",),
    "transfers": ("transfer_date", "contract_expiration_date"),
}


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
    if not isinstance(name, str) or not name:
        raise InvalidParameterError("name must be a non-empty string.")
    directory = Path(raw_dir) if raw_dir is not None else DATA_RAW
    path = directory / f"{name}.csv"
    if not path.is_file():
        raise DataLoadError(f"Missing Transfermarkt file: {path}")
    try:
        frame = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        raise DataLoadError(f"Cannot read {path}: {exc}") from exc
    for column in _DATE_COLUMNS.get(name, ()):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def load_appearances(raw_dir: Path | None = None) -> pd.DataFrame:
    """Player × game rows: minutes, goals, assists, cards, date, club, competition."""
    return load_table("appearances", raw_dir)


def load_games(raw_dir: Path | None = None) -> pd.DataFrame:
    """Match rows including competition_id and date."""
    return load_table("games", raw_dir)


def load_players(raw_dir: Path | None = None) -> pd.DataFrame:
    """Player attributes. Drop snapshot value columns before joining features."""
    return load_table("players", raw_dir)


def load_valuations(raw_dir: Path | None = None) -> pd.DataFrame:
    """Historical Transfermarkt valuations. These are the only value labels."""
    return load_table("player_valuations", raw_dir)


def load_clubs(raw_dir: Path | None = None) -> pd.DataFrame:
    """Club rows for domestic competition and names."""
    return load_table("clubs", raw_dir)


def load_transfers(raw_dir: Path | None = None) -> pd.DataFrame:
    """Transfer rows. Use only as *past* context; never future fees as features."""
    return load_table("transfers", raw_dir)
