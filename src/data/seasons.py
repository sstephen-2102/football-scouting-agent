"""Player-season table and the frozen time split. Shared Week 2 freeze."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from src.common.config import (
    HOLDOUT_DESCRIPTIVE_SEASONS,
    MIN_APPEARANCES,
    TEST_SEASONS,
    TRAIN_SEASONS,
    VAL_SEASONS,
)
from src.common.exceptions import not_built


def season_end_date(season: int) -> date:
    """Cutoff for start-year `season` (2022 → 2022/23) is 30 June of the next year."""
    return date(season + 1, 6, 30)


def filter_big5(appearances: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Keep appearances in `BIG5_COMPETITION_IDS` only."""
    not_built("shared", "filter_big5")


def build_player_season_table(
    appearances: pd.DataFrame,
    games: pd.DataFrame,
    players: pd.DataFrame,
    clubs: pd.DataFrame,
    min_appearances: int = MIN_APPEARANCES,
) -> pd.DataFrame:
    """One row per player-season with rolling stats. No market-value column."""
    not_built("shared", "build_player_season_table")


def attach_next_value_target(
    player_seasons: pd.DataFrame,
    valuations: pd.DataFrame,
) -> pd.DataFrame:
    """Add next EUR value and `log_next_value` from the first valuation after cutoff."""
    not_built("shared", "attach_next_value_target")


def assign_time_split(
    player_seasons: pd.DataFrame,
    train_seasons: tuple[int, ...] = TRAIN_SEASONS,
    val_seasons: tuple[int, ...] = VAL_SEASONS,
    test_seasons: tuple[int, ...] = TEST_SEASONS,
    descriptive_seasons: tuple[int, ...] = HOLDOUT_DESCRIPTIVE_SEASONS,
) -> pd.DataFrame:
    """Add `split` in {train, val, test, descriptive}. Never a random row split."""
    not_built("shared", "assign_time_split")


def export_player_season_split(raw_dir: Path | None = None) -> Path:
    """Write `outputs/splits/player_season_split.parquet` (pipeline: build-seasons)."""
    not_built("shared", "export_player_season_split")
