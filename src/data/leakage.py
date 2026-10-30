"""Leakage guards. Call these before any fit."""

from __future__ import annotations

from datetime import date

import pandas as pd

from src.common.config import FORBIDDEN_VALUE_COLUMNS
from src.common.exceptions import not_built


def forbidden_feature_columns() -> tuple[str, ...]:
    """Snapshot value columns that must never enter a same-window feature matrix."""
    return FORBIDDEN_VALUE_COLUMNS


def drop_snapshot_value_columns(players: pd.DataFrame) -> pd.DataFrame:
    """Remove current `market_value_in_eur` / `highest_market_value_in_eur`."""
    not_built("shared", "drop_snapshot_value_columns")


def assert_no_leakage(
    frame: pd.DataFrame,
    cutoff: date,
    date_column: str = "date",
    feature_columns: list[str] | None = None,
) -> None:
    """Raise LeakageError if any feature is dated after `cutoff` or is forbidden."""
    not_built("shared", "assert_no_leakage")


def assert_split_is_time_ordered(player_seasons: pd.DataFrame) -> None:
    """Raise LeakageError if val/test seasons overlap train or each other."""
    not_built("shared", "assert_split_is_time_ordered")
