"""Leakage guards. Call these before any fit."""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from src.common.config import FORBIDDEN_VALUE_COLUMNS
from src.common.exceptions import InvalidParameterError, LeakageError


def forbidden_feature_columns() -> tuple[str, ...]:
    """Snapshot value columns that must never enter a same-window feature matrix."""
    return FORBIDDEN_VALUE_COLUMNS


def drop_snapshot_value_columns(players: pd.DataFrame) -> pd.DataFrame:
    """Remove current `market_value_in_eur` / `highest_market_value_in_eur`."""
    if not isinstance(players, pd.DataFrame):
        raise InvalidParameterError("players must be a pandas DataFrame.")
    drop = [column for column in FORBIDDEN_VALUE_COLUMNS if column in players.columns]
    return players.drop(columns=drop).copy()


def assert_no_leakage(
    frame: pd.DataFrame,
    cutoff: date,
    date_column: str = "date",
    feature_columns: list[str] | None = None,
) -> None:
    """Raise LeakageError if any feature is dated after `cutoff` or is forbidden."""
    if not isinstance(frame, pd.DataFrame):
        raise InvalidParameterError("frame must be a pandas DataFrame.")
    if feature_columns is not None:
        bad = [name for name in feature_columns if name in FORBIDDEN_VALUE_COLUMNS]
        if bad:
            raise LeakageError(f"Forbidden value columns in features: {bad}")
    if date_column not in frame.columns or frame.empty:
        return
    dates = pd.to_datetime(frame[date_column], errors="coerce")
    cutoff_ts = pd.Timestamp(cutoff)
    if isinstance(cutoff, datetime):
        cutoff_ts = pd.Timestamp(cutoff.date())
    leaked = dates.notna() & (dates > cutoff_ts)
    if bool(leaked.any()):
        raise LeakageError(
            f"{int(leaked.sum())} row(s) in '{date_column}' are after cutoff {cutoff}."
        )


def assert_split_is_time_ordered(player_seasons: pd.DataFrame) -> None:
    """Raise LeakageError if val/test seasons overlap train or each other."""
    if not isinstance(player_seasons, pd.DataFrame):
        raise InvalidParameterError("player_seasons must be a pandas DataFrame.")
    required = {"split", "season", "player_id"}
    missing = required - set(player_seasons.columns)
    if missing:
        raise LeakageError(f"player_seasons missing columns for split check: {sorted(missing)}")

    by_split = {
        name: set(group["season"].dropna().astype(int))
        for name, group in player_seasons.groupby("split", sort=False)
    }
    train = by_split.get("train", set())
    val = by_split.get("val", set())
    test = by_split.get("test", set())
    if train and val and max(train) >= min(val):
        raise LeakageError("Train seasons must all be before validation seasons.")
    if val and test and max(val) >= min(test):
        raise LeakageError("Validation seasons must all be before test seasons.")
    if train & val or train & test or val & test:
        raise LeakageError("Split seasons must be disjoint.")

    keys = player_seasons[["player_id", "season", "split"]].dropna()
    duplicated = keys.duplicated(subset=["player_id", "season"], keep=False)
    if bool(duplicated.any()):
        clash = keys.loc[duplicated].groupby(["player_id", "season"])["split"].nunique()
        if bool((clash > 1).any()):
            raise LeakageError("A player-season appears in more than one split.")
