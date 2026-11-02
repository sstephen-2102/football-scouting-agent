"""Player-season table and the frozen time split. Shared Week 2 freeze."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.artifacts import artifact_path
from src.common.config import (
    BIG5_COMPETITION_IDS,
    DATA_RAW,
    HOLDOUT_DESCRIPTIVE_SEASONS,
    MIN_APPEARANCES,
    OUTPUTS,
    TEST_SEASONS,
    TRAIN_SEASONS,
    VAL_SEASONS,
)
from src.common.exceptions import DataValidationError, InvalidParameterError
from src.data.leakage import (
    assert_no_leakage,
    assert_split_is_time_ordered,
    drop_snapshot_value_columns,
)
from src.data.quality import generate_data_quality_report, write_data_manifest
from src.data.tables import (
    load_appearances,
    load_clubs,
    load_games,
    load_players,
    load_valuations,
)


def season_end_date(season: int) -> date:
    """Cutoff for start-year `season` (2022 → 2022/23) is 30 June of the next year."""
    return date(int(season) + 1, 6, 30)


def filter_big5(appearances: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Keep appearances in `BIG5_COMPETITION_IDS` only."""
    if not isinstance(appearances, pd.DataFrame) or not isinstance(games, pd.DataFrame):
        raise InvalidParameterError("appearances and games must be DataFrames.")
    if "game_id" not in appearances.columns:
        raise DataValidationError("appearances must include game_id.")
    if "game_id" not in games.columns:
        raise DataValidationError("games must include game_id.")

    game_cols = ["game_id"]
    for column in ("competition_id", "season", "date"):
        if column in games.columns:
            game_cols.append(column)
    game_info = games[game_cols].drop_duplicates(subset=["game_id"])
    merged = appearances.merge(game_info, on="game_id", how="left", suffixes=("", "_game"))

    if "competition_id_game" in merged.columns:
        competition = merged["competition_id_game"].fillna(merged.get("competition_id"))
    elif "competition_id" in merged.columns:
        competition = merged["competition_id"]
    else:
        raise DataValidationError("Neither games nor appearances provide competition_id.")
    merged["competition_id"] = competition

    if "season" not in merged.columns:
        raise DataValidationError("games must provide a season column.")
    merged["season"] = pd.to_numeric(merged["season"], errors="coerce").astype("Int64")

    big5 = merged[merged["competition_id"].isin(BIG5_COMPETITION_IDS)].copy()
    big5 = big5.dropna(subset=["player_id", "season"])
    big5["season"] = big5["season"].astype(int)
    big5["player_id"] = big5["player_id"].astype(int)
    return big5.reset_index(drop=True)


def build_player_season_table(
    appearances: pd.DataFrame,
    games: pd.DataFrame,
    players: pd.DataFrame,
    clubs: pd.DataFrame,
    min_appearances: int = MIN_APPEARANCES,
) -> pd.DataFrame:
    """One row per player-season with rolling stats. No market-value column."""
    big5 = filter_big5(appearances, games)
    if big5.empty:
        raise DataValidationError("No Big-5 appearances remain after filtering.")

    for column, default in (
        ("minutes_played", 0),
        ("goals", 0),
        ("assists", 0),
        ("yellow_cards", 0),
        ("red_cards", 0),
        ("player_club_id", pd.NA),
    ):
        if column not in big5.columns:
            big5[column] = default

    big5["minutes_played"] = pd.to_numeric(big5["minutes_played"], errors="coerce").fillna(0)
    for column in ("goals", "assists", "yellow_cards", "red_cards"):
        big5[column] = pd.to_numeric(big5[column], errors="coerce").fillna(0)

    grouped = (
        big5.groupby(["player_id", "season"], as_index=False)
        .agg(
            n_appearances=("game_id", "count"),
            minutes=("minutes_played", "sum"),
            goals=("goals", "sum"),
            assists=("assists", "sum"),
            yellow_cards=("yellow_cards", "sum"),
            red_cards=("red_cards", "sum"),
        )
    )
    grouped = grouped[grouped["n_appearances"] >= int(min_appearances)].copy()
    grouped = grouped[grouped["minutes"] > 0].copy()

    club_minutes = (
        big5.groupby(["player_id", "season", "player_club_id"], as_index=False)["minutes_played"]
        .sum()
        .rename(columns={"player_club_id": "club_id", "minutes_played": "club_minutes"})
    )
    club_minutes = club_minutes.sort_values(
        ["player_id", "season", "club_minutes"],
        ascending=[True, True, False],
    )
    primary_club = club_minutes.drop_duplicates(subset=["player_id", "season"], keep="first")
    grouped = grouped.merge(
        primary_club[["player_id", "season", "club_id"]],
        on=["player_id", "season"],
        how="left",
    )

    league = (
        big5.groupby(["player_id", "season"], as_index=False)["competition_id"]
        .agg(lambda values: values.mode().iloc[0] if len(values) else pd.NA)
    )
    grouped = grouped.merge(league, on=["player_id", "season"], how="left")

    for rate, source in (
        ("goals_per90", "goals"),
        ("assists_per90", "assists"),
        ("yellow_per90", "yellow_cards"),
        ("red_per90", "red_cards"),
    ):
        grouped[rate] = 90.0 * grouped[source] / grouped["minutes"]

    clean_players = drop_snapshot_value_columns(players)
    player_cols = [
        column
        for column in (
            "player_id",
            "position",
            "sub_position",
            "foot",
            "height_in_cm",
            "date_of_birth",
        )
        if column in clean_players.columns
    ]
    grouped = grouped.merge(clean_players[player_cols], on="player_id", how="left")

    if "date_of_birth" in grouped.columns and not grouped.empty:
        cutoffs = pd.to_datetime(
            grouped["season"].map(lambda year: season_end_date(int(year)))
        )
        dob = pd.to_datetime(grouped["date_of_birth"], errors="coerce")
        age_days = (cutoffs - dob).dt.days
        grouped["age"] = age_days.apply(
            lambda value: int(value / 365.25) if pd.notna(value) and value >= 0 else pd.NA
        )
    else:
        grouped["age"] = pd.NA

    if "club_id" in clubs.columns:
        club_cols = ["club_id"]
        for column in ("name", "club_name", "domestic_competition_id"):
            if column in clubs.columns:
                club_cols.append(column)
        grouped = grouped.merge(clubs[club_cols].drop_duplicates("club_id"), on="club_id", how="left")

    for column in ("position", "sub_position", "foot", "height_in_cm"):
        if column not in grouped.columns:
            grouped[column] = pd.NA

    ordered = [
        "player_id",
        "season",
        "club_id",
        "competition_id",
        "position",
        "sub_position",
        "n_appearances",
        "minutes",
        "goals",
        "assists",
        "yellow_cards",
        "red_cards",
        "goals_per90",
        "assists_per90",
        "yellow_per90",
        "red_per90",
        "age",
        "height_in_cm",
        "foot",
    ]
    return grouped[ordered].reset_index(drop=True)


def attach_next_value_target(
    player_seasons: pd.DataFrame,
    valuations: pd.DataFrame,
) -> pd.DataFrame:
    """Add next EUR value and `log_next_value` from the first valuation after cutoff."""
    if not isinstance(player_seasons, pd.DataFrame) or not isinstance(valuations, pd.DataFrame):
        raise InvalidParameterError("player_seasons and valuations must be DataFrames.")
    required = {"player_id", "season"}
    if missing := required - set(player_seasons.columns):
        raise DataValidationError(f"player_seasons missing {sorted(missing)}")
    for column in ("player_id", "date", "market_value_in_eur"):
        if column not in valuations.columns:
            raise DataValidationError(f"valuations missing column '{column}'.")

    values = valuations[["player_id", "date", "market_value_in_eur"]].copy()
    values["date"] = pd.to_datetime(values["date"], errors="coerce").astype("datetime64[ns]")
    values["market_value_in_eur"] = pd.to_numeric(values["market_value_in_eur"], errors="coerce")
    values = values.dropna(subset=["player_id", "date", "market_value_in_eur"])
    values["player_id"] = values["player_id"].astype(int)
    # merge_asof requires the `on` key to be globally sorted.
    values = values.sort_values(["date", "player_id"]).reset_index(drop=True)

    frame = player_seasons.copy().reset_index(drop=True)
    frame["player_id"] = frame["player_id"].astype(int)
    frame["cutoff"] = pd.to_datetime(
        frame["season"].map(lambda year: season_end_date(int(year)))
    ).astype("datetime64[ns]")
    frame = frame.sort_values(["cutoff", "player_id"]).reset_index(drop=True)

    prev_values = values.rename(
        columns={"date": "cutoff", "market_value_in_eur": "prev_value_eur"}
    )
    prev = pd.merge_asof(
        frame[["player_id", "season", "cutoff"]],
        prev_values,
        on="cutoff",
        by="player_id",
        direction="backward",
    )
    # First valuation strictly after cutoff: search forward from the next day.
    forward_keys = frame[["player_id", "season", "cutoff"]].copy()
    forward_keys["cutoff"] = forward_keys["cutoff"] + pd.Timedelta(days=1)
    forward_keys = forward_keys.sort_values(["cutoff", "player_id"]).reset_index(drop=True)
    next_values = values.rename(
        columns={"date": "cutoff", "market_value_in_eur": "next_value_eur"}
    )
    nxt = pd.merge_asof(
        forward_keys,
        next_values,
        on="cutoff",
        by="player_id",
        direction="forward",
    )

    frame = frame.drop(columns=["cutoff"])
    frame = frame.merge(prev[["player_id", "season", "prev_value_eur"]], on=["player_id", "season"], how="left")
    frame = frame.merge(nxt[["player_id", "season", "next_value_eur"]], on=["player_id", "season"], how="left")
    frame["log_next_value"] = np.log1p(frame["next_value_eur"])
    assert_no_leakage(
        frame,
        date(2099, 1, 1),
        feature_columns=[
            column
            for column in frame.columns
            if column not in {"next_value_eur", "log_next_value"}
        ],
    )

    descriptive_seasons = set(HOLDOUT_DESCRIPTIVE_SEASONS)
    keep_missing = frame["season"].isin(descriptive_seasons)
    if "split" in frame.columns:
        keep_missing = keep_missing | (frame["split"] == "descriptive")
    frame = frame[frame["next_value_eur"].notna() | keep_missing].copy()
    return frame.reset_index(drop=True)


def assign_time_split(
    player_seasons: pd.DataFrame,
    train_seasons: tuple[int, ...] = TRAIN_SEASONS,
    val_seasons: tuple[int, ...] = VAL_SEASONS,
    test_seasons: tuple[int, ...] = TEST_SEASONS,
    descriptive_seasons: tuple[int, ...] = HOLDOUT_DESCRIPTIVE_SEASONS,
) -> pd.DataFrame:
    """Add `split` in {train, val, test, descriptive}. Never a random row split."""
    if not isinstance(player_seasons, pd.DataFrame):
        raise InvalidParameterError("player_seasons must be a pandas DataFrame.")
    if "season" not in player_seasons.columns:
        raise DataValidationError("player_seasons must include season.")

    mapping = {}
    for year in train_seasons:
        mapping[int(year)] = "train"
    for year in val_seasons:
        mapping[int(year)] = "val"
    for year in test_seasons:
        mapping[int(year)] = "test"
    for year in descriptive_seasons:
        mapping[int(year)] = "descriptive"

    frame = player_seasons.copy()
    frame["split"] = frame["season"].map(mapping)
    frame = frame[frame["split"].notna()].copy()
    assert_split_is_time_ordered(frame)
    return frame.reset_index(drop=True)


def export_player_season_split(
    raw_dir: Path | None = None,
    outputs_dir: Path | None = None,
) -> Path:
    """Write `outputs/splits/player_season_split.parquet` (pipeline: build-seasons)."""
    directory = Path(raw_dir) if raw_dir is not None else DATA_RAW
    out_root = Path(outputs_dir) if outputs_dir is not None else OUTPUTS
    appearances = load_appearances(directory)
    games = load_games(directory)
    players = load_players(directory)
    clubs = load_clubs(directory)
    valuations = load_valuations(directory)

    player_seasons = build_player_season_table(appearances, games, players, clubs)
    player_seasons = attach_next_value_target(player_seasons, valuations)
    player_seasons = assign_time_split(player_seasons)

    report = generate_data_quality_report(player_seasons)
    tables_dir = out_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    (tables_dir / "data_quality.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_data_manifest(directory, outputs_dir=out_root)

    destination = (
        out_root / "splits" / "player_season_split.parquet"
        if outputs_dir is not None
        else artifact_path("split")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    player_seasons.to_parquet(destination, index=False)
    return destination
