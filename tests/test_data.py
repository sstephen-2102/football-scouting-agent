"""Synthetic tests for the shared Transfermarkt data freeze (no Kaggle)."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from src.common.exceptions import DataLoadError, LeakageError
from src.data.leakage import (
    assert_no_leakage,
    assert_split_is_time_ordered,
    drop_snapshot_value_columns,
)
from src.data.quality import generate_data_quality_report, write_data_manifest
from src.data.seasons import (
    assign_time_split,
    attach_next_value_target,
    build_player_season_table,
    export_player_season_split,
    filter_big5,
    season_end_date,
)
from src.data.tables import load_table


def _toy_frames() -> dict[str, pd.DataFrame]:
    """Ten appearances per player-season so the default MIN_APPEARANCES=10 applies."""
    rows: list[dict] = []
    games: list[dict] = []
    game_id = 1
    for season, start in ((2021, date(2021, 8, 15)), (2022, date(2022, 9, 1)), (2023, date(2023, 10, 1))):
        for offset in range(10):
            day = start + timedelta(days=offset)
            games.append(
                {
                    "game_id": game_id,
                    "competition_id": "GB1",
                    "season": season,
                    "date": pd.Timestamp(day),
                }
            )
            for player_id, club_id in ((1, 10), (2, 20)):
                rows.append(
                    {
                        "appearance_id": f"{game_id}_{player_id}",
                        "game_id": game_id,
                        "player_id": player_id,
                        "player_club_id": club_id,
                        "date": pd.Timestamp(day),
                        "competition_id": "GB1",
                        "goals": 1,
                        "assists": 0,
                        "yellow_cards": 0,
                        "red_cards": 0,
                        "minutes_played": 90,
                    }
                )
            game_id += 1

    appearances = pd.DataFrame(rows)
    games_df = pd.DataFrame(games)
    players = pd.DataFrame(
        {
            "player_id": [1, 2],
            "position": ["Attack", "Midfield"],
            "sub_position": ["Centre-Forward", "Central Midfield"],
            "foot": ["right", "left"],
            "height_in_cm": [180, 175],
            "date_of_birth": pd.to_datetime(["1995-01-01", "1998-06-15"]),
            "market_value_in_eur": [50_000_000, 20_000_000],
            "highest_market_value_in_eur": [60_000_000, 25_000_000],
        }
    )
    clubs = pd.DataFrame(
        {
            "club_id": [10, 20],
            "name": ["Club A", "Club B"],
            "domestic_competition_id": ["GB1", "GB1"],
        }
    )
    valuations = pd.DataFrame(
        {
            "player_id": [1, 1, 1, 2, 2, 2],
            "date": pd.to_datetime(
                [
                    "2021-06-01",
                    "2022-07-15",
                    "2023-07-10",
                    "2021-06-01",
                    "2022-07-20",
                    "2023-07-12",
                ]
            ),
            "market_value_in_eur": [
                10_000_000,
                12_000_000,
                15_000_000,
                5_000_000,
                6_000_000,
                7_000_000,
            ],
        }
    )
    return {
        "appearances": appearances,
        "games": games_df,
        "players": players,
        "clubs": clubs,
        "valuations": valuations,
    }


def test_season_end_date() -> None:
    assert season_end_date(2022) == date(2023, 6, 30)


def test_filter_big5_uses_games_competition() -> None:
    frames = _toy_frames()
    frames["games"].loc[frames["games"]["season"] == 2023, "competition_id"] = "CL"
    filtered = filter_big5(frames["appearances"], frames["games"])
    assert set(filtered["competition_id"]) == {"GB1"}
    assert filtered["season"].max() == 2022


def test_build_and_split_are_time_ordered() -> None:
    frames = _toy_frames()
    table = build_player_season_table(
        frames["appearances"],
        frames["games"],
        frames["players"],
        frames["clubs"],
    )
    assert "market_value_in_eur" not in table.columns
    assert set(table["season"]) >= {2021, 2022, 2023}
    assert (table["n_appearances"] >= 10).all()

    with_target = attach_next_value_target(table, frames["valuations"])
    assert "log_next_value" in with_target.columns
    # 2023 has no post-cutoff valuation in the toy set, so those rows drop.
    assert set(with_target["season"]) == {2021, 2022}
    assert with_target["next_value_eur"].notna().all()

    split = assign_time_split(with_target)
    assert_split_is_time_ordered(split)
    train_seasons = set(split.loc[split["split"] == "train", "season"])
    other = set(split.loc[split["split"] != "train", "season"])
    assert train_seasons.isdisjoint(other)


def test_drop_snapshot_value_columns_and_leakage() -> None:
    players = _toy_frames()["players"]
    cleaned = drop_snapshot_value_columns(players)
    assert "market_value_in_eur" not in cleaned.columns
    assert "highest_market_value_in_eur" not in cleaned.columns

    with pytest.raises(LeakageError):
        assert_no_leakage(
            pd.DataFrame({"date": pd.to_datetime(["2024-01-01"])}),
            cutoff=date(2023, 6, 30),
        )
    with pytest.raises(LeakageError):
        assert_no_leakage(
            pd.DataFrame({"x": [1]}),
            cutoff=date(2023, 6, 30),
            feature_columns=["market_value_in_eur"],
        )


def test_quality_report_and_export(tmp_path) -> None:
    frames = _toy_frames()
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "outputs"
    raw_dir.mkdir()
    for name, frame in frames.items():
        file_name = "player_valuations" if name == "valuations" else name
        frame.to_csv(raw_dir / f"{file_name}.csv", index=False)

    path = export_player_season_split(raw_dir, outputs_dir=out_dir)
    assert path == out_dir / "splits" / "player_season_split.parquet"
    assert path.is_file()
    exported = pd.read_parquet(path)
    assert {"player_id", "season", "split", "log_next_value"} <= set(exported.columns)
    report = generate_data_quality_report(exported)
    assert report["n_rows"] == len(exported)
    assert report["n_players"] >= 1
    assert (out_dir / "tables" / "data_quality.json").is_file()

    manifest = write_data_manifest(raw_dir, outputs_dir=out_dir)
    assert manifest == out_dir / "data_manifest.json"
    assert manifest.is_file()


def test_load_table_missing(tmp_path) -> None:
    with pytest.raises(DataLoadError):
        load_table("appearances", tmp_path)
