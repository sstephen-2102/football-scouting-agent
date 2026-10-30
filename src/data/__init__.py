"""Data download, load, and player-season assembly."""

from src.data.download import download_kaggle_dataset
from src.data.leakage import (
    assert_no_leakage,
    assert_split_is_time_ordered,
    drop_snapshot_value_columns,
    forbidden_feature_columns,
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
from src.data.tables import (
    load_appearances,
    load_clubs,
    load_games,
    load_players,
    load_table,
    load_transfers,
    load_valuations,
    required_tables,
)

__all__ = [
    "assert_no_leakage",
    "assert_split_is_time_ordered",
    "assign_time_split",
    "attach_next_value_target",
    "build_player_season_table",
    "download_kaggle_dataset",
    "drop_snapshot_value_columns",
    "export_player_season_split",
    "filter_big5",
    "forbidden_feature_columns",
    "generate_data_quality_report",
    "load_appearances",
    "load_clubs",
    "load_games",
    "load_players",
    "load_table",
    "load_transfers",
    "load_valuations",
    "required_tables",
    "season_end_date",
    "write_data_manifest",
]
