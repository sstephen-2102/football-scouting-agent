"""Frozen configuration for the AAI-501 scouting agent."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
OUTPUTS = PROJECT_ROOT / "outputs"

KAGGLE_DATASET = "davidcariboo/player-scores"
SNAPSHOT_DATE = "2026-07-06"

BIG5_COMPETITION_IDS = ("GB1", "ES1", "IT1", "L1", "FR1")

TRAIN_SEASONS = (2018, 2019, 2020, 2021, 2022)
VAL_SEASONS = (2023,)
TEST_SEASONS = (2024,)
HOLDOUT_DESCRIPTIVE_SEASONS = (2025,)

MIN_APPEARANCES = 10
RANDOM_STATE = 42
TARGET_VALUE_UPLIFT = 0.20

KAGGLE_TABLES = (
    "appearances",
    "club_games",
    "clubs",
    "competitions",
    "countries",
    "game_events",
    "game_lineups",
    "games",
    "national_teams",
    "player_valuations",
    "players",
    "transfers",
)
