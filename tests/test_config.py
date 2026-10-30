"""Smoke tests for frozen contracts."""

from src.common.config import (
    BIG5_COMPETITION_IDS,
    CLUSTER_FEATURE_COLUMNS,
    FORBIDDEN_VALUE_COLUMNS,
    MIN_APPEARANCES,
    RANDOM_STATE,
    SNAPSHOT_DATE,
    TEST_SEASONS,
    TRAIN_SEASONS,
    VAL_SEASONS,
    XI_SLOTS,
)


def test_big5_has_five_leagues() -> None:
    assert BIG5_COMPETITION_IDS == ("GB1", "ES1", "IT1", "L1", "FR1")


def test_split_is_time_ordered() -> None:
    assert max(TRAIN_SEASONS) < min(VAL_SEASONS) < min(TEST_SEASONS)


def test_minimum_appearances_and_seed() -> None:
    assert MIN_APPEARANCES >= 10
    assert RANDOM_STATE == 42
    assert SNAPSHOT_DATE == "2026-07-06"


def test_cluster_fit_excludes_market_value() -> None:
    assert "market_value_in_eur" not in CLUSTER_FEATURE_COLUMNS
    assert "market_value_in_eur" in FORBIDDEN_VALUE_COLUMNS
    assert len(XI_SLOTS) == 11
