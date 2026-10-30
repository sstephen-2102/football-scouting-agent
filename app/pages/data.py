"""Data freeze, split, and leakage rules."""

import streamlit as st

from app.components.results import metric_row, show_warnings
from src.common.config import (
    BIG5_COMPETITION_IDS,
    FORBIDDEN_VALUE_COLUMNS,
    KAGGLE_DATASET,
    MIN_APPEARANCES,
    SNAPSHOT_DATE,
    TEST_SEASONS,
    TRAIN_SEASONS,
    VAL_SEASONS,
)


def render() -> None:
    st.title("Data Freeze")
    st.caption("Big-5 player-seasons, time split, and leakage rules. Raw CSVs stay out of Git.")
    metric_row(
        {
            "Dataset": KAGGLE_DATASET,
            "Snapshot": SNAPSHOT_DATE,
            "Train": ", ".join(str(year) for year in TRAIN_SEASONS),
            "Val / test": f"{VAL_SEASONS[0]} / {TEST_SEASONS[0]}",
        }
    )
    st.markdown(
        f"- Leagues: `{', '.join(BIG5_COMPETITION_IDS)}`\n"
        f"- Minimum appearances: **{MIN_APPEARANCES}**\n"
        f"- Forbidden same-window features: `{', '.join(FORBIDDEN_VALUE_COLUMNS)}`"
    )
    show_warnings(
        [
            "Time split by season, never a random row split.",
            "Targets come from `player_valuations` dated after the season cutoff.",
            "Do not train on all 1.89M appearance rows; aggregate to player-season first.",
        ]
    )
