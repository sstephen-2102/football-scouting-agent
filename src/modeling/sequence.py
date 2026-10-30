"""Member B — S1 LSTM or 1D-CNN on last-k appearances. Not a portrait CNN."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.common.config import RANDOM_STATE, SEQUENCE_WINDOW
from src.common.exceptions import not_built


def last_k_appearance_windows(
    appearances: pd.DataFrame,
    player_seasons: pd.DataFrame,
    k: int = SEQUENCE_WINDOW,
) -> tuple[Any, Any]:
    """Return (X, y) with X shaped (n, k, channels) for minutes, goals, assists, cards.

    Align y to the same test players as R1. Dates in each window must be ≤ season cutoff.
    """
    not_built("member_b", "last_k_appearance_windows")


def fit_lstm_value(
    x_train: Any,
    y_train: Any,
    dropout: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> Any:
    """LSTM on appearance windows. Dropout required. Batch norm optional."""
    not_built("member_b", "fit_lstm_value")


def fit_conv1d_value(
    x_train: Any,
    y_train: Any,
    dropout: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> Any:
    """1D-CNN on appearance windows (tabular sequences, not images)."""
    not_built("member_b", "fit_conv1d_value")


def compare_sequence_to_tabular(
    sequence_metrics: dict,
    tabular_metrics: dict,
) -> dict:
    """S1 vs R1 on the same test players. Negative results stay in scope."""
    not_built("member_b", "compare_sequence_to_tabular")
