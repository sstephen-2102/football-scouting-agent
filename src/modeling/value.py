"""Member B — R0 ridge, R1 gradient boosting, C1 undervalued classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.common.config import RANDOM_STATE, TARGET_VALUE_UPLIFT
from src.common.exceptions import not_built


def fit_ridge_value(
    train: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "log_next_value",
) -> Any:
    """R0: ridge / OLS on season aggregates only. Return the fitted estimator."""
    not_built("member_b", "fit_ridge_value")


def fit_gradient_boosting_value(
    train: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "log_next_value",
    param_grid: dict | None = None,
    random_state: int = RANDOM_STATE,
) -> Any:
    """R1: gradient boosting with a validation grid. Freeze params on val, not test."""
    not_built("member_b", "fit_gradient_boosting_value")


def freeze_undervalued_threshold(
    val: pd.DataFrame,
    predicted_value_eur: str,
    actual_value_eur: str,
    uplift: float = TARGET_VALUE_UPLIFT,
) -> float:
    """Validation-only residual rule: actual ≥ (1 + uplift) × predicted. Never retune on test."""
    not_built("member_b", "freeze_undervalued_threshold")


def label_undervalued(
    frame: pd.DataFrame,
    predicted_value_eur: str,
    actual_value_eur: str,
    threshold: float,
) -> pd.Series:
    """Binary series aligned to `frame.index` using the frozen threshold."""
    not_built("member_b", "label_undervalued")


def fit_undervalued_classifier(
    train: pd.DataFrame,
    feature_columns: list[str],
    label_column: str = "undervalued",
    random_state: int = RANDOM_STATE,
) -> Any:
    """C1: classifier on R1 residuals / features. Return predict_proba-capable estimator."""
    not_built("member_b", "fit_undervalued_classifier")


def compare_value_models(results: dict[str, dict]) -> pd.DataFrame:
    """One row per setup (R0, R1, C1, S1) with the locked metrics."""
    not_built("member_b", "compare_value_models")


def predict_player_scores(
    model: Any,
    player_seasons: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Columns: player_id, season, pred_value, p_undervalued, optional archetype."""
    not_built("member_b", "predict_player_scores")


def export_player_scores() -> Path:
    """Write `outputs/experiments/player_scores.parquet` (pipeline: score)."""
    not_built("member_b", "export_player_scores")
