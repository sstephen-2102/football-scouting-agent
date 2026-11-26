"""Member B — R0 ridge, R1 gradient boosting, C1 undervalued classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.common.config import OUTPUTS, RANDOM_STATE, TARGET_VALUE_UPLIFT
from src.common.exceptions import InvalidParameterError, not_built
from src.modeling.metrics import regression_metrics

POSITIONS = ["Attack", "Defender", "Goalkeeper", "Midfield", "Missing"]
NUMERIC_FEATURES = [
    "minutes",
    "goals_per90",
    "assists_per90",
    "yellow_per90",
    "red_per90",
    "age",
    "height_in_cm",
    "n_appearances",
]
SPLIT_TABLE = OUTPUTS / "splits" / "player_season_split.parquet"
SCORES_PATH = OUTPUTS / "experiments" / "player_scores.parquet"

def add_position_dummies(player_seasons: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with one 0/1 column per position, always the same five."""
    out = player_seasons.copy()
    position = out["position"].fillna("Missing")
    for name in POSITIONS:
        out[f"pos_{name}"] = (position == name).astype(int)
    return out


def ridge_feature_columns() -> list[str]:
    """The 13 R0 inputs: 8 numeric plus 5 position dummies."""
    return NUMERIC_FEATURES + [f"pos_{name}" for name in POSITIONS]

def persistence_baseline(player_seasons: pd.DataFrame) -> pd.Series:
    """Reference rule: next value = the last known value (prev_value_eur)."""
    return player_seasons["prev_value_eur"].astype(float)


def evaluate_on_split(
    table: pd.DataFrame,
    predicted_eur: pd.Series,
    split: str,
) -> dict:
    """Locked regression metrics for one split. Descriptive rows have no target."""
    if split == "descriptive":
        raise InvalidParameterError("The descriptive split has no target to score.")
    rows = table["split"] == split
    return regression_metrics(table.loc[rows, "next_value_eur"], predicted_eur[rows])

def fit_ridge_value(
    train: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "log_next_value",
) -> Any:
    """R0: ridge on season aggregates only. Return the fitted estimator."""
    if target_column in feature_columns:
        raise InvalidParameterError("The target cannot be one of the features.")
    if "split" in train.columns and not (train["split"] == "train").all():
        raise InvalidParameterError("Fit on split == 'train' rows only.")

    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("ridge", RidgeCV(alphas=np.logspace(-2, 3, 10))),
    ])
    model.fit(train[feature_columns], train[target_column])
    return model


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
    """Columns: player_id, season, split, pred_value, p_undervalued, optional archetype."""
    frame = add_position_dummies(player_seasons)
    log_prediction = model.predict(frame[feature_columns])

    scores = frame[["player_id", "season", "split"]].copy()
    scores["pred_value"] = np.clip(np.expm1(log_prediction), 0, None)
    scores["p_undervalued"] = np.nan
    return scores


def export_player_scores() -> Path:
    """Write `outputs/experiments/player_scores.parquet` (pipeline: score)."""
    table = add_position_dummies(pd.read_parquet(SPLIT_TABLE))
    columns = ridge_feature_columns()

    train = table[table["split"] == "train"]
    model = fit_ridge_value(train, columns)
    scores = predict_player_scores(model, table, columns)

    SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(SCORES_PATH, index=False)
    return SCORES_PATH