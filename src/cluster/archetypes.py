"""Member A — unsupervised player archetypes. No market value in the fit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.common.config import CLUSTER_FEATURE_COLUMNS, RANDOM_STATE
from src.common.exceptions import not_built


def cluster_feature_columns() -> tuple[str, ...]:
    """Numeric columns used inside k-means / GMM / agglomerative."""
    return CLUSTER_FEATURE_COLUMNS


def cluster_feature_matrix(train: pd.DataFrame) -> pd.DataFrame:
    """Train-only numeric matrix. Drops forbidden value columns."""
    not_built("member_a", "cluster_feature_matrix")


def fit_kmeans(
    features: pd.DataFrame,
    k: int,
    random_state: int = RANDOM_STATE,
) -> Any:
    """k-means++ fit on `features`. Return the sklearn estimator."""
    not_built("member_a", "fit_kmeans")


def fit_gmm(
    features: pd.DataFrame,
    k: int,
    random_state: int = RANDOM_STATE,
) -> Any:
    """Gaussian mixture fit. Return the estimator (BIC lives in `select_k`)."""
    not_built("member_a", "fit_gmm")


def fit_agglomerative(
    features: pd.DataFrame,
    k: int,
    subsample: int = 2000,
    random_state: int = RANDOM_STATE,
) -> Any:
    """Ward / average linkage on a subsample, then nearest-centroid assign."""
    not_built("member_a", "fit_agglomerative")


def select_k(
    features: pd.DataFrame,
    k_range: tuple[int, int] = (4, 12),
    random_state: int = RANDOM_STATE,
) -> dict:
    """Silhouette, Davies–Bouldin, BIC over `k_range`. Return chosen k and table."""
    not_built("member_a", "select_k")


def assign_archetypes(model: Any, player_seasons: pd.DataFrame) -> pd.DataFrame:
    """Labels for train (fit) and val/test (predict or nearest centroid)."""
    not_built("member_a", "assign_archetypes")


def profile_archetypes(labeled: pd.DataFrame) -> pd.DataFrame:
    """Per-cluster goals/90, assists/90, cards/90, minutes, age, later median value."""
    not_built("member_a", "profile_archetypes")


def export_archetypes() -> Path:
    """Write `outputs/clustering/archetype_labels.parquet` (pipeline: cluster)."""
    not_built("member_a", "export_archetypes")
