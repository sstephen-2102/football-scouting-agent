"""Member B — permutation importance and the 6–8 field scorecard."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.common.config import RANDOM_STATE
from src.common.exceptions import not_built


def permutation_importance_table(
    model: Any,
    features: pd.DataFrame,
    target: pd.Series,
    n_repeats: int = 10,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """One row per feature: importance mean and std. Scoring must match the model family."""
    not_built("member_b", "permutation_importance_table")


def select_scorecard_features(
    importance: pd.DataFrame,
    k: int = 8,
    ranking: list[str] | None = None,
) -> list[str]:
    """Keep 6–8 fields. If Member C's search ranking is ready, use `ranking` instead of top-k."""
    not_built("member_b", "select_scorecard_features")
