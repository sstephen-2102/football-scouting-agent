"""Member B — locked metrics. Accuracy is out of scope for the classifier."""

from __future__ import annotations

import pandas as pd

from src.common.exceptions import not_built


def regression_metrics(
    y_true_eur: pd.Series,
    y_pred_eur: pd.Series,
) -> dict:
    """MAE and RMSE on EUR and on log1p; Spearman correlation. No R²-only report."""
    not_built("member_b", "regression_metrics")


def classification_metrics(
    y_true: pd.Series,
    y_score: pd.Series,
) -> dict:
    """PR AUC, ROC AUC, Brier. Do not return accuracy as a headline metric."""
    not_built("member_b", "classification_metrics")
