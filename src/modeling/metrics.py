"""Member B — locked metrics. Accuracy is out of scope for the classifier."""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.common.exceptions import InvalidParameterError, not_built


def regression_metrics(
    y_true_eur: pd.Series,
    y_pred_eur: pd.Series,
) -> dict:
    """MAE and RMSE on EUR and on log1p; Spearman correlation. No R²-only report."""
    actual = np.asarray(y_true_eur, dtype=float)
    predicted = np.clip(np.asarray(y_pred_eur, dtype=float), 0, None)
    if len(actual) != len(predicted):
        raise InvalidParameterError("Actual and predicted must have the same length.")

    error = predicted - actual
    log_error = np.log1p(predicted) - np.log1p(actual)
    return {
        "mae_eur": float(np.mean(np.abs(error))),
        "rmse_eur": float(np.sqrt(np.mean(error**2))),
        "mae_log": float(np.mean(np.abs(log_error))),
        "rmse_log": float(np.sqrt(np.mean(log_error**2))),
        "spearman": float(spearmanr(predicted, actual).statistic),
        "n": len(actual),
    }


def classification_metrics(
    y_true: pd.Series,
    y_score: pd.Series,
) -> dict:
    """PR AUC, ROC AUC, Brier. Do not return accuracy as a headline metric."""
    not_built("member_b", "classification_metrics")
