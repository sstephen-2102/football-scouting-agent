"""Member A — weekly Big-5 series, STL, Holt-Winters, ARIMA, budget B."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.common.exceptions import not_built


def build_weekly_goals_series(
    appearances: pd.DataFrame,
    games: pd.DataFrame,
) -> pd.Series:
    """Weekly summed Big-5 goals. DatetimeIndex at week end."""
    not_built("member_a", "build_weekly_goals_series")


def decompose_stl(series: pd.Series, period: int = 52) -> dict:
    """Classical STL pieces: trend, seasonal, resid, plus the original series."""
    not_built("member_a", "decompose_stl")


def fit_holt_winters(train: pd.Series, seasonal_periods: int = 52) -> Any:
    """Holt-Winters on the train window. Return a statsmodels result."""
    not_built("member_a", "fit_holt_winters")


def fit_arima(
    train: pd.Series,
    order: tuple[int, int, int] = (1, 1, 1),
    seasonal_order: tuple[int, int, int, int] | None = None,
) -> Any:
    """ARIMA or SARIMA on the train window."""
    not_built("member_a", "fit_arima")


def forecast_holdout(model: Any, steps: int) -> pd.Series:
    """Point forecast of `steps` weeks/months beyond the train cutoff."""
    not_built("member_a", "forecast_holdout")


def evaluate_forecast(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Holdout MAPE, MAE, RMSE. Keys: mape, mae, rmse, n."""
    not_built("member_a", "evaluate_forecast")


def map_forecast_to_budget(
    forecast: pd.Series,
    method: str = "scaled_fees",
) -> dict:
    """Optional map of the series into budget B. Flag the series if it is unstable."""
    not_built("member_a", "map_forecast_to_budget")


def export_forecast() -> Path:
    """Write `outputs/forecast/budget.json` (pipeline: forecast)."""
    not_built("member_a", "export_forecast")
