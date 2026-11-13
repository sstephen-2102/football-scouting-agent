"""Registry of unimplemented function contracts.

Members implement the body of these callables. Changing a signature requires a
team PR against `docs/contracts.md`. Shared data contracts in `src/data/` are
implemented and are no longer listed here. Implemented Member C helpers are
also omitted; only remaining export stubs stay registered.
"""

from __future__ import annotations

from collections.abc import Callable

from src.cluster.archetypes import (
    assign_archetypes,
    cluster_feature_matrix,
    export_archetypes,
    fit_agglomerative,
    fit_gmm,
    fit_kmeans,
    profile_archetypes,
    select_k,
)
from src.forecast.timeseries import (
    build_weekly_goals_series,
    decompose_stl,
    evaluate_forecast,
    export_forecast,
    fit_arima,
    fit_holt_winters,
    forecast_holdout,
    map_forecast_to_budget,
)
from src.modeling.importance import (
    permutation_importance_table,
    select_scorecard_features,
)
from src.modeling.metrics import classification_metrics, regression_metrics
from src.modeling.sequence import (
    compare_sequence_to_tabular,
    fit_conv1d_value,
    fit_lstm_value,
    last_k_appearance_windows,
)
from src.modeling.value import (
    compare_value_models,
    export_player_scores,
    fit_gradient_boosting_value,
    fit_ridge_value,
    fit_undervalued_classifier,
    freeze_undervalued_threshold,
    label_undervalued,
    predict_player_scores,
)
from src.optimize.squad import export_squad
from src.policy.bid import export_policy
from src.search.xi import export_best_xi

# (callable, owner, required parameter names)
STUBS: list[tuple[Callable[..., object], str, tuple[str, ...]]] = [
    (cluster_feature_matrix, "member_a", ("train",)),
    (fit_kmeans, "member_a", ("features", "k")),
    (fit_gmm, "member_a", ("features", "k")),
    (fit_agglomerative, "member_a", ("features", "k")),
    (select_k, "member_a", ("features",)),
    (assign_archetypes, "member_a", ("model", "player_seasons")),
    (profile_archetypes, "member_a", ("labeled",)),
    (export_archetypes, "member_a", ()),
    (build_weekly_goals_series, "member_a", ("appearances", "games")),
    (decompose_stl, "member_a", ("series",)),
    (fit_holt_winters, "member_a", ("train",)),
    (fit_arima, "member_a", ("train",)),
    (forecast_holdout, "member_a", ("model", "steps")),
    (evaluate_forecast, "member_a", ("y_true", "y_pred")),
    (map_forecast_to_budget, "member_a", ("forecast",)),
    (export_forecast, "member_a", ()),
    (regression_metrics, "member_b", ("y_true_eur", "y_pred_eur")),
    (classification_metrics, "member_b", ("y_true", "y_score")),
    (fit_ridge_value, "member_b", ("train", "feature_columns")),
    (fit_gradient_boosting_value, "member_b", ("train", "feature_columns")),
    (freeze_undervalued_threshold, "member_b", ("val", "predicted_value_eur", "actual_value_eur")),
    (label_undervalued, "member_b", ("frame", "predicted_value_eur", "actual_value_eur", "threshold")),
    (fit_undervalued_classifier, "member_b", ("train", "feature_columns")),
    (compare_value_models, "member_b", ("results",)),
    (predict_player_scores, "member_b", ("model", "player_seasons", "feature_columns")),
    (export_player_scores, "member_b", ()),
    (last_k_appearance_windows, "member_b", ("appearances", "player_seasons")),
    (fit_lstm_value, "member_b", ("x_train", "y_train")),
    (fit_conv1d_value, "member_b", ("x_train", "y_train")),
    (compare_sequence_to_tabular, "member_b", ("sequence_metrics", "tabular_metrics")),
    (permutation_importance_table, "member_b", ("model", "features", "target")),
    (select_scorecard_features, "member_b", ("importance",)),
    (export_best_xi, "member_c", ()),
    (export_squad, "member_c", ()),
    (export_policy, "member_c", ()),
]
