"""Registry of unimplemented function contracts.

Members implement the body of these callables. Changing a signature requires a
team PR against `docs/contracts.md`.
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
from src.data.leakage import (
    assert_no_leakage,
    assert_split_is_time_ordered,
    drop_snapshot_value_columns,
)
from src.data.quality import generate_data_quality_report, write_data_manifest
from src.data.seasons import (
    assign_time_split,
    attach_next_value_target,
    build_player_season_table,
    export_player_season_split,
    filter_big5,
)
from src.data.tables import (
    load_appearances,
    load_clubs,
    load_games,
    load_players,
    load_table,
    load_transfers,
    load_valuations,
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
from src.modeling.importance import permutation_importance_table, select_scorecard_features
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
from src.optimize.simplex import simplex
from src.optimize.squad import (
    compare_mip_vs_greedy,
    export_squad,
    formulate_squad_mip,
    greedy_spend,
    solve_squad_mip,
)
from src.policy.bid import (
    compare_policies,
    decide_bid,
    epsilon_greedy,
    export_policy,
    greedy_by_undervalued,
    mdp_state,
    q_learning_update,
    reward_if_signed,
    simulate_transfer_window,
    ucb_action,
)
from src.search.xi import (
    apply_action,
    compare_search,
    export_best_xi,
    informed_search,
    initial_xi_state,
    is_goal,
    legal_actions,
    uninformed_search,
    xi_quality,
)

# (callable, owner, required parameter names)
STUBS: list[tuple[Callable[..., object], str, tuple[str, ...]]] = [
    (load_table, "shared", ("name",)),
    (load_appearances, "shared", ()),
    (load_games, "shared", ()),
    (load_players, "shared", ()),
    (load_valuations, "shared", ()),
    (load_clubs, "shared", ()),
    (load_transfers, "shared", ()),
    (filter_big5, "shared", ("appearances", "games")),
    (build_player_season_table, "shared", ("appearances", "games", "players", "clubs")),
    (attach_next_value_target, "shared", ("player_seasons", "valuations")),
    (assign_time_split, "shared", ("player_seasons",)),
    (export_player_season_split, "shared", ()),
    (drop_snapshot_value_columns, "shared", ("players",)),
    (assert_no_leakage, "shared", ("frame", "cutoff")),
    (assert_split_is_time_ordered, "shared", ("player_seasons",)),
    (generate_data_quality_report, "shared", ("player_seasons",)),
    (write_data_manifest, "shared", ()),
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
    (initial_xi_state, "member_c", ("shortlist_player_ids",)),
    (legal_actions, "member_c", ("state", "players")),
    (apply_action, "member_c", ("state", "action")),
    (is_goal, "member_c", ("state",)),
    (xi_quality, "member_c", ("state", "scores")),
    (uninformed_search, "member_c", ("state", "players", "scores")),
    (informed_search, "member_c", ("state", "players", "scores")),
    (compare_search, "member_c", ("uninformed_result", "informed_result")),
    (export_best_xi, "member_c", ()),
    (simplex, "member_c", ("objective", "constraints_a", "constraints_b")),
    (formulate_squad_mip, "member_c", ("candidates", "budget_eur")),
    (solve_squad_mip, "member_c", ("problem",)),
    (greedy_spend, "member_c", ("candidates", "budget_eur")),
    (compare_mip_vs_greedy, "member_c", ("mip", "greedy")),
    (export_squad, "member_c", ()),
    (mdp_state, "member_c", ("position_hole", "remaining_budget_eur")),
    (reward_if_signed, "member_c", ("player_row",)),
    (epsilon_greedy, "member_c", ("q_values", "state", "epsilon")),
    (ucb_action, "member_c", ("counts", "q_values", "state")),
    (q_learning_update, "member_c", ("q_table", "state", "action", "reward", "next_state")),
    (greedy_by_undervalued, "member_c", ("shortlist", "remaining_budget_eur")),
    (simulate_transfer_window, "member_c", ("shortlist", "budget_eur", "method")),
    (compare_policies, "member_c", ("results",)),
    (decide_bid, "member_c", ("player_id", "remaining_budget_eur", "position_hole")),
    (export_policy, "member_c", ()),
]
