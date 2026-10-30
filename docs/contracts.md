# Frozen contracts

Change these only by PR after the whole team agrees. Assign real names to A/B/C after the instructor groups you.

The product API is a **standalone FastAPI service**. The instructor-facing lab is Streamlit (`streamlit run streamlit_app.py`). Domain code lives in `src/`. Batch jobs write `outputs/`. Neither UI trains.

## Shared freeze

| Item | Value |
|---|---|
| Universe | Big 5 leagues, player-seasons with ≥ 10 appearances, seasons 2018/19–2025/26 |
| Split | Time-based: train through 2022/23, validation 2023/24, test 2024/25. 2025/26 is descriptive only |
| Seed | 42 |
| Regression target | log1p of next market value (EUR), from `player_valuations` after the season cutoff |
| Classification target | `undervalued`: actual next value is ≥ 20% above the regression prediction on **validation** residuals |
| Clustering | Fit on **train** rolling stats only. No market-value column inside the cluster fit |
| Ranking metrics (regression) | MAE and RMSE on EUR and on log scale; Spearman correlation |
| Ranking metrics (classification) | PR AUC, ROC AUC, Brier — not accuracy |
| Budget *B* | Forecast of Big-5 transfer-fee outlay, or a cap on `POST /v1/squad/allocate` |
| Interface | Streamlit lab for the instructor; FastAPI at `/` for the product API. No notebooks. |
| PEP 8 | ruff on CI; pytest on fixtures |

## Integration files

| Producer | Path | HTTP consumer |
|---|---|---|
| Shared `build-seasons` | `outputs/splits/player_season_split.parquet` | all later jobs |
| A `cluster` | `outputs/clustering/archetype_labels.parquet` | `GET /v1/archetypes` |
| A `forecast` | `outputs/forecast/budget.json` | `GET /v1/forecast`, allocate |
| B `score` | `outputs/experiments/player_scores.parquet` | `GET /v1/scores/{id}` |
| C `search-xi` | `outputs/search/best_xi.json` | `POST /v1/search/xi` |
| C `allocate` | `outputs/optimize/squad.csv` | `POST /v1/squad/allocate` |
| C `policy` | `outputs/policy/q_table.json` | `POST /v1/policy/bid` |

Missing files → HTTP **503** with `owner` and `producer`.

## Module owners

| Member | Course weeks | Packages | Routes |
|---|---|---|---|
| A | 1, 4 unsupervised, 5 | `src/agent`, `src/cluster`, `src/forecast` | `/v1/agent`, `/v1/archetypes`, `/v1/forecast` |
| B | 3 metrics, 4 supervised, 6 | `src/modeling` | `/v1/scores/{player_id}` |
| C | 2, 3 LP/MIP, 7 | `src/search`, `src/optimize`, `src/policy` | `/v1/search/xi`, `/v1/squad/allocate`, `/v1/policy/bid` |

Shared and frozen: `src/data`, `src/common`, `service/` shell.

## Git

Branches: `feature/a-archetypes-forecast`, `feature/b-value-models`, `feature/c-search-policy`.  
PRs into `main` only with tests green. Raw Kaggle files never committed.

## Function contracts

Signatures live in `src/` and are listed in `src/common/signatures.py`. **Algorithms, column names, artifact schemas, and done-checks** are in [`IMPLEMENTATION_DEFINITIONS.md`](IMPLEMENTATION_DEFINITIONS.md). Implement the **body**; do not rename arguments or change return types without a team PR. Unimplemented functions raise `NotBuiltError`. Helpers that are already filled in (`season_end_date`, `ten_environment_dimensions`, `cluster_feature_columns`, `xi_slots`, `bid_actions`, `required_tables`) are not stubs.

### Shared data

```python
load_table(name: str, raw_dir: Path | None = None) -> pd.DataFrame
load_appearances(raw_dir: Path | None = None) -> pd.DataFrame
load_games(raw_dir: Path | None = None) -> pd.DataFrame
load_players(raw_dir: Path | None = None) -> pd.DataFrame
load_valuations(raw_dir: Path | None = None) -> pd.DataFrame
load_clubs(raw_dir: Path | None = None) -> pd.DataFrame
load_transfers(raw_dir: Path | None = None) -> pd.DataFrame
filter_big5(appearances, games) -> pd.DataFrame
build_player_season_table(appearances, games, players, clubs, min_appearances=10) -> pd.DataFrame
attach_next_value_target(player_seasons, valuations) -> pd.DataFrame
assign_time_split(player_seasons, train_seasons=..., val_seasons=..., test_seasons=...) -> pd.DataFrame
export_player_season_split(raw_dir: Path | None = None) -> Path
drop_snapshot_value_columns(players) -> pd.DataFrame
assert_no_leakage(frame, cutoff, date_column="date", feature_columns=None) -> None
assert_split_is_time_ordered(player_seasons) -> None
generate_data_quality_report(player_seasons) -> dict
write_data_manifest(raw_dir: Path | None = None) -> Path
```

### Member A — archetypes and forecast

```python
cluster_feature_matrix(train) -> pd.DataFrame
fit_kmeans(features, k, random_state=42) -> object
fit_gmm(features, k, random_state=42) -> object
fit_agglomerative(features, k, subsample=2000, random_state=42) -> object
select_k(features, k_range=(4, 12), random_state=42) -> dict
assign_archetypes(model, player_seasons) -> pd.DataFrame
profile_archetypes(labeled) -> pd.DataFrame
export_archetypes() -> Path

build_weekly_goals_series(appearances, games) -> pd.Series
decompose_stl(series, period=52) -> dict
fit_holt_winters(train, seasonal_periods=52) -> object
fit_arima(train, order=(1, 1, 1), seasonal_order=None) -> object
forecast_holdout(model, steps) -> pd.Series
evaluate_forecast(y_true, y_pred) -> dict
map_forecast_to_budget(forecast, method="scaled_fees") -> dict
export_forecast() -> Path
```

### Member B — value models and sequences

```python
regression_metrics(y_true_eur, y_pred_eur) -> dict
classification_metrics(y_true, y_score) -> dict
fit_ridge_value(train, feature_columns, target_column="log_next_value") -> object
fit_gradient_boosting_value(train, feature_columns, target_column="log_next_value", param_grid=None) -> object
freeze_undervalued_threshold(val, predicted_value_eur, actual_value_eur, uplift=0.20) -> float
label_undervalued(frame, predicted_value_eur, actual_value_eur, threshold) -> pd.Series
fit_undervalued_classifier(train, feature_columns, label_column="undervalued") -> object
compare_value_models(results: dict[str, dict]) -> pd.DataFrame
predict_player_scores(model, player_seasons, feature_columns) -> pd.DataFrame
export_player_scores() -> Path

last_k_appearance_windows(appearances, player_seasons, k=10) -> tuple
fit_lstm_value(x_train, y_train, dropout=0.2) -> object
fit_conv1d_value(x_train, y_train, dropout=0.2) -> object
compare_sequence_to_tabular(sequence_metrics, tabular_metrics) -> dict
permutation_importance_table(model, features, target, n_repeats=10) -> pd.DataFrame
select_scorecard_features(importance, k=8, ranking=None) -> list[str]
```

### Member C — search, MIP, policy

```python
initial_xi_state(shortlist_player_ids, slots=None) -> object
legal_actions(state, players) -> list[tuple[str, int]]
apply_action(state, action: tuple[str, int]) -> object
is_goal(state) -> bool
xi_quality(state, scores) -> float
uninformed_search(state, players, scores, algorithm="bfs", expansion_cap=5000) -> dict
informed_search(state, players, scores, algorithm="astar", expansion_cap=5000) -> dict
compare_search(uninformed_result, informed_result) -> dict
export_best_xi(club_id=None, season=None, shortlist_player_ids=None) -> Path

simplex(objective, constraints_a, constraints_b, maximize=True) -> dict
formulate_squad_mip(candidates, budget_eur, position_minima=None, n_starters=11) -> object
solve_squad_mip(problem) -> dict
greedy_spend(candidates, budget_eur, n_starters=11) -> dict
compare_mip_vs_greedy(mip, greedy) -> dict
export_squad(club_id=None, season=None, budget_eur=None) -> Path

mdp_state(position_hole, remaining_budget_eur, archetype_mix=None) -> object
reward_if_signed(player_row) -> float
epsilon_greedy(q_values, state, epsilon, random_state=42) -> str
ucb_action(counts, q_values, state, exploration=2.0) -> str
q_learning_update(q_table, state, action, reward, next_state, alpha=0.1, gamma=0.9) -> dict
greedy_by_undervalued(shortlist, remaining_budget_eur) -> str
simulate_transfer_window(shortlist, budget_eur, method, n_episodes=200) -> dict
compare_policies(results: dict[str, dict]) -> pd.DataFrame
decide_bid(player_id, remaining_budget_eur, position_hole, q_table=None) -> dict
export_policy() -> Path
```
