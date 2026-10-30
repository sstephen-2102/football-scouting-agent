# Implementation definitions

This is the implementer’s spec. Signatures are frozen in `src/` and catalogued in `docs/contracts.md`. **Fill the body. Do not rename arguments.** Change a return key or column name only by team PR.

Related files: `src/common/config.py` (constants), `src/common/signatures.py` (stub registry), `docs/DATA.md` (joins and leakage).

---

## 0. Rules that apply to every function

1. `random_state=42` wherever an estimator or shuffle exists.
2. Raise `DataLoadError` for missing/unreadable CSVs, `DataValidationError` for bad schema, `LeakageError` for future-dated or forbidden-value features, `InvalidParameterError` for illegal arguments.
3. Tests use **synthetic frames** (tens of rows). Do not hit Kaggle in pytest.
4. `export_*` functions create parent directories, write the contracted path, and return that `Path`.
5. Neither Streamlit nor FastAPI may fit models. They only read `outputs/`.
6. Season integer `Y` means season `Y/(Y+1)`. Cutoff is `season_end_date(Y)` → `date(Y+1, 6, 30)`.

**Build order:** shared `build-seasons` → A `cluster` and A `forecast` in parallel with B `score` (B needs the split; A cluster labels are optional on scores) → C `search-xi` / `allocate` / `policy` (need scores; allocate may use `budget.json`).

---

## 1. Shared data (`src/data/`)

Owner: the whole team. Freeze this in Week 2. Command: `python -m src.pipeline build-seasons`.

Confirm CSV column names on the first download. If a name differs, map it once in `load_*` and keep the internal names below.

### 1.1 Loaders — `src/data/tables.py`

| Function | Definition |
|---|---|
| `load_table(name, raw_dir=None)` | Read `{raw_dir or DATA_RAW}/{name}.csv`. Raise `DataLoadError` if missing. Return a DataFrame; do not filter leagues. |
| `load_appearances` … `load_transfers` | Call `load_table` with the matching name. `load_players` must still contain snapshot value columns so `drop_snapshot_value_columns` can remove them. |

Internal appearance columns after load: `date`, `player_id`, `game_id`, `player_club_id`, `competition_id`, `goals`, `assists`, `yellow_cards`, `red_cards`, `minutes_played`. Parse `date` as datetime.

### 1.2 Leakage — `src/data/leakage.py`

| Function | Definition |
|---|---|
| `drop_snapshot_value_columns(players)` | Drop `market_value_in_eur` and `highest_market_value_in_eur` if present. Return a copy. Never use those columns as features. |
| `assert_no_leakage(frame, cutoff, date_column="date", feature_columns=None)` | If `date_column` exists, raise `LeakageError` when any row has `date > cutoff`. If `feature_columns` is given, raise if any name is in `FORBIDDEN_VALUE_COLUMNS`. |
| `assert_split_is_time_ordered(player_seasons)` | Require column `split`. Raise `LeakageError` if any train `season` ≥ min val season, or any val `season` ≥ min test season, or a player-season appears in two splits. |

### 1.3 Player-season table — `src/data/seasons.py`

**`filter_big5(appearances, games)`**  
Join appearances to games on `game_id`. Keep rows whose competition is in `BIG5_COMPETITION_IDS` (`GB1`, `ES1`, `IT1`, `L1`, `FR1`). Prefer `games.competition_id`; fall back to `appearances.competition_id`. Attach `season` from `games.season` (start year). Drop rows with null `season` or `player_id`.

**`build_player_season_table(appearances, games, players, clubs, min_appearances=10)`**

1. `big5 = filter_big5(appearances, games)`.
2. Group by `(player_id, season)` (and the club with the most minutes that season as `club_id`).
3. Aggregates: `n_appearances`, `minutes`, `goals`, `assists`, `yellow_cards`, `red_cards`.
4. Keep rows with `n_appearances >= min_appearances`.
5. `goals_per90 = 90 * goals / minutes` (same for assists, yellow, red). If `minutes == 0`, drop the row.
6. Join `players` (after `drop_snapshot_value_columns`) for `position`, `sub_position`, `foot`, `height_in_cm`, `date_of_birth`.
7. `age` = completed years at `season_end_date(season)`.
8. Join `clubs` for `domestic_competition_id` / club name. Keep `competition_id` as that club’s Big-5 league.

**Required columns (no market value):**

```text
player_id, season, club_id, competition_id, position, sub_position,
n_appearances, minutes, goals, assists, yellow_cards, red_cards,
goals_per90, assists_per90, yellow_per90, red_per90,
age, height_in_cm, foot
```

**`attach_next_value_target(player_seasons, valuations)`**

For each row with cutoff `C = season_end_date(season)`:

- Target = first `player_valuations.market_value_in_eur` for that `player_id` with `date > C` (sort by date ascending).
- `next_value_eur` (float), `log_next_value = log1p(next_value_eur)`.
- Optional lag feature (allowed): last valuation with `date <= C` as `prev_value_eur`. Never a valuation after `C`.
- Drop rows with missing target (except `split == "descriptive"`).
- Call `assert_no_leakage` on the valuation slice used as features.

**`assign_time_split(...)`**  
Add `split`:

| `season` in | `split` |
|---|---|
| `TRAIN_SEASONS` (2018–2022) | `train` |
| `VAL_SEASONS` (2023) | `val` |
| `TEST_SEASONS` (2024) | `test` |
| `HOLDOUT_DESCRIPTIVE_SEASONS` (2025) | `descriptive` |

Drop any other season. Call `assert_split_is_time_ordered`.

**`export_player_season_split(raw_dir=None)`**  
Load required tables → build → attach target → split → `generate_data_quality_report` (write `outputs/tables/data_quality.json`) → `write_data_manifest` → write `outputs/splits/player_season_split.parquet`. Return that path.

### 1.4 Quality — `src/data/quality.py`

**`generate_data_quality_report(player_seasons)`** — JSON-serializable dict:

```text
n_rows, n_players, n_seasons,
counts_by_split, counts_by_league, counts_by_season,
missingness (column → count),
dropped_below_min_appearances (if known, else 0)
```

**`write_data_manifest(raw_dir=None)`** — `outputs/data_manifest.json`: Kaggle dataset id, snapshot `2026-07-06`, SHA-256 of each CSV, file sizes. Create `outputs/` if needed.

### 1.5 Shared done

- `python -m src.pipeline build-seasons` produces the parquet.
- pytest: synthetic 3-season frame; val/test seasons never appear in train; forbidden columns absent from features; `season_end_date(2022) == date(2023, 6, 30)`.
- Exact player-season count recorded for the Week 3 proposal.

---

## 2. Member A — agent, archetypes, forecast

Packages: `src/agent` (done), `src/cluster`, `src/forecast`.  
Commands: `python -m src.pipeline cluster`, `python -m src.pipeline forecast`.

PEAS, ten dimensions, and `python -m src.agent.literature` are already implemented. Do not redo them.

### 2.1 Clustering — `src/cluster/archetypes.py`

Fit **only** on `split == "train"`. **Never** pass a value column into the fit.

**`cluster_feature_matrix(train)`**  
Select `CLUSTER_FEATURE_COLUMNS`. Median-impute remaining nulls. `StandardScaler`. Raise `LeakageError` if a forbidden value column is present. Return a numeric DataFrame with the same index as `train`.

**`fit_kmeans(features, k, random_state=42)`**  
`sklearn.cluster.KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)`. Return the fitted estimator.

**`fit_gmm(features, k, random_state=42)`**  
`GaussianMixture(n_components=k, covariance_type="full", random_state=42)`. Return the fitted estimator.

**`fit_agglomerative(features, k, subsample=2000, random_state=42)`**  
If `len(features) > subsample`, sample `subsample` rows (`random_state`). Fit `AgglomerativeClustering(n_clusters=k, linkage="ward")`. Compute centroids of the subsample labels. Return an object with `predict(X)` = nearest-centroid assignment (so val/test can be labeled).

**`select_k(features, k_range=(4, 12), random_state=42)`**  
For each k in `range(lo, hi+1)` compute silhouette (higher better), Davies–Bouldin (lower better), and GMM BIC (lower better). **Chosen k** = argmax mean silhouette, ties broken by min Davies–Bouldin. Return:

```python
{"k": int, "scores": DataFrame[k, silhouette, davies_bouldin, bic], "rule": "max_silhouette"}
```

**`assign_archetypes(model, player_seasons)`**  
Build the scaled matrix for **all** rows with the **train** scaler (store scaler on the model or a small wrapper). `predict` labels. Return `player_seasons` plus `archetype` (int). Train rows must match `model.labels_` when the model is k-means/GMM.

**`profile_archetypes(labeled)`**  
Group by `archetype`. Means of `goals_per90`, `assists_per90`, `yellow_per90`, `red_per90`, `minutes`, `age`; median of `next_value_eur` **after** the fit (profiling only); mode of `position`. Write `outputs/figures/archetype_profiles.png` (grouped bars or parallel coordinates). Return the profile DataFrame.

**`export_archetypes()`**  
Load split parquet → `cluster_feature_matrix` on train → `select_k` → `fit_kmeans` (primary labels) and also fit GMM + agglomerative for the paper comparison → `assign_archetypes` → `profile_archetypes` → write:

- `outputs/clustering/archetype_labels.parquet` — `player_id, season, split, archetype` plus the profile join keys
- `outputs/clustering/clustering_manifest.json` — k, rule, silhouette table, scaler note, `random_state`

**Graph (Q1):** do archetypes differ in value and style? Profile plot + a small table of median `next_value_eur` by cluster.

### 2.2 Forecast — `src/forecast/timeseries.py`

Universe: Big-5 appearances only.

**`build_weekly_goals_series(appearances, games)`**  
Filter Big-5. Sum `goals` by week ending Sunday (`W-SUN`). `pd.Series` named `goals`, DatetimeIndex, missing weeks filled with `0`. Restrict to 2018-07-01 through snapshot.

**`decompose_stl(series, period=52)`**  
`statsmodels.tsa.seasonal.STL`. Return `{"observed", "trend", "seasonal", "resid"}` as Series. Save `outputs/figures/stl_decompose.png`.

**Train / holdout cut:** last week of June 2023 (end of train seasons) is the last train week. Weeks after that through June 2025 are holdout for MAPE. Do not tune on 2024/25 test weeks if you also report a second holdout; the paper may use val weeks (2023/24) for model choice and 2024/25 as confirmation.

**`fit_holt_winters(train, seasonal_periods=52)`**  
`ExponentialSmoothing(..., trend="add", seasonal="add", seasonal_periods=52).fit()`. If the series is too short, fall back to `seasonal=None` and set `unstable=True` in the export.

**`fit_arima(train, order=(1,1,1), seasonal_order=None)`**  
`ARIMA` or `SARIMAX`. Default `(1,1,1)` with optional `(1,1,1,52)`. Return the fitted result.

**`forecast_holdout(model, steps)`**  
Point forecast, index continuing the train index.

**`evaluate_forecast(y_true, y_pred)`**  
Align on index. `mape = mean(|y - yhat| / max(|y|, 1))` so zero-goal weeks do not explode. Also `mae`, `rmse`, `n`.

**`map_forecast_to_budget(forecast, method="scaled_fees")`**

- `scaled_fees`: `B = median(forecast) * scale`. Use `scale = 1.0` as a documented placeholder **or** scale so that the last train-year implied B matches a constant (state the constant in the JSON). This is a **research budget**, not a real transfer cap.
- If holdout MAPE > 50% or STL residual variance dominates, set `unstable: true` and `B_source: "user_input"`. Squad Builder then requires the user-supplied `budget_eur`.

Return `{"budget_eur": float, "unstable": bool, "B_source": "forecast"|"user_input", "mape": float, "method": str}`.

**`export_forecast()`**  
Build series → STL → fit HW and ARIMA on train → evaluate both → pick lower MAPE → `map_forecast_to_budget` → write `outputs/forecast/budget.json` and `outputs/forecast/series.csv`. Return the JSON path.

### 2.3 Member A done

- Both pipeline commands write their artifacts.
- `GET /v1/archetypes` and `GET /v1/forecast` return 200.
- Streamlit Archetype Lab and Forecast Lab show tables/figures, not only the missing banner.
- Tests: toy 40×7 numeric matrix for `select_k`; 3-year weekly series for MAPE keys.
- Paper: PEAS, freeze, clustering, forecast (~8–10 min).

---

## 3. Member B — value models and sequences

Package: `src/modeling`. Command: `python -m src.pipeline score`.  
Reads `outputs/splits/player_season_split.parquet`. Joins `archetype` if Member A’s parquet exists; otherwise omit the column.

**Feature list (season aggregates only for R0/R1/C1):**  
`minutes`, `goals_per90`, `assists_per90`, `yellow_per90`, `red_per90`, `age`, `height_in_cm`, `n_appearances`, plus one-hot `position` (and `archetype` if present). **Not** `next_value_eur`, `log_next_value`, `market_value_in_eur`, `highest_market_value_in_eur`. `prev_value_eur` is allowed only if dated ≤ cutoff (shared layer).

Drop `descriptive` rows from every fit and metric.

### 3.1 Metrics — `src/modeling/metrics.py`

**`regression_metrics(y_true_eur, y_pred_eur)`** — dict, no R²-only report:

```text
mae_eur, rmse_eur,
mae_log, rmse_log,          # on log1p of both
spearman, n
```

Clip negative EUR predictions to 0 before log metrics. Spearman via `scipy.stats.spearmanr`.

**`classification_metrics(y_true, y_score)`** — `y_score` is P(undervalued) in `[0,1]`:

```text
pr_auc, roc_auc, brier, n, prevalence
```

Do **not** return accuracy as a headline key.

### 3.2 Tabular models — `src/modeling/value.py`

**`fit_ridge_value(train, feature_columns, target_column="log_next_value")`**  
`Pipeline(StandardScaler, RidgeCV(alphas=logspace(-2,3,10)))` or `LinearRegression` as a documented sub-baseline. Fit on `split=="train"` only. Return the pipeline.

**`fit_gradient_boosting_value(...)`**  
`HistGradientBoostingRegressor` (or `GradientBoostingRegressor`). Grid on **validation** only, e.g. `max_depth ∈ {2,3,4}`, `learning_rate ∈ {0.05,0.1}`, `max_iter ∈ {100,200}`. Freeze the best val Spearman (or val RMSE on EUR). **Never** grid on test. Return the fitted model plus stash `best_params_` on the estimator.

**`freeze_undervalued_threshold(val, predicted_value_eur, actual_value_eur, uplift=0.20)`**  
On **val** rows only, compute `actual / predicted` where predicted > 0. The frozen rule is: label 1 iff `actual >= (1+uplift) * predicted`. Return `1+uplift` (i.e. `1.20`) as the multiplier used at apply time. Do not pick a different cutoff from test.

**`label_undervalued(frame, predicted_value_eur, actual_value_eur, threshold)`**  
Boolean/int series, same index. `1` if `actual >= threshold * predicted` and predicted > 0, else `0`.

**`fit_undervalued_classifier(train, feature_columns, label_column="undervalued")`**  
Labels on train must use the **val-frozen** threshold applied to R1 train predictions (or val labels only for the classifier train set — pick one, document it: recommended = score train with R1, apply frozen threshold, then fit classifier on those labels with the same features). `HistGradientBoostingClassifier` or logistic regression with `predict_proba`. Return the estimator.

**`compare_value_models(results)`**  
`results` maps setup name (`R0`,`R1`,`C1`,`S1`,`R1_scorecard`) to a metrics dict. Return a DataFrame, one row per setup, columns = metric keys.

**`predict_player_scores(model, player_seasons, feature_columns)`**  
Use R1 for `pred_value` (`expm1` of log pred, clip at 0) and C1 for `p_undervalued`. Columns:

```text
player_id, season, split, pred_value, p_undervalued, archetype?
```

**`export_player_scores()`**  
Load split → fit R0, R1, C1 → metrics on **test** → permutation importance + scorecard (see 3.4) → S1 if sequence code is ready → write:

- `outputs/experiments/player_scores.parquet`
- `outputs/experiments/model_comparison.csv`
- `outputs/experiments/best_params.json`
- `outputs/figures/r0_vs_r1.png` (test MAE/Spearman)
- `outputs/figures/pr_roc.png`

### 3.3 Sequences — `src/modeling/sequence.py`

Same **test player-seasons** as R1. No portraits.

**`last_k_appearance_windows(appearances, player_seasons, k=10)`**  
For each player-season, take the last `k` appearances with `date <= season_end_date(season)`, ordered oldest→newest. Channels: `minutes_played, goals, assists, yellow_cards, red_cards`. Pad the left with zeros if fewer than k. Return `X` shape `(n, k, 5)` and `y` (`log_next_value`) aligned to `player_seasons` index. Raise `LeakageError` if any window date is after cutoff.

**`fit_lstm_value(x_train, y_train, dropout=0.2)`**  
LSTM (or GRU) with dropout **required**. Batch norm optional. Small net (e.g. 32 units, 1–2 layers). CPU-ok. Return a fitted object with `predict(X) -> log value`. PyTorch or Keras; add the extra dependency in `requirements.txt` when you implement.

**`fit_conv1d_value(...)`**  
1D convolution over the time axis (not images). Dropout required. Same predict contract. Implement **at least one** of LSTM or Conv1D as S1; the other may stay a stub if the paper compares S1 vs R1 only.

**`compare_sequence_to_tabular(sequence_metrics, tabular_metrics)`**  
Dict with both metric blocks plus `winner` = `"S1"` / `"R1"` / `"tie"` on test Spearman. Negative S1 results stay in scope.

### 3.4 Importance — `src/modeling/importance.py`

**`permutation_importance_table(model, features, target, n_repeats=10)`**  
sklearn `permutation_importance` on **val** (not test) for the paper’s feature ranking. Scoring: `"neg_root_mean_squared_error"` for regressors. DataFrame: `feature, importance_mean, importance_std`.

**`select_scorecard_features(importance, k=8, ranking=None)`**  
If `ranking` (list of names from Member C) is provided and length ≥ 6, use `ranking[:k]`. Else take top `k` from `importance`, clipped to 6–8. Refit a ridge on that subset; export as `R1_scorecard` in the comparison table. Graph (Q4): scorecard vs full R1 on test Spearman/MAE.

### 3.5 Member B done

- `python -m src.pipeline score` writes `player_scores.parquet`.
- `GET /v1/scores/{id}` returns 200 with `pred_value` and `p_undervalued`.
- Tests: no feature date after cutoff; train/val/test seasons disjoint; metrics dict keys present on a 20-row toy.
- Paper: R0 vs R1 vs S1, why not portrait CNNs (~8–10 min).

---

## 4. Member C — search, Simplex/MIP, policy

Packages: `src/search`, `src/optimize`, `src/policy`.  
Commands: `search-xi`, `allocate`, `policy`. Need `player_scores.parquet`. Allocate may read `budget.json`.

Map `position` / `sub_position` onto `XI_SLOTS` and onto groups `{GK, DEF, MID, FWD}`:

| Group | Slots |
|---|---|
| GK | GK |
| DEF | RB, RCB, LCB, LB |
| MID | CDM, CM, CAM |
| FWD | RW, ST, LW |

A player is legal for a slot if their mapped group matches the slot’s group (CB may fill RCB or LCB, etc.). If mapping is ambiguous, prefer `sub_position`.

### 4.1 XI search — `src/search/xi.py`

State must be **hashable**. Recommended: `tuple[tuple[str, int | None], ...]` of length 11 (slot, player_id or None) plus `frozenset` remaining ids — or a frozen dataclass.

**`initial_xi_state(shortlist_player_ids, slots=None)`**  
All slots empty; remaining = unique ids in the shortlist (length 11–40). Default slots = `XI_SLOTS`.

**`legal_actions(state, players)`**  
List of `(slot, player_id)`: slot vacant, player still remaining, player legal for that slot. Empty if `is_goal`.

**`apply_action(state, action)`**  
Fill slot, remove player from remaining. Do not mutate the input.

**`is_goal(state)`**  
True iff every slot has a player_id.

**`xi_quality(state, scores)`**  
Sum of `pred_value` for assigned players (unfilled slots contribute 0). This is both the path-cost complement and the A* heuristic (admissible as an overestimate of remaining quality if you use `h = sum of top remaining legal values` — **recommended heuristic:** `h(n) = sum of the highest remaining `pred_value` that could still fill vacant slots without reuse). Document the heuristic in the paper.

**`uninformed_search(..., algorithm="bfs", expansion_cap=5000)`**  
`algorithm` in `{"bfs", "random"}`. Stop at `expansion_cap` node expansions. Track best goal (or best partial if none). Return:

```python
{
  "algorithm": str,
  "xi": {slot: player_id, ...},
  "quality": float,
  "nodes_expanded": int,
  "hit_cap": bool,
}
```

**`informed_search(..., algorithm="astar")`**  
`algorithm` in `{"greedy", "astar"}`. Greedy = expand max `xi_quality` (or max `h`). A* = maximize `g + h` with `g = current assigned quality`. Same return contract. Same `expansion_cap`.

**`compare_search(uninformed_result, informed_result)`**  
Dict with both qualities, node counts, `informed_minus_uninformed` quality, `nodes_ratio`. Save `outputs/figures/search_nodes_vs_quality.png` (two points or a small bar chart). Graph (Q2).

**`export_best_xi(club_id=None, season=None, shortlist_player_ids=None)`**  
If ids omitted, take top 30 test-split players by `p_undervalued` (or `pred_value`) from scores. Run BFS and A*. Write `outputs/search/best_xi.json` with both results plus `club_id`, `season`. Return that path.

Toy for Week 2: 30 players, 11 slots, cap 500 expansions — must run in tests without the Kaggle file.

### 4.2 Simplex — `src/optimize/simplex.py`

**`simplex(objective, constraints_a, constraints_b, maximize=True)`**  
Hand tableau Simplex (not `scipy.linprog` for the **educational** path). Size: 2–3 variables, ≤ 5 constraints. Example story: spend on `(x_fwd, x_def)` with `x_fwd + x_def ≤ B`, bounds, maximize `c·x`.

Use Bland’s rule (smallest subscript) to avoid cycling. Return:

```python
{"status": "optimal"|"infeasible"|"unbounded", "x": list[float], "objective_value": float, "tableau_history": list}
```

`tableau_history` may be truncated to the first 20 pivots. pytest: a 2-variable problem whose optimum is known by inspection.

You **may** also call `scipy.linprog` in a test to cross-check, but the function body must implement pivots.

### 4.3 Squad MIP — `src/optimize/squad.py`

Candidates = shortlist joined to scores with `pred_value` and a cost. Cost = `pred_value` (Transfermarkt-style ask) unless a fee column exists; document that cost is predicted value, not a real bid.

**`formulate_squad_mip(candidates, budget_eur, position_minima=None, n_starters=11)`**  
PuLP binary `x_i`. Objective: `max sum pred_value_i * x_i`. Constraints:

- `sum x_i == n_starters` (11)
- `sum cost_i x_i <= budget_eur`
- `sum_{i in GK} x_i >= 1` (and DEF≥4, MID≥3, FWD≥3 from `POSITION_MINIMA`)
- at most one player identity (already binary per player)

Return the `LpProblem`.

**`solve_squad_mip(problem)`**  
`status`, `player_ids`, `spend`, `quality` (objective). If infeasible, `player_ids=[]` and `status="infeasible"`.

**`greedy_spend(candidates, budget_eur, n_starters=11)`**  
Sort by `pred_value` descending. Take a player if they fit remaining budget and a remaining position-group quota. Stop at 11 or when none fit. Same return keys as the MIP dict.

**`compare_mip_vs_greedy(mip, greedy)`**  
`quality_gap = mip.quality - greedy.quality`, both spends, both id lists. Graph (Q5): `outputs/figures/mip_vs_greedy.png`. Negative gaps stay in scope.

**`export_squad(club_id=None, season=None, budget_eur=None)`**  
If `budget_eur` is None, read `outputs/forecast/budget.json` (`budget_eur` if `unstable` is false, else default `80_000_000`). Shortlist = top 40 test players by `pred_value`. Write `outputs/optimize/squad.csv` (one row per selected player: `player_id, slot_or_group, pred_value, cost, source=mip|greedy`) and `outputs/optimize/allocation_manifest.json`.

### 4.4 Policy — `src/policy/bid.py`

MDP for a **finite** window. Competing clubs are not agents.

**State.** `position_hole` (slot or group string), `remaining_budget_eur` discretized to bins of €5m (or €10m), optional `archetype_mix` tuple of counts. `mdp_state(...)` returns a hashable tuple.

**Actions.** `("bid", "pass")`.

**`reward_if_signed(player_row)`**  
`actual_or_pred_residual = next_value_eur - pred_value` if actual exists (val replay), else `0` if you only have predictions — **preferred on val:** `next_value_eur - pred_value` (positive = undervalued). If the agent **passes**, reward `0`. If it **bids** and `cost > remaining B`, treat as pass (reward 0, state unchanged). If it bids and can pay, sign: reward = residual, subtract cost, fill the hole (next hole = another vacant group or episode end).

**`epsilon_greedy(q_values, state, epsilon)`**  
With prob ε random action, else argmax Q(state, ·). Missing Q entries = 0.

**`ucb_action(counts, q_values, state, exploration=2.0)`**  
UCB1: `Q + exploration * sqrt(ln(N+1) / (n_a+1))`.

**`q_learning_update(q_table, state, action, reward, next_state, alpha=0.1, gamma=0.9)`**  
`Q[s,a] ← Q[s,a] + α (r + γ max_a' Q[s',a'] − Q[s,a])`. Terminal: `max Q[s']=0`. Mutate and return the table. Keys: `(state, action)`.

**`greedy_by_undervalued(shortlist, remaining_budget_eur)`**  
`"bid"` if the next player (scan by `p_undervalued` desc) has `cost <= remaining`, else `"pass"`.

**`simulate_transfer_window(shortlist, budget_eur, method, n_episodes=200)`**  
Replay shuffled val shortlists. `method` in `{"epsilon_greedy","ucb","q","greedy"}`. Return `{method, mean_return, std_return, n_episodes}`. ε = 0.1 for ε-greedy.

**`compare_policies(results)`**  
DataFrame of methods. Graph (Q6): `outputs/figures/policy_regret.png` (mean return vs greedy).

**`decide_bid(player_id, remaining_budget_eur, position_hole, q_table=None)`**  
Load q-table from artifact if `q_table` is None. Greedy on Q (ε=0) at `mdp_state(...)`. Return `{"action": "bid"|"pass", "player_id": int, "rationale": str}`. This is what FastAPI `POST /v1/policy/bid` and Streamlit Bid Policy must call once the artifact exists.

**`export_policy()`**  
Train Q on val episodes, evaluate four methods, write `outputs/policy/q_table.json` (JSON-serializable keys as strings) and `outputs/policy/policy_comparison.csv`.

### 4.5 Member C done

- Three pipeline commands write artifacts.
- POST routes and Streamlit Decision pages return real XI / squad / bid.
- Tests: 30-player toy search; 2-variable Simplex optimum; Q-update changes the table; `decide_bid` keys present.
- Paper/talk: search, MIP vs greedy, policy (~8–10 min). README glue, slide master, recording logistics.

---

## 5. UI wiring (after artifacts exist)

Do not train in the UI. Replace placeholder copy with artifact reads.

| Surface | Read | Show |
|---|---|---|
| Streamlit Archetype Lab / `GET /v1/archetypes` | `archetype_labels.parquet`, profile figure | cluster table + plot |
| Forecast Lab / `GET /v1/forecast` | `budget.json`, series | MAPE, B, STL figure |
| Experiment Board | `model_comparison.csv` | R0/R1/C1/S1 table, ROC/PR |
| Player Scores / `GET /v1/scores/{id}` | `player_scores.parquet` | pred_value, p_undervalued |
| Search Lab / `POST /v1/search/xi` | run search **or** return `best_xi.json` | XI + node counts |
| Squad Builder / `POST /v1/squad/allocate` | solve MIP with request `budget_eur` | selected squad vs greedy |
| Bid Policy / `POST /v1/policy/bid` | `decide_bid(...)` | action + rationale |

FastAPI 503 until the file exists remains correct.

---

## 6. Definition of done (team)

| Item | Evidence |
|---|---|
| Shared split | `outputs/splits/player_season_split.parquet` + quality JSON |
| A | cluster + forecast artifacts; Q1 figure |
| B | scores + comparison CSV; Q3–Q4 figures |
| C | best_xi, squad.csv, q_table; Q2, Q5, Q6 figures |
| Tests | pytest green on fixtures; ruff clean |
| Course | Week 3 proposal; Week 7 paper, GitHub, 20–30 min equal-time recording |

Names on A/B/C in `docs/contracts.md` are still TBD until the instructor groups you.
