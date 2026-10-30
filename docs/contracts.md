# Frozen contracts

Change these only by PR after the whole team agrees. Assign real names to A/B/C after the instructor groups you.

The product is a **standalone FastAPI service**. Domain code lives in `src/`. Batch jobs write `outputs/`. The HTTP process does not train.

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
| Interface | FastAPI (`python -m service.main`). No notebooks. No Streamlit. |
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
