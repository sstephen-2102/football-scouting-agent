# Frozen contracts

Change these only by PR after the whole team agrees. Assign real names to A/B/C after the instructor groups you.

## Shared freeze

| Item | Value |
|---|---|
| Universe | Big 5 leagues, player-seasons with ≥ 10 appearances, seasons 2018/19–2025/26 |
| Split | Time-based: train through 2022/23, validation 2023/24, test 2024/25. 2025/26 is descriptive only (season incomplete at the 6 July 2026 snapshot) |
| Seed | 42 |
| Regression target | log1p of next market value (EUR), from `player_valuations` after the season cutoff |
| Classification target | `undervalued`: actual next value is ≥ 20% above the regression prediction on **validation** residuals (threshold frozen on val, applied to test). Alternative reported: value-up vs value-down vs previous valuation |
| Clustering | Fit on **train** rolling stats only. No market-value column inside the cluster fit (value is used after the fact to profile clusters) |
| Ranking metrics (regression) | MAE and RMSE on EUR and on log scale; Spearman correlation |
| Ranking metrics (classification) | PR AUC, ROC AUC, Brier — not accuracy |
| Budget *B* | Next-window forecast of Big-5 total transfer-fee outlay (Member A), or a club-specific cap the user sets in the app |
| PEP 8 | ruff on CI; pytest on fixtures |

## Integration files

| Producer | Path | Consumers |
|---|---|---|
| Shared | `outputs/splits/player_season_split.parquet` | A, B, C |
| A | `outputs/clustering/archetype_labels.parquet`, `clustering_manifest.json` | B (optional feature), C (MIP groups) |
| A | `outputs/forecast/budget.json` | C (MIP and RL) |
| B | `outputs/experiments/best_params.json`, `model_comparison.csv`, `player_scores.parquet` (predicted value, P(undervalued)) | C |
| C | `outputs/search/best_xi.json`, `outputs/optimize/squad.csv`, `outputs/policy/q_table.json` | App |

## Module owners

| Member | Course weeks | Packages |
|---|---|---|
| A | 1, 4 unsupervised, 5 | `src/agent`, `src/cluster`, `src/forecast` |
| B | 3 metrics, 4 supervised, 6 | `src/modeling` |
| C | 2, 3 LP/MIP, 7 | `src/search`, `src/optimize`, `src/policy` |

Shared and frozen after Week 1: `src/data`, `src/common`.

## App pages

| Page | Owner |
|---|---|
| Overview (PEAS, disclaimer, headline numbers) | C + all |
| Archetype Lab | A |
| Burden / value forecast | A |
| Experiment Board | B |
| Search Lab (XI / shortlist) | C |
| Squad MIP + Bid policy / Screener | C |

## Git

Branches: `feature/a-archetypes-forecast`, `feature/b-value-models`, `feature/c-search-policy`.  
PRs into `main` only with tests green. Raw Kaggle files never committed.
