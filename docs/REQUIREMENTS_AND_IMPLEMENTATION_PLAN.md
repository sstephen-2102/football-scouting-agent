# AAI-501 Requirements and Implementation Plan

**Course:** AAI-501 Introduction to AI  
**Working title:** A Football Scouting Agent on Transfermarkt  
**Team size:** 3  
**Dataset:** [davidcariboo/player-scores](https://www.kaggle.com/datasets/davidcariboo/player-scores) (CC0; freeze 6 July 2026)  
**Status:** signatures frozen in `src/`; names on A/B/C; implement function bodies next  
**Team:** Arunava Banerjee (A), Binish Hamdani (B), Sanjay Varghese Stephen (C)

This document is the team specification. It restates the AAI-501 brief, maps every weekly learning outcome onto **one** sporting-director agent, and assigns build work.

---

## 1. Course requirements

| Brief requirement | How this project meets it |
|---|---|
| AI-driven real-world problem | Club sporting director: who is undervalued, how to fill an XI, how to spend a transfer budget |
| Public dataset, not small, not made-up | Transfermarkt: 50k+ players, 1.89M appearances, 650k+ valuations, 175k+ transfers |
| Complexity comparable to a course assignment | Agent + clustering + search + forecast + LP/MIP + regression/classification + sequence DL + RL |
| At least two algorithm types, experimental comparison | Unsupervised archetypes, supervised value models, heuristic search, LP/MIP, forecasting, LSTM/1D-CNN, bandit/Q-learning — compared graphically |
| Tuning and/or multiple families | Validation-set grids; uninformed vs informed search; ε-greedy vs UCB vs Q vs greedy-by-value |
| Feature selection / interpret what was learned | Feature-subset search for a 6–8 field value scorecard; permutation importance |
| Visualization | Archetype profiles, forecast decomposition, ROC/PR or calibration, search nodes vs quality, squad allocation, regret |
| GitHub, README, PEP 8 | Public repo for this project: `football-scouting-agent` |
| Proposal end of Week 3 | Section 10.1 |
| ~10-page APA 7 paper + code, Week 7 | Section 10.2 |
| 20–30 min recording, equal speaking time | Section 10.3 |
| Do not reuse a problem already analyzed in *this* course | Football scouting is not an AAI-501 homework problem |

---

## 2. System requirement

### 2.1 Problem

Transfermarkt publishes noisy, lagged market values. Clubs cannot buy every high-value player. They need archetypes (“pressing forward”, “ball-progressing full-back”), a view of who looks cheap relative to on-pitch output, a legal starting XI, and a plan for a finite budget.

**Goal.** Design a scouting *agent* that, given a club, a season cutoff, and budget *B*, (a) assigns player archetypes, (b) estimates market value from appearances, (c) flags undervalued players, (d) searches a feasible XI, (e) allocates *B* with MIP, and (f) bids or passes under remaining funds.

Educational research, not betting advice and not a claim that Transfermarkt values are true prices.

### 2.2 Required behaviors

1. Cluster train players from rolling stats **without** using market value in the fit; then profile clusters by later value and position.
2. Predict next log market value; report EUR MAE and Spearman correlation on the locked test seasons.
3. Classify undervalued vs not, using a validation-frozen residual rule; rank with PR AUC.
4. Forecast weekly Big-5 goals (or monthly total market value) with STL + Holt-Winters/ARIMA; optionally map that to a budget *B*.
5. Search a starting XI: state = which of 11 slots are filled; compare BFS/random vs greedy/A* (heuristic = predicted XI quality).
6. Solve a MIP: maximize predicted squad quality subject to *B*, 11 starters, min/max per position.
7. Learn a bid/pass policy (bandit + Q-learning) vs greedy-by-undervalued-score under remaining *B*.
8. Streamlit and FastAPI both expose PEAS, disclaimers, scores, XI, allocation, and bid/pass. Comparison graphs are files under `outputs/figures/`, not notebooks.

### 2.3 Questions that must have graphs

1. Do archetypes differ in value and playing style?
2. Does informed search match or beat uninformed search for XI quality at a node budget?
3. Does a tuned GB/MLP beat linear regression on next value?
4. Does a 6–8 feature scorecard (search) match the full model?
5. Does MIP under *B* beat “buy the top predicted values until broke”?
6. Do bandits/Q-learning improve cumulative undervalued signings vs greedy, same *B*?

Negative results are in scope.

### 2.4 Out of scope

- Predicting match betting outcomes as the primary product
- CNN on player *portraits* to predict value (faces are not a scouting signal we will defend)
- Using `players.market_value_in_eur` or `highest_market_value_in_eur` as a feature for the same period’s target
- Training on all 1.89M appearance rows without aggregating to player-season
- DQN robotics, NLP
- Jupyter notebooks. Streamlit is the instructor-facing lab; FastAPI remains the product API.

---

## 3. Course-module map (all seven weeks)

| Module | Learning outcomes | In this project | Evidence |
|---|---|---|---|
| **1** | History of AI; agent, environment, 10 dimensions; AI-assisted lit review | Scouting agent PEAS in `src/agent/peas.py`; 10-dimension score; `python -m src.agent.literature` writes `docs/references.md` | `GET /v1/agent`; paper §1 |
| **2** | State-space graphs; uninformed vs informed search | XI construction as a graph. State = filled slots + remaining players. Actions = assign a shortlisted player to a legal position. BFS/random vs greedy/A* | `src/search/`; `POST /v1/search/xi` |
| **3** | LP/MIP; implement Simplex; ML metrics | Educational Simplex on a 2–3 position toy budget. MIP on the full XI + bench cap. Metrics as in `contracts.md` | `src/optimize/` |
| **4** | Supervised + unsupervised; ML lifecycle | K-means / GMM archetypes; ridge/GB value regression; undervalued classifier; time split; lifecycle figure | cluster + modeling exports |
| **5** | STL / classical decompose; ARIMA, Holt-Winters | Weekly Big-5 goals (or monthly summed valuations). Holdout MAPE. Feed a transformed forecast into *B* if it is stable; otherwise use *B* as a user input and still report the forecast | `src/forecast/` |
| **6** | CNN (images), RNN (sequences); dropout, batch norm | LSTM or 1D-CNN on last *k* appearances (minutes, goals, assists, cards) to predict next-window output or log value. Dropout. Paper: portraits exist but we refuse face→value; no ImageNet transfer | `src/modeling/sequence.py` |
| **7** | MDP, bandits, Bellman, Q-learning | State = (position hole, remaining *B*, archetype mix). Action = bid on next shortlist player or pass. Reward = predicted residual (undervaluation) if signed. ε-greedy vs UCB vs Q vs greedy | `src/policy/` |

**One pipeline:** observe stats (M1) → forecast load/budget (M5) → archetypes (M4) → score value (M4, M6) → search XI (M2) → allocate *B* (M3) → bid/pass (M7).

---

Function-level implementation definitions (inputs, algorithms, artifact schemas, tests) are in [`IMPLEMENTATION_DEFINITIONS.md`](IMPLEMENTATION_DEFINITIONS.md).

## 4. Implementation by member

If you are the only person from another USD project, still take a full vertical here. Nobody is “just slides.”

### Member A — Arunava Banerjee — Agent, archetypes, forecast

**Build**

- `src/agent/peas.py` and `docs/agent.md`: PEAS (performance = test Spearman + undervalued $ captured under *B*; environment = Transfermarkt snapshot, partially observable, stochastic, sequential, dynamic, mixed discrete/continuous, single-agent); 10 dimensions. Served at `GET /v1/agent`.
- `python -m src.agent.literature`: writes the human-verified APA list to `docs/references.md` (no notebooks).
- `src/cluster/`: k-means, GMM, agglomerative-on-subsample; `select_k` (silhouette, Davies–Bouldin, BIC); labels for train/val/test via nearest centroid / predict; **no value in the fit**.
- Profiles: goals/90, assists/90, cards/90, minutes, age, later median value.
- `src/forecast/`: weekly Big-5 goals from `appearances`×`games`; classical + STL; Holt-Winters and ARIMA; holdout; `outputs/forecast/budget.json`.
- HTTP: `GET /v1/archetypes`, `GET /v1/forecast` (503 until exports exist).
- Tests on synthetic player-seasons and a tiny weekly series.

**Hands B/C:** `archetype_labels.parquet`, `budget.json`.

**Paper / talk:** problem, PEAS, data freeze, clustering, forecast demo (~8–10 min).

### Member B — Binish Hamdani — Value models and sequences

**Build**

| Setup | What |
|---|---|
| R0 | Ridge / OLS on log value, season aggregates only |
| R1 | Gradient boosting, validation grid |
| C1 | Undervalued classifier on R1 residuals (threshold frozen on val) |
| S1 | LSTM or 1D-CNN on last 10 appearances vs R1 (same test players) |

- Time split from `outputs/splits/`.
- Permutation importance; 6–8 field subset consumed from C’s search if ready, else a simple top-k ablation.
- Export comparison tables/figures under `outputs/`; expose scores at `GET /v1/scores/{player_id}`.
- Tests: no future dates in features; val disjoint from test seasons.

**Hands C:** `player_scores.parquet` (player_id, season, pred_value, p_undervalued, archetype).

**Paper / talk:** metrics, R0 vs R1 vs S1, why not portrait CNNs (~8–10 min).

### Member C — Sanjay Varghese Stephen — Search, allocate, act

**Build**

- `src/search/`: XI state graph; uninformed vs informed; cap expansions; heuristic = sum of predicted values (or R1 quality) with position constraints.
- `src/optimize/simplex.py`: hand Simplex on a 2–3 variable budget LP (e.g. two positions).
- `src/optimize/squad.py`: PuLP MIP, max predicted quality s.t. *B*, one player per slot, min 1 GK, etc.
- `src/policy/`: bandit over archetypes/positions; Q-learning simulator sampling val shortlist; compare cumulative undervaluation captured.
- HTTP: `POST /v1/search/xi`, `POST /v1/squad/allocate`, `POST /v1/policy/bid`. Keep the FastAPI shell healthy.
- README, contribution appendix, slide master, recording logistics.

**Paper / talk:** search, MIP vs greedy, policy demo, limits (~8–10 min).

---

## 5. Week-by-week (aligned to AAI-501)

| Week | Course | A (Arunava) | B (Binish) | C (Sanjay) |
|---|---|---|---|---|
| **1** | Agents, lit review | PEAS in code; `python -m src.agent.literature` | Inspect schema; draft feature list | GitHub, FastAPI shell, instructor-facing README |
| **2** | Search | Clustering skeleton | Player-season table + split parquet | XI graph; BFS vs greedy on a 30-player toy |
| **3** | LP/MIP, metrics | k-selection MVP | Metric contract locked; R0 | Toy Simplex tests. **Submit 1–2 page proposal** |
| **4** | Sup + unsup lifecycle | Archetypes on train; `/v1/archetypes` live | R0/R1 + undervalued C1 | Search comparison on val shortlist |
| **5** | Time series | STL + HW/ARIMA; `budget.json` | Residuals vs time (no leakage) | MIP uses *B* |
| **6** | Deep learning | Paper clustering + forecast | S1 LSTM/1D-CNN vs R1; `/v1/scores` | Bid/search routes wired to scores |
| **7** | RL | Talk 8–10 min | Talk 8–10 min | Bandit + Q; talk 8–10 min; **submit paper, GitHub, recording, peer eval** |

No extensions on Week 7.

---

## 6. Definition of done

**Each member:** package + tests; pipeline command writes artifacts; owned HTTP routes return 200 on those artifacts (503 before); APA sections; equal minutes on the recording.

**Team:** public GitHub with this README; all seven modules evidenced as in §3; graphs for §2.3; snapshot date and leakage rules stated; contribution appendix.

---

## 7. Deliverables

### 7.1 Proposal (end of Week 3) — 1–2 pages

1. Topic statement (§2.1).
2. Algorithms and the agent to be built.
3. Course topics: agents, heuristic search, clustering, regression/classification, forecasting, LP/MIP, sequence deep learning, RL/bandits.
4. Expected behaviors (§2.2).
5. Issues: Transfermarkt values are not transaction prices; valuation dates are bursty; 2025/26 incomplete; leakage if current value is used as a feature.
6. APA 7 starter list (§8).
7. Dataset citation and 6 July 2026 freeze.

### 7.2 Paper (~10 pages + appendix)

| Approx. pages | Content | Lead |
|---|---|---|
| 1 | Purpose, agent, related work | Arunava + Sanjay |
| 1 | Data, freeze, leakage | Arunava |
| 1.5 | Clustering | Arunava |
| 1 | Forecast | Arunava |
| 2 | Value models, sequence DL, metrics | Binish |
| 1.5 | XI search | Sanjay |
| 1 | Simplex + MIP | Sanjay |
| 1.5 | MDP / bandits / Q | Sanjay |
| rest | Limits, next steps | all |
| Appendix | Contributions | each |

### 7.3 Presentation (20–30 min)

| Block | Speaker |
|---|---|
| Problem, PEAS, data | Arunava |
| Archetype Lab + forecast demo | Arunava |
| Experiment Board, sequence vs tabular | Binish |
| Search, MIP, bid policy demo | Sanjay |
| Limits, next step, close | Sanjay then Binish |

Upload the recording; put the URL on the title slide. One member submits Canvas.

---

## 8. Starter APA 7 references

*Football data from Transfermarkt* [Data set]. (2026). Kaggle. https://www.kaggle.com/datasets/davidcariboo/player-scores

Arthur, D., & Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding. In *Proceedings of the Eighteenth Annual ACM-SIAM Symposium on Discrete Algorithms* (pp. 1027–1035).

Dantzig, G. B. (1963). *Linear programming and extensions*. Princeton University Press.

Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). A formal basis for the heuristic determination of minimum cost paths. *IEEE Transactions on Systems Science and Cybernetics, 4*(2), 100–107.

He, M., Cachucho, R., & Knobbe, A. (2015). Football player’s performance and market value. In *Proceedings of the 2nd Workshop on Sports Analytics* (pp. 1–9).

Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and practice* (3rd ed.). OTexts. https://otexts.com/fpp3/

Müller, O., Simons, A., & Weinmann, M. (2017). Beyond crowd judgments: Data-driven estimation of market value in association football. *European Journal of Operational Research, 263*(2), 611–624. https://doi.org/10.1016/j.ejor.2017.05.005

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.

Russell, S., & Norvig, P. (2021). *Artificial intelligence: A modern approach* (4th ed.). Pearson.

Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction* (2nd ed.). MIT Press.

Add exact Kaggle version and retrieval date in the final paper.

---

## 9. Immediate next steps

1. Invite Binish and Arunava as GitHub collaborators on `football-scouting-agent`.
2. Keep the GitHub repository **public** (course requirement).
3. Run `python -m src.data.download` locally; fill `outputs/data_manifest.json`.
4. Freeze `outputs/splits/player_season_split.parquet` in Week 2.
5. Submit the Week 3 proposal from §2, §3, and §8.
