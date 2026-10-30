# Data

**Dataset:** Football Data from Transfermarkt  
**Kaggle:** https://www.kaggle.com/datasets/davidcariboo/player-scores  
**Upstream:** https://github.com/dcaribou/transfermarkt-datasets  
**License:** CC0 (public domain)  
**Snapshot freeze:** 6 July 2026 (Kaggle pipeline paused; no 2026/27 squads)

Do not treat `players.market_value_in_eur` as a historical label. That column is the *current* snapshot value. All supervised targets come from `player_valuations` dated **on or before** the feature cutoff.

## Scale (publisher counts at freeze)

| Table | Grain | Approx. rows |
|---|---|---|
| `competitions` | league / tournament | 65 |
| `clubs` | club | 790+ |
| `players` | player | 50,000+ |
| `games` | match | 88,000+ |
| `appearances` | player × game | 1,890,000+ |
| `player_valuations` | player × valuation date | 650,000+ |
| `club_games` | club × game | 177,000+ |
| `game_events` | event | 1,270,000+ |
| `game_lineups` | lineup slot | 3,170,000+ |
| `transfers` | transfer | 175,000+ |
| `countries` | country | 124 |
| `national_teams` | national team | 124 |

Appearances run from **2012-07-03** through **2026-06-28**. Valuations through **2026-06-12**. Games through **2026-07-06**.

## Modeling universe (frozen)

Keep the project tractable and still far above the 1,000-row course floor.

- **Leagues:** Big 5 domestic — `GB1` (Premier League), `ES1` (LaLiga), `IT1` (Serie A), `L1` (Bundesliga), `FR1` (Ligue 1). Optional context: `CL` / `EL`.
- **Seasons:** 2018/19 through 2025/26 (incomplete 2025/26 is labeled as such).
- **Grain for ML:** player-season, only if the player has **≥ 10** appearances in that season for a Big-5 club.
- **Expected size:** on the order of 10,000–20,000 player-seasons (confirm in the data-quality export and put the exact count in the proposal).

`game_lineups` is optional (large). Start with `appearances` + `games` + `players` + `player_valuations` + `clubs` + `transfers`.

## Join keys

```text
competitions.competition_id  → games, clubs
games.game_id                → appearances, club_games, game_events
players.player_id            → appearances, player_valuations, transfers, game_events
clubs.club_id                → games.home_club_id / away_club_id, players.current_club_id
```

Useful `appearances` fields: `date`, `player_id`, `game_id`, `player_club_id`, `competition_id`, `goals`, `assists`, `yellow_cards`, `red_cards`, `minutes_played` (confirm names on download).

Useful `players` fields: `position`, `sub_position`, `foot`, `height_in_cm`, `date_of_birth`, `country_of_citizenship`, `current_club_id`. Ignore `image_url` / `url` for modeling. Do **not** predict value from portraits.

## Leakage rules

1. **Time split by season**, never a random row split.
2. Features for season *t* use only appearances and valuations with `date` before the season’s cutoff (end of that season for in-season stats; previous seasons for lagged value).
3. Target for season *t* is the first valuation **after** that cutoff (or log value at a fixed date such as 1 July following the season).
4. Do not use `highest_market_value_in_eur` as a feature for predicting current/next value (it can leak the peak that includes the target).
5. Do not use future transfers as features for a past valuation.

## Download (not committed)

```text
python -m src.data.download
```

Preferred source: Kaggle Hub dataset `davidcariboo/player-scores` into `data/raw/`. Alternative: the publisher DuckDB file (see upstream README). Record the Kaggle version number and SHA-256 of each CSV in `outputs/data_manifest.json`.
