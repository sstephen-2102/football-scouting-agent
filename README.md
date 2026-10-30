# Football Scouting Agent

University of San Diego M.S. Applied AI — **AAI-501** final team project (3 members).

A sporting-director agent for European football: it groups players into archetypes, forecasts market-value and scoring load, scores who is over- or under-valued, searches a feasible starting XI, allocates a transfer budget with LP/MIP, and learns a bid/pass policy under remaining funds.

**Data.** [Football Data from Transfermarkt](https://www.kaggle.com/datasets/davidcariboo/player-scores) (CC0), current to **6 July 2026**. Source: [dcaribou/transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets).

This is **not** the AAI-500 diabetes project. Raw CSVs stay out of Git (~900 MB).

## Documents

| File | What it is |
|---|---|
| [`docs/REQUIREMENTS_AND_IMPLEMENTATION_PLAN.md`](docs/REQUIREMENTS_AND_IMPLEMENTATION_PLAN.md) | Course brief mapped onto this system; week-by-week build |
| [`docs/contracts.md`](docs/contracts.md) | Frozen split, metrics, module owners, integration files |
| [`docs/DATA.md`](docs/DATA.md) | Tables, joins, leakage rules, snapshot date |

## Quick start

```text
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.data.download
streamlit run streamlit_app.py
```

`src.data.download` pulls the Kaggle snapshot into `data/raw/` (git-ignored). Tests use small fixtures only:

```text
python -m pytest tests/ -q
```

## Team (assign names in Week 1)

| Role | Owns |
|---|---|
| Member A | Agent framing, player archetypes (clustering), valuation/scoring forecasts |
| Member B | Value regression + over/under classification, sequence model, experiment board |
| Member C | Lineup search, MIP squad builder, bid/pass policy, README/slides glue |

## Scope in one sentence

Given a club, a season cutoff, and a transfer budget *B*, recommend who to shortlist, how to fill the XI, how to spend *B*, and whether to bid or pass on the next target — with graphs comparing every algorithm family the course teaches.
