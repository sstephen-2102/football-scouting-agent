# Football Scouting Agent

University of San Diego M.S. Applied AI — **AAI-501** final team project (3 members).

A **standalone HTTP service** for a sporting director: archetypes, market-value scores, starting-XI search, MIP budget allocation, and a bid/pass policy.

Python packages and a FastAPI process only. **No notebooks.**

**Data.** [Football Data from Transfermarkt](https://www.kaggle.com/datasets/davidcariboo/player-scores) (CC0), freeze **6 July 2026**. Raw CSVs are git-ignored.

## Architecture

```text
HTTP clients  →  service/ (FastAPI)  →  outputs/ artifacts
                                         ↑
python -m src.pipeline …   (batch jobs in src/)
```

The service never fits models. Members write domain code under `src/`, export artifacts, and the API reads them.

| Layer | Path | Role |
|---|---|---|
| Domain | `src/` | Clustering, forecast, models, search, MIP, policy |
| Batch | `python -m src.pipeline` | Download, train, write `outputs/` |
| Service | `python -m service.main` | HTTP API + status page at `/`, OpenAPI at `/docs` |

## Quick start

```text
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.pipeline status
python -m service.main
```

Then open http://127.0.0.1:8000 and http://127.0.0.1:8000/docs.

Download the Kaggle snapshot (once, locally):

```text
python -m src.pipeline download
```

Tests do not need the CSV:

```text
python -m pytest tests/ -q
```

Write the APA starter list:

```text
python -m src.agent.literature
```

## API (v1)

| Method | Path | Ready when |
|---|---|---|
| GET | `/health` | always |
| GET | `/v1/agent` | always (PEAS) |
| GET | `/v1/meta` | always |
| GET | `/v1/status` | always (artifact flags) |
| GET | `/v1/archetypes` | Member A cluster export |
| GET | `/v1/forecast` | Member A forecast export |
| GET | `/v1/scores/{player_id}` | Member B scores export |
| POST | `/v1/search/xi` | scores + search export |
| POST | `/v1/squad/allocate` | forecast + squad export |
| POST | `/v1/policy/bid` | scores + policy export |

Missing artifacts return **503** with `owner` and the producer command.

## Team

| Role | Owns |
|---|---|
| Member A | PEAS, archetypes, forecast, `GET /v1/archetypes`, `GET /v1/forecast` |
| Member B | Value models, sequence net, `GET /v1/scores/{id}` |
| Member C | XI search, MIP, policy, remaining POST routes, service wiring |

## Documents

- [`docs/REQUIREMENTS_AND_IMPLEMENTATION_PLAN.md`](docs/REQUIREMENTS_AND_IMPLEMENTATION_PLAN.md)
- [`docs/contracts.md`](docs/contracts.md)
- [`docs/DATA.md`](docs/DATA.md)
- [`docs/agent.md`](docs/agent.md)
