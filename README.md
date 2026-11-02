# Football Scouting Agent

University of San Diego M.S. Applied AI — **AAI-501** final team project (3 members).

A sporting-director lab: archetypes, market-value scores, starting-XI search, MIP budget allocation, and a bid/pass policy.

**Two UIs, one domain layer.** FastAPI is the product API. Streamlit is the instructor-facing lab (same pages, easy to share). **No notebooks.**

**Data.** [Football Data from Transfermarkt](https://www.kaggle.com/datasets/davidcariboo/player-scores) (CC0), freeze **6 July 2026**. Raw CSVs are git-ignored.

## Architecture

```text
instructor  →  streamlit run streamlit_app.py  ─┐
HTTP clients →  python -m service.main         ─┴→  outputs/
                                                    ↑
                                   python -m src.pipeline …
```

Neither UI fits models. Members write domain code under `src/`, export artifacts, and both UIs read them.

| Layer | Path | Role |
|---|---|---|
| Domain | `src/` | Clustering, forecast, models, search, MIP, policy |
| Batch | `python -m src.pipeline` | Download, train, write `outputs/` |
| Streamlit | `streamlit run streamlit_app.py` | Instructor lab (share this) |
| FastAPI | `python -m service.main` | Product API + lab at `/`, OpenAPI at `/docs` |

## Quick start

```text
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.pipeline status
streamlit run streamlit_app.py
```

Share **Streamlit** with the instructor (http://127.0.0.1:8501). Same pages as the FastAPI lab: Dashboard, Agent, Data Freeze, Archetype Lab, Forecast Lab, Experiment Board, Player Scores, Search Lab, Squad Builder, Bid Policy. Missing artifacts show the owner and the pipeline command.

Product API (optional):

```text
python -m service.main
```

Then http://127.0.0.1:8000 and http://127.0.0.1:8000/docs.

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

## Function contracts

Every planned feature has a typed signature in `src/`. Bodies raise `NotBuiltError` until the owner implements them.

- Signatures: [`docs/contracts.md`](docs/contracts.md)
- **How to implement each function:** [`docs/IMPLEMENTATION_DEFINITIONS.md`](docs/IMPLEMENTATION_DEFINITIONS.md)
- Test registry: `src/common/signatures.py`

Implement the body. Do not rename arguments without a team PR.

```python
from src.cluster import fit_kmeans          # Member A
from src.modeling import fit_ridge_value    # Member B
from src.search import informed_search      # Member C
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

| Role | Name | Owns |
|---|---|---|
| Member A | Arunava Banerjee | PEAS, archetypes, forecast, `GET /v1/archetypes`, `GET /v1/forecast` |
| Member B | Binish Hamdani | Value models, sequence net, `GET /v1/scores/{id}` |
| Member C | Sanjay Varghese Stephen | XI search, MIP, policy, remaining POST routes, service wiring |

## Documents

- [`docs/REQUIREMENTS_AND_IMPLEMENTATION_PLAN.md`](docs/REQUIREMENTS_AND_IMPLEMENTATION_PLAN.md)
- [`docs/IMPLEMENTATION_DEFINITIONS.md`](docs/IMPLEMENTATION_DEFINITIONS.md)
- [`docs/contracts.md`](docs/contracts.md)
- [`docs/DATA.md`](docs/DATA.md)
- [`docs/agent.md`](docs/agent.md)
