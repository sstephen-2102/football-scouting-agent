"""Expected pipeline artifacts. The HTTP service reads these; it does not train."""

from __future__ import annotations

from pathlib import Path

from src.common.config import OUTPUTS

ARTIFACTS: dict[str, dict[str, str]] = {
    "split": {
        "path": "splits/player_season_split.parquet",
        "owner": "shared",
        "producer": "python -m src.pipeline build-seasons",
    },
    "archetypes": {
        "path": "clustering/archetype_labels.parquet",
        "owner": "member_a",
        "producer": "python -m src.pipeline cluster",
    },
    "forecast": {
        "path": "forecast/budget.json",
        "owner": "member_a",
        "producer": "python -m src.pipeline forecast",
    },
    "scores": {
        "path": "experiments/player_scores.parquet",
        "owner": "member_b",
        "producer": "python -m src.pipeline score",
    },
    "search": {
        "path": "search/best_xi.json",
        "owner": "member_c",
        "producer": "python -m src.pipeline search-xi",
    },
    "squad": {
        "path": "optimize/squad.csv",
        "owner": "member_c",
        "producer": "python -m src.pipeline allocate",
    },
    "policy": {
        "path": "policy/q_table.json",
        "owner": "member_c",
        "producer": "python -m src.pipeline policy",
    },
}


def artifact_path(name: str) -> Path:
    return OUTPUTS / ARTIFACTS[name]["path"]


def artifact_status() -> dict[str, dict[str, str | bool]]:
    """Return readiness of every contracted output file."""
    status: dict[str, dict[str, str | bool]] = {}
    for name, spec in ARTIFACTS.items():
        path = OUTPUTS / spec["path"]
        status[name] = {
            "ready": path.is_file(),
            "path": str(path),
            "owner": spec["owner"],
            "producer": spec["producer"],
        }
    return status
