"""Batch jobs that produce artifacts. The HTTP service never trains models."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from src.agent.literature import write_references
from src.artifacts import artifact_status
from src.cluster.archetypes import export_archetypes
from src.common.exceptions import NotBuiltError
from src.data.download import download_kaggle_dataset
from src.data.seasons import export_player_season_split
from src.forecast.timeseries import export_forecast
from src.modeling.value import export_player_scores
from src.optimize.squad import export_squad
from src.policy.bid import export_policy
from src.search.xi import export_best_xi

EXPORTS: dict[str, Callable[..., Path]] = {
    "build-seasons": export_player_season_split,
    "cluster": export_archetypes,
    "forecast": export_forecast,
    "score": export_player_scores,
    "search-xi": export_best_xi,
    "allocate": export_squad,
    "policy": export_policy,
}


def _print_status() -> None:
    print("Pipeline artifact status")
    for name, row in artifact_status().items():
        flag = "ready" if row["ready"] else "missing"
        print(f"  {name:12} {flag:8} {row['path']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Football scouting agent batch pipeline.",
    )
    parser.add_argument(
        "command",
        choices=["status", "download", "references", *EXPORTS],
    )
    arguments = parser.parse_args()
    if arguments.command == "status":
        _print_status()
        return
    if arguments.command == "download":
        path = download_kaggle_dataset()
        print(f"Downloaded Transfermarkt files to {path}")
        return
    if arguments.command == "references":
        print(f"Wrote {write_references()}")
        return
    try:
        written = EXPORTS[arguments.command]()
    except NotBuiltError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Wrote {written}")


if __name__ == "__main__":
    main()
