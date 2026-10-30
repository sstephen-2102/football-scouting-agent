"""Batch jobs that produce artifacts. The HTTP service never trains models."""

from __future__ import annotations

import argparse

from src.agent.literature import write_references
from src.artifacts import artifact_status
from src.data.download import download_kaggle_dataset


COMMANDS_NOT_BUILT = {
    "build-seasons": "Assemble player-season table and freeze the time split.",
    "cluster": "Fit archetypes on train stats (no market value in the fit).",
    "forecast": "STL / Holt-Winters / ARIMA weekly series → budget.json.",
    "score": "Train value models and write player_scores.parquet.",
    "search-xi": "Uninformed vs informed XI search.",
    "allocate": "Simplex toy LP plus MIP squad under budget B.",
    "policy": "Bandit / Q-learning bid-pass policy.",
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
        choices=["status", "download", "references", *COMMANDS_NOT_BUILT],
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
    raise SystemExit(
        f"{arguments.command} is not implemented yet. "
        f"{COMMANDS_NOT_BUILT[arguments.command]} "
        "See docs/contracts.md for the owning member."
    )


if __name__ == "__main__":
    main()
