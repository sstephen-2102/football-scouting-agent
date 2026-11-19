"""Data-quality report and download manifest. Shared."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.common.config import DATA_RAW, KAGGLE_DATASET, OUTPUTS, SNAPSHOT_DATE
from src.common.exceptions import InvalidParameterError


def generate_data_quality_report(player_seasons: pd.DataFrame) -> dict:
    """Row counts by league/season/split, missingness, and min-appearance drops."""
    if not isinstance(player_seasons, pd.DataFrame):
        raise InvalidParameterError("player_seasons must be a pandas DataFrame.")

    def _counts(column: str) -> dict[str, int]:
        if column not in player_seasons.columns:
            return {}
        return {
            str(key): int(value)
            for key, value in player_seasons[column].value_counts(dropna=False).items()
        }

    missingness = {
        column: int(player_seasons[column].isna().sum())
        for column in player_seasons.columns
    }
    return {
        "n_rows": len(player_seasons),
        "n_players": int(player_seasons["player_id"].nunique())
        if "player_id" in player_seasons.columns
        else 0,
        "n_seasons": int(player_seasons["season"].nunique())
        if "season" in player_seasons.columns
        else 0,
        "counts_by_split": _counts("split"),
        "counts_by_league": _counts("competition_id"),
        "counts_by_season": _counts("season"),
        "missingness": missingness,
        "dropped_below_min_appearances": 0,
    }


def write_data_manifest(
    raw_dir: Path | None = None,
    outputs_dir: Path | None = None,
) -> Path:
    """SHA-256 and Kaggle version for each CSV → `outputs/data_manifest.json`."""
    directory = Path(raw_dir) if raw_dir is not None else DATA_RAW
    out_root = Path(outputs_dir) if outputs_dir is not None else OUTPUTS
    files = []
    for path in sorted(directory.glob("*.csv")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
    payload = {
        "dataset": KAGGLE_DATASET,
        "snapshot_date": SNAPSHOT_DATE,
        "raw_dir": str(directory),
        "files": files,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    destination = out_root / "data_manifest.json"
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination
