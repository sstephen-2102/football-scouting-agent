"""Data-quality report and download manifest. Shared."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.exceptions import not_built


def generate_data_quality_report(player_seasons: pd.DataFrame) -> dict:
    """Row counts by league/season/split, missingness, and min-appearance drops."""
    not_built("shared", "generate_data_quality_report")


def write_data_manifest(raw_dir: Path | None = None) -> Path:
    """SHA-256 and Kaggle version for each CSV → `outputs/data_manifest.json`."""
    not_built("shared", "write_data_manifest")
