"""Shared scored-player loader for Member C exports (search, squad, policy).

Joins `player_scores.parquet` (Member B) with position and next value from the
shared split table. Reads only; never fits a model.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.config import OUTPUTS
from src.common.exceptions import InvalidParameterError

SPLITS = ("train", "val", "test", "descriptive")
SCORES_RELATIVE = Path("experiments") / "player_scores.parquet"
SPLIT_RELATIVE = Path("splits") / "player_season_split.parquet"


def load_scored_players(
    split: str | None,
    season: int | None = None,
    root: Path | None = None,
) -> pd.DataFrame:
    """Scores joined to position and next value.

    `split=None` returns every split. `season` filters when given.
    Raises FileNotFoundError when an upstream artifact is missing.
    """
    if split is not None and split not in SPLITS:
        raise InvalidParameterError(f"split must be one of {SPLITS} or None.")

    base = OUTPUTS if root is None else Path(root)
    scores_path = base / SCORES_RELATIVE
    split_path = base / SPLIT_RELATIVE
    for path, producer in (
        (scores_path, "python -m src.pipeline score"),
        (split_path, "python -m src.pipeline build-seasons"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing. Run `{producer}` first.")

    scores = pd.read_parquet(scores_path)
    table = pd.read_parquet(
        split_path,
        columns=["player_id", "season", "position", "sub_position", "next_value_eur"],
    )
    frame = scores.merge(table, on=["player_id", "season"], how="left", validate="one_to_one")

    if split is not None:
        frame = frame[frame["split"] == split]
    if season is not None:
        frame = frame[frame["season"] == int(season)]
    return frame.reset_index(drop=True)


def top_by_pred_value(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    """Top-n rows by predicted value. `p_undervalued` stays unused until C1 lands."""
    if n < 1:
        raise InvalidParameterError("n must be at least 1.")
    return frame.sort_values("pred_value", ascending=False).head(int(n)).reset_index(drop=True)
