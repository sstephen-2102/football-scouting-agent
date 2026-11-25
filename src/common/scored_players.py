"""Shared scored-player loader for Member C exports (search, squad, policy).

Joins `player_scores.parquet` (Member B) with position, cutoff-dated previous
value, and next value from the shared split table. Reads only; never fits a model.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.config import OUTPUTS
from src.common.exceptions import InvalidParameterError

SPLITS = ("train", "val", "test", "descriptive")
SCORES_RELATIVE = Path("experiments") / "player_scores.parquet"
SPLIT_RELATIVE = Path("splits") / "player_season_split.parquet"

# Per-group shortlist quotas. Top-N by pred_value alone picks no goalkeepers and
# one defender on the 2024 test split, so every slot group gets its own quota.
SHORTLIST_PER_GROUP = {"GK": 3, "DEF": 10, "MID": 10, "FWD": 10}


def load_scored_players(
    split: str | None,
    season: int | None = None,
    root: Path | None = None,
) -> pd.DataFrame:
    """Scores joined to position, previous value, and next value.

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
        columns=[
            "player_id",
            "season",
            "position",
            "sub_position",
            "prev_value_eur",
            "next_value_eur",
        ],
    )
    frame = scores.merge(table, on=["player_id", "season"], how="left", validate="one_to_one")

    if split is not None:
        frame = frame[frame["split"] == split]
    if season is not None:
        frame = frame[frame["season"] == int(season)]
    return frame.reset_index(drop=True)


def with_ask_cost(frame: pd.DataFrame) -> pd.DataFrame:
    """Cost = last valuation on or before the cutoff (`prev_value_eur`).

    Falls back to `pred_value` only where no prior valuation exists. Both are
    known at the cutoff, so the cost is not leaked from the next-season target.
    """
    out = frame.copy()
    out["cost"] = out["prev_value_eur"].fillna(out["pred_value"]).astype(float)
    return out


def top_per_group(frame: pd.DataFrame, quotas: dict[str, int] | None = None) -> pd.DataFrame:
    """Top `quota` players per slot group by predicted value.

    Groups come from `position` and `sub_position` via `src.search.positions`.
    The import is local to avoid a cycle (search imports this module).
    """
    from src.search.positions import position_group

    quota_map = dict(SHORTLIST_PER_GROUP if quotas is None else quotas)
    if any(int(v) < 1 for v in quota_map.values()):
        raise InvalidParameterError("every group quota must be at least 1.")

    groups = [
        position_group(
            row.position if pd.notna(row.position) else None,
            row.sub_position if pd.notna(row.sub_position) else None,
        )
        for row in frame[["position", "sub_position"]].itertuples(index=False)
    ]
    tagged = frame.assign(group=groups)
    tagged = tagged[tagged["group"].isin(quota_map)]
    picked = [
        tagged[tagged["group"] == group]
        .sort_values("pred_value", ascending=False)
        .head(int(quota))
        for group, quota in quota_map.items()
    ]
    result = pd.concat(picked, ignore_index=True)
    return result.sort_values("pred_value", ascending=False).reset_index(drop=True)
