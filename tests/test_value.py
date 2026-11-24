"""Member B — R0 ridge: feature frame, fit guards and the scores table."""

import numpy as np
import pandas as pd
import pytest

from src.common.exceptions import InvalidParameterError
from src.modeling.value import (
    POSITIONS,
    add_position_dummies,
    fit_ridge_value,
    predict_player_scores,
    ridge_feature_columns,
)


def _toy_table(rows: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    table = pd.DataFrame(
        {
            "player_id": np.arange(rows),
            "season": 2018 + np.arange(rows) % 5,
            "split": "train",
            "position": rng.choice(["Attack", "Defender", "Midfield"], size=rows),
            "minutes": rng.uniform(500, 3000, rows),
            "goals_per90": rng.uniform(0, 1, rows),
            "assists_per90": rng.uniform(0, 1, rows),
            "yellow_per90": rng.uniform(0, 0.5, rows),
            "red_per90": rng.uniform(0, 0.1, rows),
            "age": rng.uniform(18, 36, rows),
            "height_in_cm": rng.uniform(165, 200, rows),
            "n_appearances": rng.integers(10, 38, rows),
        }
    )
    table["log_next_value"] = 14 + 0.5 * table["goals_per90"] + rng.normal(0, 0.1, rows)
    return table


def test_position_dummies_tick_exactly_one_box_per_row() -> None:
    table = _toy_table()
    table.loc[0, "position"] = None
    frame = add_position_dummies(table)
    boxes = [f"pos_{name}" for name in POSITIONS]
    assert list(frame[boxes].sum(axis=1).unique()) == [1]
    assert frame.loc[0, "pos_Missing"] == 1
    assert "pos_Attack" not in table.columns  # the caller's table is untouched


def test_feature_list_has_thirteen_columns() -> None:
    assert len(ridge_feature_columns()) == 13


def test_fit_rejects_the_target_as_a_feature() -> None:
    table = add_position_dummies(_toy_table())
    with pytest.raises(InvalidParameterError):
        fit_ridge_value(table, ridge_feature_columns() + ["log_next_value"])


def test_fit_rejects_non_training_rows() -> None:
    table = add_position_dummies(_toy_table())
    table.loc[0, "split"] = "val"
    with pytest.raises(InvalidParameterError):
        fit_ridge_value(table, ridge_feature_columns())


def test_scores_table_follows_the_contract() -> None:
    table = _toy_table()
    columns = ridge_feature_columns()
    model = fit_ridge_value(add_position_dummies(table), columns)
    scores = predict_player_scores(model, table, columns)
    assert list(scores.columns) == ["player_id", "season", "split", "pred_value", "p_undervalued"]
    assert len(scores) == len(table)
    assert (scores["pred_value"] >= 0).all()
    assert scores["p_undervalued"].isna().all()