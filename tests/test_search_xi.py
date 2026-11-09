"""Synthetic XI search tests (no Kaggle, no scores artifact)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.search.xi import (
    apply_action,
    compare_search,
    informed_search,
    initial_xi_state,
    is_goal,
    legal_actions,
    uninformed_search,
    xi_quality,
)


def _synthetic_players() -> pd.DataFrame:
    rows = [
        (1, "Goalkeeper", "Goalkeeper", 10.0),
        (2, "Defender", "Right-Back", 8.0),
        (3, "Defender", "Centre-Back", 9.0),
        (4, "Defender", "Centre-Back", 7.0),
        (5, "Defender", "Left-Back", 6.0),
        (6, "Midfield", "Defensive Midfield", 8.0),
        (7, "Midfield", "Central Midfield", 9.0),
        (8, "Midfield", "Attacking Midfield", 7.0),
        (9, "Attack", "Right Winger", 8.0),
        (10, "Attack", "Centre-Forward", 12.0),
        (11, "Attack", "Left Winger", 8.0),
        (12, "Attack", "Centre-Forward", 5.0),
        # extras so shortlist > 11
        (13, "Defender", "Centre-Back", 4.0),
        (14, "Midfield", "Central Midfield", 3.0),
        (15, "Attack", "Left Winger", 2.0),
    ]
    return pd.DataFrame(
        rows,
        columns=["player_id", "position", "sub_position", "pred_value"],
    )


def test_fixed_order_actions_and_goal() -> None:
    players = _synthetic_players()
    state = initial_xi_state(list(players["player_id"]))
    actions = legal_actions(state, players)
    assert actions
    assert all(slot == "GK" for slot, _ in actions)

    nxt = apply_action(state, ("GK", 1))
    assert not is_goal(nxt)
    assert nxt.assignment[0] == 1
    assert state.assignment[0] is None


def test_bfs_and_astar_fill_xi() -> None:
    players = _synthetic_players()
    scores = players[["player_id", "pred_value"]]
    state = initial_xi_state(list(players["player_id"]))

    bfs = uninformed_search(state, players, scores, algorithm="bfs", expansion_cap=500)
    astar = informed_search(state, players, scores, algorithm="astar", expansion_cap=500)

    assert len(bfs["xi"]) == 11
    assert len(astar["xi"]) == 11
    assert bfs["quality"] > 0
    assert astar["quality"] >= bfs["quality"] - 1e-9
    assert astar["quality"] >= bfs["quality"] - 1e-9

    summary = compare_search(bfs, astar)
    assert summary["informed_minus_uninformed"] == astar["quality"] - bfs["quality"]
    assert Path(summary["figure_path"]).is_file()