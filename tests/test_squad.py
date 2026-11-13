"""Synthetic MIP vs greedy squad tests."""

from pathlib import Path

import pandas as pd

from src.optimize.squad import (
    compare_mip_vs_greedy,
    formulate_squad_mip,
    greedy_spend,
    solve_squad_mip,
)


def _candidates() -> pd.DataFrame:
    rows = []
    pid = 1
    # 2 GK, 5 DEF, 4 MID, 4 FWD → enough to choose 1/4/3/3
    specs = [
        ("Goalkeeper", "Goalkeeper", "GK", [10, 3]),
        ("Defender", "Centre-Back", "DEF", [9, 8, 7, 6, 2]),
        ("Midfield", "Central Midfield", "MID", [9, 8, 7, 2]),
        ("Attack", "Centre-Forward", "FWD", [12, 8, 7, 2]),
    ]
    for position, sub, _group, values in specs:
        for value in values:
            rows.append(
                {
                    "player_id": pid,
                    "position": position,
                    "sub_position": sub,
                    "pred_value": float(value),
                }
            )
            pid += 1
    return pd.DataFrame(rows)


def test_mip_and_greedy_fill_eleven(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.optimize.squad.OUTPUTS", tmp_path)
    candidates = _candidates()
    budget = 1_000.0
    problem = formulate_squad_mip(candidates, budget)
    mip = solve_squad_mip(problem)
    greedy = greedy_spend(candidates, budget)
    assert mip["status"] == "optimal"
    assert len(mip["player_ids"]) == 11
    assert len(greedy["player_ids"]) == 11
    assert mip["quality"] >= greedy["quality"] - 1e-9
    summary = compare_mip_vs_greedy(mip, greedy)
    assert "quality_gap" in summary
    assert Path(summary["figure_path"]).is_file()
    assert Path(summary["figure_path"]).is_relative_to(tmp_path)
