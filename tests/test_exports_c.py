"""Member C exports on a synthetic workspace (no Kaggle, no real artifacts)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

import src.artifacts
import src.common.config
import src.optimize.squad
import src.policy.bid
import src.search.xi
from src.common.exceptions import InvalidParameterError
from src.common.scored_players import (
    SHORTLIST_PER_GROUP,
    load_scored_players,
    top_per_group,
    with_ask_cost,
)
from src.policy.bid import _decode_key, _encode_key, _load_q_table, decide_bid

# (sub_position, pred_value) for the 2024 test pool; one GK, four DEF, four MID, five FWD.
TEST_PLAYERS = [
    ("Goalkeeper", 9.0),
    ("Right-Back", 6.0),
    ("Centre-Back", 7.0),
    ("Centre-Back", 5.5),
    ("Left-Back", 5.0),
    ("Defensive Midfield", 8.0),
    ("Central Midfield", 7.5),
    ("Attacking Midfield", 6.5),
    ("Central Midfield", 4.0),
    ("Right Winger", 9.5),
    ("Centre-Forward", 10.0),
    ("Left Winger", 8.5),
    ("Centre-Forward", 3.0),
    ("Left Winger", 2.0),
]
SUB_TO_POSITION = {
    "Goalkeeper": "Goalkeeper",
    "Right-Back": "Defender",
    "Left-Back": "Defender",
    "Centre-Back": "Defender",
    "Defensive Midfield": "Midfield",
    "Central Midfield": "Midfield",
    "Attacking Midfield": "Midfield",
    "Right Winger": "Attack",
    "Left Winger": "Attack",
    "Centre-Forward": "Attack",
}
MILLION = 1_000_000


def _write_workspace(root, test_players=TEST_PLAYERS, *, with_val: bool = True) -> None:
    rows, split_rows = [], []
    for index, (sub, pred) in enumerate(test_players):
        pid = 1000 + index
        rows.append({"player_id": pid, "season": 2024, "split": "test", "pred_value": pred * MILLION, "p_undervalued": float("nan")})
        split_rows.append({"player_id": pid, "season": 2024, "position": SUB_TO_POSITION[sub], "sub_position": sub, "prev_value_eur": pred * 0.9 * MILLION, "next_value_eur": pred * 1.1 * MILLION, "split": "test"})
    if with_val:
        for index, (sub, pred) in enumerate(TEST_PLAYERS):
            pid = 2000 + index
            rows.append({"player_id": pid, "season": 2023, "split": "val", "pred_value": pred * MILLION, "p_undervalued": float("nan")})
            split_rows.append({"player_id": pid, "season": 2023, "position": SUB_TO_POSITION[sub], "sub_position": sub, "prev_value_eur": pred * 0.9 * MILLION, "next_value_eur": (pred * 1.2 + index % 3) * MILLION, "split": "val"})
    (root / "experiments").mkdir(parents=True, exist_ok=True)
    (root / "splits").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(root / "experiments" / "player_scores.parquet", index=False)
    pd.DataFrame(split_rows).to_parquet(root / "splits" / "player_season_split.parquet", index=False)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    for module in (src.search.xi, src.optimize.squad, src.policy.bid, src.artifacts):
        monkeypatch.setattr(module, "OUTPUTS", tmp_path)
    _write_workspace(tmp_path)
    return tmp_path


def test_loader_joins_position_and_filters_split(workspace) -> None:
    test_rows = load_scored_players("test", root=workspace)
    assert len(test_rows) == len(TEST_PLAYERS)
    assert set(test_rows["split"]) == {"test"}
    assert {"position", "sub_position", "next_value_eur"} <= set(test_rows.columns)


def test_per_group_shortlist_keeps_every_group(workspace) -> None:
    shortlist = top_per_group(load_scored_players("test", root=workspace))
    groups = set(shortlist["sub_position"].map(lambda s: SUB_TO_POSITION[s]))
    assert len(shortlist) == len(TEST_PLAYERS)  # quotas are larger than the pool per group
    assert {"Goalkeeper", "Defender", "Midfield", "Attack"} == groups


def test_per_group_shortlist_caps_each_group(workspace) -> None:
    quotas = {"GK": 1, "DEF": 2, "MID": 2, "FWD": 2}
    shortlist = top_per_group(load_scored_players("test", root=workspace), quotas)
    counts = shortlist["sub_position"].map(lambda s: SUB_TO_POSITION[s]).value_counts().to_dict()
    assert counts == {"Goalkeeper": 1, "Defender": 2, "Midfield": 2, "Attack": 2}
    assert set(SHORTLIST_PER_GROUP) == {"GK", "DEF", "MID", "FWD"}


def test_ask_cost_uses_prev_value_then_falls_back_to_prediction(workspace) -> None:
    frame = load_scored_players("test", root=workspace)
    frame.loc[0, "prev_value_eur"] = None
    costed = with_ask_cost(frame)
    assert costed.loc[0, "cost"] == frame.loc[0, "pred_value"]
    assert costed.loc[1, "cost"] == frame.loc[1, "prev_value_eur"]


def test_loader_rejects_unknown_split(workspace) -> None:
    with pytest.raises(InvalidParameterError):
        load_scored_players("holdout", root=workspace)


def test_loader_names_the_missing_upstream_artifact(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="player_scores"):
        load_scored_players("test", root=tmp_path)


def test_best_xi_fills_eleven_and_reports_both_searches(workspace) -> None:
    path = src.search.xi.export_best_xi(club_id=11, season=2024)
    payload = json.loads(path.read_text())
    assert payload["shortlist_source"].endswith("test_by_pred_value")
    assert len(payload["shortlist_player_ids"]) == len(TEST_PLAYERS)
    assert len(payload["uninformed"]["xi"]) == 11
    assert len(payload["informed"]["xi"]) == 11
    assert payload["informed"]["quality"] >= payload["uninformed"]["quality"] - 1e-9
    assert payload["comparison"]["uninformed_algorithm"] == "bfs"
    assert payload["comparison"]["nodes_ratio"] is not None


def test_best_xi_rejects_ids_without_scores(workspace) -> None:
    ids = [1000 + i for i in range(10)] + [999_999]
    with pytest.raises(InvalidParameterError, match="999999"):
        src.search.xi.export_best_xi(shortlist_player_ids=ids)


def test_squad_mip_meets_position_minima_under_a_large_budget(workspace) -> None:
    csv = src.optimize.squad.export_squad(budget_eur=1_000 * MILLION)
    table = pd.read_csv(csv)
    manifest = json.loads((csv.parent / "allocation_manifest.json").read_text())
    assert manifest["budget_source"] == "argument"
    assert manifest["mip"]["status"] == "optimal"
    mip_rows = table[table["source"] == "mip"]
    assert len(mip_rows) == 11
    counts = mip_rows["slot_or_group"].value_counts().to_dict()
    for group, minimum in src.common.config.POSITION_MINIMA.items():
        assert counts.get(group, 0) >= minimum


def test_squad_is_infeasible_without_a_goalkeeper(tmp_path, monkeypatch) -> None:
    for module in (src.search.xi, src.optimize.squad, src.policy.bid, src.artifacts):
        monkeypatch.setattr(module, "OUTPUTS", tmp_path)
    _write_workspace(tmp_path, test_players=[p for p in TEST_PLAYERS if p[0] != "Goalkeeper"], with_val=False)
    csv = src.optimize.squad.export_squad(budget_eur=1_000 * MILLION)
    manifest = json.loads((csv.parent / "allocation_manifest.json").read_text())
    assert manifest["mip"]["status"] == "infeasible"
    table = pd.read_csv(csv)
    assert set(table["source"]) <= {"greedy"}


def test_squad_uses_min_feasible_xi_budget_when_no_forecast(workspace) -> None:
    csv = src.optimize.squad.export_squad()
    manifest = json.loads((csv.parent / "allocation_manifest.json").read_text())
    assert manifest["budget_source"].startswith("min_feasible_xi")
    assert manifest["mip"]["status"] == "optimal"
    assert len(manifest["mip"]["player_ids"]) == 11
    assert manifest["mip"]["spend"] <= manifest["budget_eur"] + 1e-6


def test_budget_comes_from_forecast_only_when_stable(workspace) -> None:
    forecast = workspace / "forecast"
    forecast.mkdir()
    (forecast / "budget.json").write_text(json.dumps({"budget_eur": 55 * MILLION, "unstable": False}))
    assert src.optimize.squad._budget_from_forecast(workspace)[0] == 55 * MILLION
    (forecast / "budget.json").write_text(json.dumps({"budget_eur": 55 * MILLION, "unstable": True}))
    assert src.optimize.squad._budget_from_forecast(workspace)[0] == src.optimize.squad.DEFAULT_BUDGET_EUR


def test_policy_export_round_trips_into_decide_bid(workspace) -> None:
    path = src.policy.bid.export_policy()
    assert path.is_file()
    comparison = pd.read_csv(path.parent / "policy_comparison.csv")
    assert set(comparison["method"]) == set(src.policy.bid.POLICY_METHODS)

    table = _load_q_table()
    assert table, "trained Q-table should load back from the artifact"
    assert all(isinstance(key, tuple) and len(key) == 2 for key in table)
    decision = decide_bid(1000, 50 * MILLION, "FWD")
    assert decision["action"] in {"bid", "pass"}


def test_epsilon_greedy_requires_a_trained_table(workspace) -> None:
    shortlist = with_ask_cost(top_per_group(load_scored_players("val", root=workspace)))
    with pytest.raises(InvalidParameterError, match="trained q_table"):
        src.policy.bid.simulate_transfer_window(shortlist, 80 * MILLION, "epsilon_greedy", n_episodes=5)


def test_epsilon_greedy_does_not_update_the_table(workspace) -> None:
    shortlist = with_ask_cost(top_per_group(load_scored_players("val", root=workspace)))
    table = src.policy.bid.train_q_table(shortlist, 80 * MILLION, n_episodes=20)
    before = dict(table)
    src.policy.bid.simulate_transfer_window(shortlist, 80 * MILLION, "epsilon_greedy", n_episodes=20, q_table=table)
    assert table == before


def test_q_key_encoding_round_trips() -> None:
    state = ("FWD", 7, (2, 0, 1))
    assert _decode_key(_encode_key(state, "bid")) == (state, "bid")
    empty = ("FWD", 0, ())
    assert _decode_key(_encode_key(empty, "pass")) == (empty, "pass")
