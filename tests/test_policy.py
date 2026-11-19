"""Synthetic bid-policy tests."""

import pandas as pd

from src.policy.bid import (
    decide_bid,
    mdp_state,
    q_learning_update,
    reward_if_signed,
)


def test_q_update_changes_table() -> None:
    state = mdp_state("FWD", 50_000_000)
    nxt = mdp_state("FWD", 40_000_000)
    table: dict = {}
    q_learning_update(table, state, "bid", reward=1.0, next_state=nxt)
    assert table[(state, "bid")] != 0.0


def test_decide_bid_keys() -> None:
    decision = decide_bid(7, 25_000_000, "ST", q_table={})
    assert set(decision) >= {"action", "player_id", "rationale"}
    assert decision["action"] in {"bid", "pass"}
    assert decision["player_id"] == 7


def test_reward_prefers_residual() -> None:
    row = pd.Series({"pred_value": 10.0, "next_value_eur": 13.0})
    assert reward_if_signed(row) == 3.0