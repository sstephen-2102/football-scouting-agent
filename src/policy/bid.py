"""Member C — bid/pass bandits and Q-learning under remaining budget B."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.common.config import RANDOM_STATE
from src.common.exceptions import not_built

BID_ACTIONS = ("bid", "pass")


def bid_actions() -> tuple[str, ...]:
    """Discrete actions for the transfer-window MDP."""
    return BID_ACTIONS


def mdp_state(
    position_hole: str,
    remaining_budget_eur: float,
    archetype_mix: tuple[int, ...] | None = None,
) -> Any:
    """Hashable state: hole, remaining B, optional archetype mix."""
    not_built("member_c", "mdp_state")


def reward_if_signed(player_row: pd.Series) -> float:
    """Reward = predicted residual (undervaluation) if the agent bids and signs."""
    not_built("member_c", "reward_if_signed")


def epsilon_greedy(
    q_values: dict,
    state: Any,
    epsilon: float,
    random_state: int = RANDOM_STATE,
) -> str:
    """ε-greedy action from Q(state, ·)."""
    not_built("member_c", "epsilon_greedy")


def ucb_action(
    counts: dict,
    q_values: dict,
    state: Any,
    exploration: float = 2.0,
) -> str:
    """UCB1 over bid/pass for this state."""
    not_built("member_c", "ucb_action")


def q_learning_update(
    q_table: dict,
    state: Any,
    action: str,
    reward: float,
    next_state: Any,
    alpha: float = 0.1,
    gamma: float = 0.9,
) -> dict:
    """One-step Q update. Return the mutated table."""
    not_built("member_c", "q_learning_update")


def greedy_by_undervalued(
    shortlist: pd.DataFrame,
    remaining_budget_eur: float,
) -> str:
    """Myopic baseline: bid if the next player is the top undervalued fit under B."""
    not_built("member_c", "greedy_by_undervalued")


def simulate_transfer_window(
    shortlist: pd.DataFrame,
    budget_eur: float,
    method: str,
    n_episodes: int = 200,
    random_state: int = RANDOM_STATE,
) -> dict:
    """Roll out ε-greedy, UCB, Q, or greedy-by-value. Return cumulative undervaluation."""
    not_built("member_c", "simulate_transfer_window")


def compare_policies(results: dict[str, dict]) -> pd.DataFrame:
    """One row per method: mean return, regret vs greedy, n_episodes."""
    not_built("member_c", "compare_policies")


def decide_bid(
    player_id: int,
    remaining_budget_eur: float,
    position_hole: str,
    q_table: dict | None = None,
) -> dict:
    """Runtime decision for `POST /v1/policy/bid`. Keys: action, player_id, rationale."""
    not_built("member_c", "decide_bid")


def export_policy() -> Path:
    """Write `outputs/policy/q_table.json` (pipeline: policy)."""
    not_built("member_c", "export_policy")
