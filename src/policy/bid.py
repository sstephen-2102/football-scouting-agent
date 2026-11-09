"""Member C — bid/pass bandits and Q-learning under remaining budget B."""

from __future__ import annotations

import json
import math
from pathlib import Path
from random import Random
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.artifacts import artifact_path
from src.common.config import OUTPUTS, RANDOM_STATE
from src.common.exceptions import InvalidParameterError, not_built

BID_ACTIONS = ("bid", "pass")
BUDGET_BIN_EUR = 5_000_000  # €5m bins


def bid_actions() -> tuple[str, ...]:
    """Discrete actions for the transfer-window MDP."""
    return BID_ACTIONS


def mdp_state(
    position_hole: str,
    remaining_budget_eur: float,
    archetype_mix: tuple[int, ...] | None = None,
) -> tuple:
    """Hashable state: hole, remaining B bin, optional archetype mix."""
    if not isinstance(position_hole, str) or not position_hole:
        raise InvalidParameterError("position_hole must be a non-empty string.")
    budget = float(remaining_budget_eur)
    if budget < 0:
        raise InvalidParameterError("remaining_budget_eur must be non-negative.")
    budget_bin = int(budget // BUDGET_BIN_EUR)
    mix = tuple(archetype_mix) if archetype_mix is not None else ()
    return (position_hole, budget_bin, mix)


def reward_if_signed(player_row: pd.Series) -> float:
    """Residual next_value_eur - pred_value when available, else 0."""
    if not isinstance(player_row, pd.Series):
        raise InvalidParameterError("player_row must be a Series.")
    if "pred_value" not in player_row.index:
        raise InvalidParameterError("player_row needs pred_value.")
    pred = float(player_row["pred_value"])
    if "next_value_eur" in player_row.index and pd.notna(player_row["next_value_eur"]):
        return float(player_row["next_value_eur"]) - pred
    return 0.0


def _q_get(q_values: dict, state: Any, action: str) -> float:
    return float(q_values.get((state, action), 0.0))


def _best_action(q_values: dict, state: Any) -> str:
    scores = {action: _q_get(q_values, state, action) for action in BID_ACTIONS}
    # Stable tie-break: prefer bid if tied (scout bias); either is fine if documented.
    return max(BID_ACTIONS, key=lambda action: (scores[action], action == "bid"))


def epsilon_greedy(
    q_values: dict,
    state: Any,
    epsilon: float,
    random_state: int = RANDOM_STATE,
) -> str:
    """ε-greedy action from Q(state, ·)."""
    if not 0.0 <= float(epsilon) <= 1.0:
        raise InvalidParameterError("epsilon must be in [0, 1].")
    rng = Random(random_state)
    if rng.random() < float(epsilon):
        return rng.choice(list(BID_ACTIONS))
    return _best_action(q_values, state)


def ucb_action(
    counts: dict,
    q_values: dict,
    state: Any,
    exploration: float = 2.0,
) -> str:
    """UCB1 over bid/pass for this state."""
    total = sum(int(counts.get((state, action), 0)) for action in BID_ACTIONS)
    best_action = BID_ACTIONS[0]
    best_score = float("-inf")
    for action in BID_ACTIONS:
        n_a = int(counts.get((state, action), 0))
        q = _q_get(q_values, state, action)
        bonus = float(exploration) * math.sqrt(math.log(total + 1) / (n_a + 1))
        score = q + bonus
        if score > best_score:
            best_score = score
            best_action = action
    return best_action


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
    if action not in BID_ACTIONS:
        raise InvalidParameterError(f"action must be one of {BID_ACTIONS}.")
    key = (state, action)
    old = float(q_table.get(key, 0.0))
    if next_state is None:
        bootstrap = 0.0
    else:
        bootstrap = max(_q_get(q_table, next_state, a) for a in BID_ACTIONS)
    q_table[key] = old + float(alpha) * (
        float(reward) + float(gamma) * bootstrap - old
    )
    return q_table


def _player_cost(row: pd.Series) -> float:
    if "cost" in row.index and pd.notna(row["cost"]):
        return float(row["cost"])
    return float(row["pred_value"])


def greedy_by_undervalued(
    shortlist: pd.DataFrame,
    remaining_budget_eur: float,
) -> str:
    """Bid if the top p_undervalued player fits the remaining budget."""
    if not isinstance(shortlist, pd.DataFrame) or shortlist.empty:
        return "pass"
    col = "p_undervalued" if "p_undervalued" in shortlist.columns else "pred_value"
    ordered = shortlist.sort_values(col, ascending=False)
    row = ordered.iloc[0]
    return "bid" if _player_cost(row) <= float(remaining_budget_eur) else "pass"


def _episode_return(
    shortlist: pd.DataFrame,
    budget_eur: float,
    method: str,
    q_table: dict,
    counts: dict,
    rng: Random,
) -> float:
    order = list(shortlist.index)
    rng.shuffle(order)
    budget = float(budget_eur)
    hole = "FWD"
    total = 0.0
    for idx in order:
        row = shortlist.loc[idx]
        state = mdp_state(hole, budget)
        if method == "epsilon_greedy":
            action = epsilon_greedy(q_table, state, epsilon=0.1, random_state=rng.randint(0, 10**9))
        elif method == "ucb":
            action = ucb_action(counts, q_table, state)
        elif method == "q":
            action = epsilon_greedy(q_table, state, epsilon=0.1, random_state=rng.randint(0, 10**9))
        elif method == "greedy":
            action = greedy_by_undervalued(shortlist.loc[[idx]], budget)
        else:
            raise InvalidParameterError(f"Unknown method: {method}")

        cost = _player_cost(row)
        reward = 0.0
        next_state: Any = state
        if action == "bid" and cost <= budget:
            reward = reward_if_signed(row)
            budget -= cost
            next_state = mdp_state(hole, budget)
        # pass or unaffordable bid → reward 0, state unchanged

        if method in {"epsilon_greedy", "ucb", "q"}:
            q_learning_update(q_table, state, action, reward, next_state)
            counts[(state, action)] = int(counts.get((state, action), 0)) + 1
        total += reward
    return total


def simulate_transfer_window(
    shortlist: pd.DataFrame,
    budget_eur: float,
    method: str,
    n_episodes: int = 200,
    random_state: int = RANDOM_STATE,
) -> dict:
    """Roll out ε-greedy, UCB, Q, or greedy. Return mean/std cumulative residual."""
    if method not in {"epsilon_greedy", "ucb", "q", "greedy"}:
        raise InvalidParameterError("method must be epsilon_greedy|ucb|q|greedy.")
    if not isinstance(shortlist, pd.DataFrame) or shortlist.empty:
        raise InvalidParameterError("shortlist must be a non-empty DataFrame.")
    required = {"pred_value"}
    if missing := required - set(shortlist.columns):
        raise InvalidParameterError(f"shortlist missing {sorted(missing)}")

    rng = Random(random_state)
    q_table: dict = {}
    counts: dict = {}
    returns: list[float] = []
    for _ in range(int(n_episodes)):
        returns.append(
            _episode_return(shortlist, budget_eur, method, q_table, counts, rng)
        )
    mean = sum(returns) / len(returns)
    var = sum((value - mean) ** 2 for value in returns) / len(returns)
    return {
        "method": method,
        "mean_return": float(mean),
        "std_return": float(math.sqrt(var)),
        "n_episodes": int(n_episodes),
    }


def compare_policies(results: dict[str, dict]) -> pd.DataFrame:
    """One row per method: mean return, regret vs greedy, n_episodes."""
    if not isinstance(results, dict) or not results:
        raise InvalidParameterError("results must be a non-empty dict.")
    rows = []
    greedy_mean = float(results.get("greedy", {}).get("mean_return", 0.0))
    for method, payload in results.items():
        mean_return = float(payload["mean_return"])
        rows.append(
            {
                "method": method,
                "mean_return": mean_return,
                "std_return": float(payload.get("std_return", 0.0)),
                "n_episodes": int(payload.get("n_episodes", 0)),
                "regret_vs_greedy": greedy_mean - mean_return,
            }
        )
    frame = pd.DataFrame(rows).sort_values("method").reset_index(drop=True)

    figures = OUTPUTS / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    path = figures / "policy_regret.png"
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(frame["method"], frame["mean_return"])
    ax.axhline(greedy_mean, color="black", linestyle="--", label="greedy")
    ax.set_ylabel("mean return")
    ax.set_title("Policy mean return vs greedy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return frame

def _load_q_table() -> dict:
    path = artifact_path("policy")
    if not path.is_file():
        return {}
    return {}  # real parsing lands with export_policy

#def _load_q_table() -> dict:
    path = artifact_path("policy")
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    # Expect {"(state)|action": value} or list of records; keep simple string keys → skip
    # For runtime without export yet, empty table is fine.
    table: dict = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            # keys stored as "repr(state)|action"
            if "|" not in str(key):
                continue
            state_part, action = str(key).rsplit("|", 1)
            try:
                state = eval(state_part, {"__builtins__": {}})  # noqa: S307 — controlled artifact
            except Exception:
                continue
            if action in BID_ACTIONS:
                table[(state, action)] = float(value)
    return table


def decide_bid(
    player_id: int,
    remaining_budget_eur: float,
    position_hole: str,
    q_table: dict | None = None,
) -> dict:
    """Runtime decision for POST /v1/policy/bid."""
    table = q_table if q_table is not None else _load_q_table()
    state = mdp_state(position_hole, remaining_budget_eur)
    action = epsilon_greedy(table, state, epsilon=0.0, random_state=RANDOM_STATE)
    rationale = (
        f"Q-greedy at hole={position_hole!r}, "
        f"budget_bin={state[1]}, action={action}."
    )
    return {
        "action": action,
        "player_id": int(player_id),
        "rationale": rationale,
    }


def export_policy() -> Path:
    """Write `outputs/policy/q_table.json` (pipeline: policy)."""
    not_built("member_c", "export_policy")